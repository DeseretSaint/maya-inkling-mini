// tests/ik2_alspec_test.cpp - CPU-only checks for maya-ik2's ALSpec gated-attention POLICY
// (src/ik2/ik2_alspec.hpp): which blocks one global visit may drop, what it never drops, whether the
// tau_total budget really bounds the dropped mass, and whether the sampled re-read catches a wrong drop.
// No model, no GPU, no kernels - this is the reasoning half of P5-4a; P5-4b measures on the card.
//
//   ./ik2_alspec_test                 the assertions, then the skip-fraction and catch-rate tables
//   ./ik2_alspec_test --eval <file>   replay a recorded gate file (one visit per line, per-block masses)
#include "../src/ik2/ik2_alspec.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::ik2::alspec;

#define CHECK(expr)                                                     \
    do {                                                                \
        if (!(expr)) {                                                  \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr);      \
            return 1;                                                   \
        }                                                               \
    } while (0)

// A block-mass profile with the shape long-context attention is reported to have: a few blocks holding
// most of the mass, the rest decaying geometrically with distance, and no mass outside the context.
static std::vector<float> tail_profile(int n, double top_share, int top_blocks, double ratio) {
    std::vector<float> m((size_t) n, 0.0f);
    double head_sum = 0.0, tail_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const double w = i < top_blocks ? 1.0 : std::pow(ratio, (double) (i - top_blocks));
        m[(size_t) i] = (float) w;
        (i < top_blocks ? head_sum : tail_sum) += w;
    }
    for (int i = 0; i < n; ++i) {
        const double w = m[(size_t) i];
        m[(size_t) i] = (float) (i < top_blocks ? top_share * w / head_sum : (1.0 - top_share) * w / tail_sum);
    }
    return m;
}

// 8 KV heads, logits i.i.d. N(0, sigma), softmax, then the per-block mass of 128 keys summed.  Nothing
// here is measured from the model; it is the null hypothesis a head's sparsity must beat, and the reason
// a sharpened score at depth (the model's log scaling) is expected to make more blocks skippable.
static std::vector<float> gauss_profile(int n_blocks, int keys_per_block, int heads, double sigma, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> m((size_t) n_blocks, 0.0);
    for (int h = 0; h < heads; ++h) {
        std::vector<double> lg((size_t) n_blocks * (size_t) keys_per_block);
        double mx = -1e300;
        for (double& x : lg) {
            x = sigma * nd(rng);
            if (x > mx) mx = x;
        }
        double z = 0.0;
        for (double& x : lg) z += std::exp(x - mx);
        for (int b = 0; b < n_blocks; ++b) {
            double s = 0.0;
            for (int k = 0; k < keys_per_block; ++k) s += std::exp(lg[(size_t) b * (size_t) keys_per_block + (size_t) k] - mx);
            m[(size_t) b] += s / z / (double) heads;
        }
    }
    std::vector<float> out((size_t) n_blocks);
    for (int b = 0; b < n_blocks; ++b) out[(size_t) b] = (float) m[(size_t) b];
    return out;
}

static int print_eval(const char* path) {
    std::vector<std::vector<float>> visits;
    if (!read_gate_file(path, visits)) {
        std::fprintf(stderr, "cannot read %s\n", path);
        return 1;
    }
    Policy p = Policy::from_env();
    if (!p.enabled) p.enabled = true;   // (--eval exists to exercise the policy, so it runs armed)
    std::printf("alspec eval: %zu visit(s), block %d keys, tau %g, tau_total %g, verify 1/%d\n", visits.size(),
                p.block, (double) p.tau, (double) p.tau_total, p.verify_stride);
    double tot_blocks = 0, tot_skipped = 0, tot_mass = 0;
    int tot_sampled = 0;
    for (size_t v = 0; v < visits.size(); ++v) {
        const Plan pl = plan(visits[v], p);
        const std::vector<int> s = sample(pl, (int) v, p);
        std::printf("  visit %-3zu blocks %-5d keep %-5d skip %-5d (%5.1f%%)  dropped mass %.3e  sampled %d\n", v,
                    pl.n_blocks, pl.n_kept, pl.n_skipped, 100.0 * pl.skip_fraction(), pl.skipped_mass, (int) s.size());
        tot_blocks += pl.n_blocks;
        tot_skipped += pl.n_skipped;
        tot_mass += pl.skipped_mass;
        tot_sampled += (int) s.size();
    }
    std::printf("  total: %d blocks, %d skipped (%.1f%%), worst-case dropped mass %.3e, %d block(s) re-read exactly\n",
                (int) tot_blocks, (int) tot_skipped, tot_blocks ? 100.0 * tot_skipped / tot_blocks : 0.0, tot_mass,
                tot_sampled);
    return 0;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--eval")) return print_eval(i + 1 < argc ? argv[i + 1] : "-");

    // ---- the flag is dark by default, and parses when set --------------------------------------
    unsetenv("IK2_ALSPEC");
    unsetenv("IK2_ALSPEC_TAU");
    unsetenv("IK2_ALSPEC_TAU_TOTAL");
    unsetenv("IK2_ALSPEC_VERIFY");
    unsetenv("IK2_ALSPEC_BLOCK");
    Policy d = Policy::from_env();
    CHECK(!d.enabled);
    CHECK(d.tau == 1e-4f && d.tau_total == 1e-2f && d.block == 128 && d.verify_stride == 16);
    setenv("IK2_ALSPEC", "1", 1);
    setenv("IK2_ALSPEC_TAU", "2.5e-5", 1);
    setenv("IK2_ALSPEC_VERIFY", "8", 1);
    Policy e = Policy::from_env();
    CHECK(e.enabled && std::abs(e.tau - 2.5e-5f) < 1e-12f && e.verify_stride == 8);
    unsetenv("IK2_ALSPEC");
    unsetenv("IK2_ALSPEC_TAU");
    unsetenv("IK2_ALSPEC_VERIFY");

    Policy off = d;               // disabled: the mask is all keep, whatever the scores say
    Policy on = d;
    on.enabled = true;

    const int N131 = 1024;        // 131072 keys / 128
    // ---- a uniform visit is not skippable -------------------------------------------------------
    {
        std::vector<float> u((size_t) N131, 1.0f / (float) N131);
        const Plan pl = plan(u, on);
        CHECK(pl.n_skipped == 0 && pl.n_kept == N131);
        const Plan pl_off = plan(u, off);
        CHECK(pl_off.n_skipped == 0);
    }

    // ---- the head and the tail are never dropped ------------------------------------------------
    {
        std::vector<float> z((size_t) 64, 0.0f);   // every block at zero mass, so the gate would drop all
        z[0] = 1.0f;
        const Plan pl = plan(z, on);
        CHECK(pl.skip[0] == 0 && pl.skip[1] == 0);            // keep_first = 2
        CHECK(pl.skip[62] == 0 && pl.skip[63] == 0);          // keep_recent = 2
        // (the structural keeps count towards min_keep: at zero mass everywhere the visit falls exactly to
        //  the floor of 4 blocks rather than being blanked)
        CHECK(pl.n_kept == on.min_keep && pl.n_skipped == 64 - on.min_keep);
        Policy floor5 = on;
        floor5.keep_first = 0;
        floor5.keep_recent = 0;
        floor5.min_keep = 5;
        const Plan pl5 = plan(z, floor5);
        CHECK(pl5.n_kept == 5 && pl5.n_skipped == 59);
    }

    // ---- the tau_total budget bounds what a visit drops -----------------------------------------
    {
        // 1000 blocks at 1e-4 * 0.9 (just under tau): unconstrained the policy would drop all of them,
        // carrying 0.09 of the mass; the budget allows 0.01, so at most 111 of them may go.
        std::vector<float> m((size_t) 1000, 9e-5f);
        const Plan pl = plan(m, on);
        CHECK(pl.skipped_mass <= (double) on.tau_total + 1e-12);
        CHECK(pl.n_skipped <= 112);
        CHECK(pl.n_skipped >= 110);
        // a profile whose tail is far below tau still cannot blow the budget.
        const std::vector<float> g = gauss_profile(512, 128, 8, 1.0, 7);
        const Plan pl2 = plan(g, on);
        CHECK(pl2.skipped_mass <= (double) on.tau_total + 1e-12);
    }

    // ---- the mask is deterministic ---------------------------------------------------------------
    {
        const std::vector<float> g = tail_profile(256, 0.05, 4, 0.95);
        const Plan a = plan(g, on), b = plan(g, on);
        CHECK(a.n_skipped > 0);
        CHECK(a.skip == b.skip && a.skipped_mass == b.skipped_mass);
        const std::vector<int> s1 = sample(a, 3, on), s2 = sample(a, 3, on);
        CHECK(!s1.empty() && s1 == s2);
        CHECK(sample(a, 4, on) != s1);          // (a different visit draws a different sample)
        for (int i : s1) CHECK(a.skip[(size_t) i] == 1);
    }

    // ---- a wrong drop is caught, an unchecked one is not claimed --------------------------------
    {
        const std::vector<float> g = tail_profile(256, 0.05, 4, 0.95);
        const Plan pl = plan(g, on);
        CHECK(pl.n_skipped > 0);
        // 1. exact mass equal to the gate's: nothing failed, and the checked count is the sample size.
        std::vector<float> exact((size_t) pl.n_blocks, std::nanf(""));
        const std::vector<int> s = sample(pl, 0, on);
        for (int i : s) exact[(size_t) i] = g[(size_t) i];
        const Verdict ok = verify(pl, exact, on);
        CHECK(ok.ok() && ok.checked == (int) s.size());
        // 2. one sampled block really carried more than tau: the visit fails and the worst is reported.
        CHECK(!s.empty());
        exact[(size_t) s[0]] = 5.0f * on.tau;
        const Verdict bad = verify(pl, exact, on);
        CHECK(!bad.ok() && bad.failed == 1 && std::abs(bad.worst - 5.0 * on.tau) < 1e-9);
        // 3. an unsampled block is not silently counted as verified.
        std::vector<float> nan_((size_t) pl.n_blocks, std::nanf(""));
        const Verdict none = verify(pl, nan_, on);
        CHECK(none.checked == 0 && none.ok());
    }

    // ---- the sample's catch rate matches its model ----------------------------------------------
    {
        const int stride = on.verify_stride;
        CHECK(std::abs(catch_probability(0, stride) - 0.0) < 1e-12);
        CHECK(std::abs(catch_probability(1, 1) - 1.0) < 1e-12);
        // empirical: drop 200 blocks, mark `bad` of them as gate-underestimated, 4000 visits
        Policy p = on;
        std::vector<float> m((size_t) 260, 0.0f);
        for (int i = 2; i < 258; ++i) m[(size_t) i] = 1e-6f;   // (all droppable, budget covers them all)
        const Plan pl = plan(m, p);
        CHECK(pl.n_skipped > 0);
        std::vector<int> skipped;
        for (int i = 0; i < pl.n_blocks; ++i)
            if (pl.skip[(size_t) i]) skipped.push_back(i);
        for (int bad = 1; bad <= 6; ++bad) {
            int hits = 0, trials = 4000;
            for (int t = 0; t < trials; ++t) {
                const std::vector<int> s = sample(pl, t, p);
                bool hit = false;
                for (int i : s)
                    for (int b = 0; b < bad; ++b)
                        if (i == skipped[(size_t) b]) hit = true;
                hits += hit ? 1 : 0;
            }
            const double want = catch_probability(bad, stride);
            const double got = (double) hits / (double) trials;
            std::printf("  catch: %d bad block(s), stride %d -> predicted %.3f, measured %.3f over %d visits\n", bad,
                        stride, want, got, trials);
            CHECK(std::abs(got - want) < 0.03);
        }
    }

    // ---- what fraction of a 131K visit this policy drops ---------------------------------------
    // 1024 blocks of 128 keys.  The table is the policy's answer for a shape, not a claim about the
    // model: the top_share column is where a head's real mass profile has to sit for the drop to happen.
    std::printf("\n  131K visit = 1024 blocks x 128 keys; tau %g, tau_total %g\n", (double) on.tau,
                (double) on.tau_total);
    std::printf("  %-34s %8s %10s %12s\n", "profile", "skip%", "dropped", "compare with");
    const double mult = 0.1 * std::log(131072.0 / 8192.0);   // the GGUF's log scaling at 131K
    std::printf("  (model log_scale at 131072: 1 + 0.1*ln(131072/8192) = %.4f, i.e. scores x%.3f)\n", 1.0 + mult,
                1.0 + mult);
    for (double top : {0.02, 0.05, 0.10, 0.20, 0.40}) {
        for (double r : {0.99, 0.999}) {
            const std::vector<float> m = tail_profile(N131, top, 4, r);
            const Plan pl = plan(m, on);
            const double ideal = 1.0 - top;   // (the mass outside the 4 head blocks)
            char label[96];
            std::snprintf(label, sizeof label, "top4=%.2f geometric tail r=%.3f (tail %.2f)", top, r, ideal);
            std::printf("  %-34s %7.1f%% %10.3e %12s\n", label, 100.0 * pl.skip_fraction(), pl.skipped_mass,
                        pl.n_skipped ? "mass budget binds" : "nothing under tau");
        }
    }
    std::printf("\n  %-34s %8s %10s %12s\n", "gaussian logits, 8 heads", "skip%", "dropped", "sigma_eff");
    for (double sigma : {1.0, 2.0, 3.0, 4.0, 6.0}) {
        const std::vector<float> m = gauss_profile(N131, 128, 8, sigma, 12345);
        const Plan pl = plan(m, on);
        char label[96];
        std::snprintf(label, sizeof label, "logit sigma %.1f at 4K (x%.2f at 131K)", sigma, 1.0 + mult);
        std::printf("  %-34s %7.1f%% %10.3e %12.2f\n", label, 100.0 * pl.skip_fraction(), pl.skipped_mass,
                    sigma * (1.0 + mult));
    }

    std::puts("\nik2 alspec policy: dark-by-default flag, head/tail/min_keep floor, tau_total budget, "
              "deterministic 1-in-N re-read and its catch rate all passed");
    return 0;
}
