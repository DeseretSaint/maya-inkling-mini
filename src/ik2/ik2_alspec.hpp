// src/ik2/ik2_alspec.hpp - ALSpec-style gated attention, the POLICY half only: which key blocks a global
// visit may drop, which dropped blocks are re-read on the exact path, and how a wrong drop is caught.
// (No kernels, no device memory, no CUDA / HIP names - this header is host code and is compiled by the
//  engine and by tests/ik2_alspec_test.cpp in both a GPU and a CPU-only build.  What is missing here is
//  deliberate: the gate score per block comes from a probe kernel that does not exist in this build, so
//  IK2_ALSPEC=1 arms the policy and the engine still takes the exact path - P5-4b adds the kernel.)
//
// The rule, as ALSpec states it: a score is computed cheaply for every KV block, and a block whose mass
// is below tau may be skipped rather than read.  Two guards make the skip exact-or-conservative instead
// of merely fast:
//   * budget     - the blocks dropped in one visit may carry at most tau_total of the probability mass,
//                  so many small drops cannot add up to a material change (default 1% of the mass).
//   * structure  - the first keep_first blocks (sink / first tokens) and the last keep_recent blocks
//                  (the local window the query is about) are never dropped, and min_keep blocks survive
//                  in any case.  Dropping them is the failure mode that breaks retrieval needles.
//   * sample     - a deterministic 1-in-verify_stride sample of the dropped blocks is re-read exactly and
//                  compared against tau; any sampled block above tau fails the visit, which falls back to
//                  the exact path.  The sample is hash-chosen from (visit, block, seed), so it is stable
//                  across runs (a failing visit is reproducible) and independent of the gate score (it is
//                  not biased towards blocks the gate already likes).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata {
namespace ik2 {
namespace alspec {

inline float env_f(const char* name, float dflt) {
    const char* v = std::getenv(name);
    if (!v || !*v) return dflt;
    return (float) std::atof(v);
}

inline int env_i(const char* name, int dflt) {
    const char* v = std::getenv(name);
    if (!v || !*v) return dflt;
    return std::atoi(v);
}

struct Policy {
    bool enabled = false;          // IK2_ALSPEC, default 0: off, and nothing below is reached
    float tau = 1e-4f;             // IK2_ALSPEC_TAU: per-block mass under which a block may be dropped
    float tau_total = 1e-2f;       // IK2_ALSPEC_TAU_TOTAL: the mass one visit's drops may carry, in total
    int block = 128;               // IK2_ALSPEC_BLOCK: keys per block (the kernels split in 64-key halves)
    int keep_first = 2;            // IK2_ALSPEC_KEEP_FIRST: leading blocks that are never dropped
    int keep_recent = 2;           // IK2_ALSPEC_KEEP_RECENT: trailing blocks that are never dropped
    int min_keep = 4;              // IK2_ALSPEC_MIN_KEEP: blocks a visit always keeps
    int verify_stride = 16;        // IK2_ALSPEC_VERIFY: 1 of every N dropped blocks is re-read exactly
    uint64_t seed = 0x9e3779b97f4a7c15ull;   // IK2_ALSPEC_SEED: the sample's hash seed

    static Policy from_env() {
        Policy p;
        p.enabled = env_i("IK2_ALSPEC", 0) != 0;
        p.tau = env_f("IK2_ALSPEC_TAU", p.tau);
        p.tau_total = env_f("IK2_ALSPEC_TAU_TOTAL", p.tau_total);
        p.block = env_i("IK2_ALSPEC_BLOCK", p.block);
        p.keep_first = env_i("IK2_ALSPEC_KEEP_FIRST", p.keep_first);
        p.keep_recent = env_i("IK2_ALSPEC_KEEP_RECENT", p.keep_recent);
        p.min_keep = env_i("IK2_ALSPEC_MIN_KEEP", p.min_keep);
        p.verify_stride = env_i("IK2_ALSPEC_VERIFY", p.verify_stride);
        if (const char* s = std::getenv("IK2_ALSPEC_SEED"); s && *s) p.seed = std::strtoull(s, nullptr, 10);
        p.clamp();
        return p;
    }

    // (A mis-set env var must not be able to blank a visit or to make the sample empty of self-check.)
    void clamp() {
        if (!(tau > 0.0f)) tau = 1e-4f;
        if (!(tau_total >= 0.0f)) tau_total = 1e-2f;
        if (tau_total > 0.25f) tau_total = 0.25f;
        if (block < 16) block = 16;
        if (keep_first < 0) keep_first = 0;
        if (keep_recent < 0) keep_recent = 0;
        if (min_keep < 1) min_keep = 1;
        if (verify_stride < 1) verify_stride = 1;
    }

    /// The blocks a span of nk keys covers: ceil(nk / block).
    int blocks_for(int nk) const { return nk <= 0 ? 0 : (nk + block - 1) / block; }
};

struct Plan {
    int n_blocks = 0;
    int n_kept = 0;
    int n_skipped = 0;
    double skipped_mass = 0.0;      // (the gate's own accounting: what the drops are believed to carry)
    std::vector<uint8_t> skip;      // 1 = this block is not read on the cheap path

    /// Fraction of the blocks the visit no longer reads, 0 when nothing is dropped.
    double skip_fraction() const { return n_blocks ? (double) n_skipped / (double) n_blocks : 0.0; }
};

/// Decide one visit's drop mask from its per-block gate mass (assumed normalized per visit, sum <= 1).
/// The mask is a subset of {blocks with gate mass < tau}, chosen smallest-first until the tau_total
/// budget is spent, with the head, the tail and min_keep blocks unconditionally kept.
inline Plan plan(const std::vector<float>& mass, const Policy& p) {
    Plan r;
    const int n = (int) mass.size();
    r.n_blocks = n;
    r.skip.assign((size_t) std::max(0, n), 0);
    if (!p.enabled || n <= 0) {
        r.n_kept = std::max(0, n);
        return r;
    }
    const int keep_head = std::min(std::max(0, p.keep_first), n);
    const int keep_tail = std::min(std::max(0, p.keep_recent), n - keep_head);
    const int min_keep = std::min(n, std::max(1, p.min_keep));

    std::vector<int> cand;
    cand.reserve((size_t) n);
    for (int i = 0; i < n; ++i) {
        if (i < keep_head || i >= n - keep_tail) continue;
        const float m = mass[(size_t) i];
        if (!(m < p.tau)) continue;   // (NaN or >= tau: kept)
        cand.push_back(i);
    }
    std::stable_sort(cand.begin(), cand.end(),
                     [&](int a, int b) { return mass[(size_t) a] < mass[(size_t) b]; });

    double dropped = 0.0;
    for (int i : cand) {
        if (n - r.n_skipped - 1 < min_keep) break;            // never leave a visit with fewer than min_keep
        const double m = std::max(0.0, (double) mass[(size_t) i]);
        if (dropped + m > (double) p.tau_total) continue;     // the budget is spent: keep the rest
        r.skip[(size_t) i] = 1;
        ++r.n_skipped;
        dropped += m;
    }
    r.n_kept = n - r.n_skipped;
    r.skipped_mass = dropped;
    return r;
}

inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

/// Is this dropped block in the verification sample? (A function of the visit, the block and the seed
/// alone, so the same visit re-checks the same blocks on a re-run.)
inline bool sampled(int block_index, int visit, const Policy& p) {
    if (p.verify_stride <= 1) return true;
    const uint64_t h = splitmix64(splitmix64((uint64_t) (uint32_t) block_index ^ p.seed) ^
                                  (uint64_t) (uint32_t) visit);
    return h % (uint64_t) p.verify_stride == 0;
}

/// The dropped blocks to re-read on the exact path, in block order.
inline std::vector<int> sample(const Plan& pl, int visit, const Policy& p) {
    std::vector<int> out;
    for (int i = 0; i < pl.n_blocks; ++i)
        if (pl.skip[(size_t) i] && sampled(i, visit, p)) out.push_back(i);
    return out;
}

struct Verdict {
    int checked = 0;    // sampled blocks whose exact mass was computed
    int failed = 0;     // sampled blocks whose exact mass reached tau: the gate under-read them
    double worst = 0.0; // the largest exact mass among the sampled drops
    bool ok() const { return failed == 0; }
};

/// Compare the sampled drops against their exact mass.  `exact` is indexed like `mass`; a NaN entry
/// means that block was not recomputed (not sampled), and is not counted as checked.  A visit that
/// fails here must be taken on the exact path, not reported as a speed win.
inline Verdict verify(const Plan& pl, const std::vector<float>& exact, const Policy& p) {
    Verdict v;
    const int n = std::min(pl.n_blocks, (int) exact.size());
    for (int i = 0; i < n; ++i) {
        if (!pl.skip[(size_t) i]) continue;
        const float m = exact[(size_t) i];
        if (!(m == m)) continue;                     // (NaN = not sampled)
        ++v.checked;
        if (m > v.worst) v.worst = (double) m;
        if (m >= p.tau) ++v.failed;
    }
    return v;
}

/// P(at least one of `bad` independently gate-underestimated drops lands in the sample).  With the
/// 1/stride hash sample each dropped block is drawn independently, so this is the exact model; a visit
/// with one bad block and stride 16 is still missed 15 times in 16, which is why the gate's own
/// accounting (skipped_mass <= tau_total) carries the correctness argument and the sample only catches
/// gross regressions.
inline double catch_probability(int bad, int stride) {
    if (bad <= 0) return 0.0;
    if (stride <= 1) return 1.0;
    return 1.0 - std::pow(1.0 - 1.0 / (double) stride, (double) bad);
}

/// Read a recorded gate file: one visit per line, whitespace- or comma-separated per-block masses,
/// '#' to end of line a comment.  This is the format P5-4b's probe kernel writes and the engine (or
/// tests/ik2_alspec_test --eval) replays on the CPU, so the policy can be exercised with no GPU.
inline bool read_gate_file(const std::string& path, std::vector<std::vector<float>>& visits) {
    visits.clear();
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::string line;
    char buf[1 << 16];
    while (std::fgets(buf, (int) sizeof buf, f)) {
        line.assign(buf);
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        std::vector<float> row;
        const char* s = line.c_str();
        char* end = nullptr;
        while (true) {
            while (*s == ' ' || *s == '\t' || *s == ',' || *s == '\r' || *s == '\n') ++s;
            if (!*s) break;
            const double v = std::strtod(s, &end);
            if (end == s) break;
            row.push_back((float) v);
            s = end;
        }
        if (!row.empty()) visits.push_back(std::move(row));
    }
    std::fclose(f);
    return true;
}

}  // namespace alspec
}  // namespace ik2
}  // namespace strata
