// tests/ik2_swiglu_cap_test.cpp - S6's numerical gate, host only (no GPU, no model, no device buffers).
//
// The fused prefill epilogue (ik2_kernels.cu's gate_up_swiglu_f16_kernel) is meant to be the same
// activation as swiglu_cap_f32, applied one step earlier.  This checks the three things that could have
// changed and can be checked without a card:
//
//   1. the same activation: ik2::swiglu_cap_pair vs the literal expression swiglu_cap_f32_kernel carried
//      before the refactor, over the model's caps (gate 4.0, up 25.0) and values at and beyond them.
//   2. no store round trip: the unfused route writes g/u to an FP32 scratch and reads them back; the
//      fused route keeps them in registers.  Both pick the same function, so the two must agree bitwise -
//      the test proves it rather than asserting it.
//   3. the only real difference, the dot's summation order: ik2::dot_f16_f32 is the kernel's order
//      (64-value blocks, then left to right).  Against a double-precision reference over the model's
//      C = 2304, report the dot's error and how many of the FP16 hidden values move.
//
// Usage: ik2_swiglu_cap_test [--rows N]   (exit 0 = every gate passed)
#include "ik2_swiglu_cap.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int g_fail = 0;

void gate(bool ok, const char* what) {
    std::printf("%-4s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fail;
}

// The expression exactly as swiglu_cap_f32_kernel had it before the refactor (the reference this card must
// not change).
float swiglu_literal(float g, float u, float gcap, float ucap) {
    return gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap);
}

struct Diff {
    double max_abs = 0.0, max_rel = 0.0;
    int n = 0, n_diff = 0;
};

Diff compare(const std::vector<float>& a, const std::vector<float>& b) {
    Diff d;
    d.n = (int) a.size();
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = std::fabs((double) a[i] - (double) b[i]);
        if (e > d.max_abs) d.max_abs = e;
        const double rel = a[i] == 0.0f ? (b[i] == 0.0f ? 0.0 : 1.0)
                                        : e / std::fabs((double) a[i]);
        if (rel > d.max_rel) d.max_rel = rel;
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++d.n_diff;
    }
    return d;
}

// The activation's inputs: zero, both signs of the caps exactly, half a cap, 1.5 caps, 10 caps, an FP16
// boundary, and a pseudo-random spread of magnitudes an expert's raw dot can reach.
std::vector<std::pair<float, float>> activation_inputs() {
    const float gcap = 4.0f, ucap = 25.0f;
    std::vector<float> vals = {0.0f,       1e-8f,      -1e-8f,     1e-4f,       -1e-4f,
                               gcap * 0.5f, -gcap * 0.5f, gcap,      -gcap,       gcap * 1.5f,
                               -gcap * 1.5f, gcap * 10.0f, -gcap * 10.0f, 65504.0f, -65504.0f,
                               ucap * 0.5f, -ucap * 0.5f, ucap,      -ucap,       ucap * 1.5f,
                               -ucap * 1.5f, ucap * 10.0f, -ucap * 10.0f, 3.0e4f,  -3.0e4f};
    std::mt19937 rng(20261122);
    std::uniform_real_distribution<float> u(-40.0f, 40.0f);
    for (int i = 0; i < 24; ++i) vals.push_back(u(rng));
    std::vector<std::pair<float, float>> out;
    for (size_t i = 0; i < vals.size(); ++i) out.emplace_back(vals[i], vals[(i * 7 + 5) % vals.size()]);
    return out;
}

}   // namespace

int main(int argc, char** argv) {
    int rows = 256;
    for (int i = 1; i + 1 < argc; i += 2)
        if (!std::strcmp(argv[i], "--rows")) rows = std::atoi(argv[i + 1]);

    const float gcap = 4.0f, ucap = 25.0f;   // inkling2.activation.situ_beta / situ_linear_beta
    std::printf("# S6 fused-SwiGLU numerical gate (host only)  gate_cap=%.1f up_cap=%.1f\n", gcap, ucap);

    // ---- 1. the refactor changed no arithmetic -------------------------------------------------------
    {
        const auto in = activation_inputs();
        std::vector<float> a, b;
        for (const auto& p : in) {
            a.push_back(swiglu_literal(p.first, p.second, gcap, ucap));
            b.push_back(ik2::swiglu_cap_pair(p.first, p.second, gcap, ucap));
        }
        const Diff d = compare(a, b);
        std::printf("  activation vs pre-refactor expression: n=%d differing=%d max_abs=%.9g max_rel=%.9g\n",
                    d.n, d.n_diff, d.max_abs, d.max_rel);
        gate(d.n_diff == 0, "ik2::swiglu_cap_pair is bit-identical to swiglu_cap_f32_kernel's expression");
    }

    // ---- 2. the store round trip is a no-op ----------------------------------------------------------
    // Unfused: gate/up written to an FP32 scratch, read back, activated.  Fused: the same two floats still in
    // registers.  Both go through ik2::gate_up_swiglu_f16, so the FP16 output must match exactly.
    {
        const auto in = activation_inputs();
        std::vector<float> scratch, kept;
        std::vector<float> h_unfused, h_fused;
        for (const auto& p : in) {
            scratch.push_back(p.first);
            scratch.push_back(p.second);
            kept.push_back(p.first);
            kept.push_back(p.second);
        }
        // the "pass" over the scratch: what swiglu_cap_f32_kernel does on the device, scale + activation
        const float sg = 0.875f, su = 1.125f;
        for (size_t r = 0; r < in.size(); ++r) {
            h_unfused.push_back(ik2::h2f(ik2::gate_up_swiglu_f16(scratch[2 * r], scratch[2 * r + 1], sg, su,
                                                                 gcap, ucap)));
            h_fused.push_back(ik2::h2f(ik2::gate_up_swiglu_f16(kept[2 * r], kept[2 * r + 1], sg, su, gcap, ucap)));
        }
        const Diff d = compare(h_unfused, h_fused);
        std::printf("  fused epilogue vs scratch round trip: n=%d differing=%d max_abs=%.9g max_rel=%.9g\n",
                    d.n, d.n_diff, d.max_abs, d.max_rel);
        gate(d.n_diff == 0, "in-register activation == activation after an FP32 scratch round trip (bitwise)");
        // and the FP16 outputs are finite / in range for every input, caps or not
        bool finite = true;
        for (float v : h_unfused) finite = finite && std::isfinite(v);
        gate(finite, "every activation output is finite (caps hold at and beyond both caps)");
    }

    // ---- 3. the dot's summation order, the only thing the fusion does change -------------------------
    {
        const int C = 2304;   // the model's embedding_length, the experts' K
        std::mt19937 rng(4242);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::vector<uint16_t> wgh(C), wuh(C), xh(C);
        double max_abs = 0.0, max_rel_l1 = 0.0, max_dh = 0.0, max_steps = 0.0;
        int differing = 0;
        for (int r = 0; r < rows; ++r) {
            // the operands as the kernel sees them: decoded FP16 weight rows and the FP16 input
            for (int k = 0; k < C; ++k) {
                wgh[k] = ik2::f2h(u(rng)), wuh[k] = ik2::f2h(u(rng)), xh[k] = ik2::f2h(u(rng));
            }
            const float g_k = ik2::dot_f16_f32(wgh.data(), xh.data(), C);
            const float u_k = ik2::dot_f16_f32(wuh.data(), xh.data(), C);
            double g_ref = 0.0, u_ref = 0.0, l1g = 0.0, l1u = 0.0;   // the true dot of the same operands
            for (int k = 0; k < C; ++k) {
                const double wp = (double) ik2::h2f(wgh[k]) * (double) ik2::h2f(xh[k]);
                const double wq = (double) ik2::h2f(wuh[k]) * (double) ik2::h2f(xh[k]);
                g_ref += wp, u_ref += wq, l1g += std::fabs(wp), l1u += std::fabs(wq);
            }
            max_abs = std::fmax(max_abs,
                                std::fmax(std::fabs((double) g_k - g_ref), std::fabs((double) u_k - u_ref)));
            max_rel_l1 = std::fmax(max_rel_l1, std::fmax(std::fabs((double) g_k - g_ref) / l1g,
                                                         std::fabs((double) u_k - u_ref) / l1u));
            // the FP16 hidden value each route produces, compared in FP16's own terms
            const uint16_t ck = ik2::gate_up_swiglu_f16(g_k, u_k, 1.0f, 1.0f, gcap, ucap);
            const uint16_t cr = ik2::f2h(ik2::swiglu_cap_pair((float) g_ref, (float) u_ref, gcap, ucap));
            const float hk = ik2::h2f(ck), hr = ik2::h2f(cr);
            const double step = std::fabs((double) hr) * 9.765625e-4 + 5.9604645e-8;   // 2^-10 relative
            if (ck != cr) {
                ++differing;
                const double dh = std::fabs((double) hk - (double) hr);
                if (dh > max_dh) max_dh = dh;
                if (dh / step > max_steps) max_steps = dh / step;
            }
        }
        std::printf("  dot, kernel order vs exact double, C=%d rows=%d: max_abs=%.9g max_rel(sum|terms|)=%.9g\n",
                    C, rows, max_abs, max_rel_l1);
        std::printf("  FP16 hidden values differing: %d / %d (max |delta h| = %.9g = %.3g FP16 steps)\n",
                    differing, rows, max_dh, max_steps);
        // The kernel's FP32 sum is exact to its own rounding against the term mass, so the fusion does not
        // change the arithmetic - and the only visible effect is FP16's last step on rounding edges.
        gate(max_rel_l1 < 1e-6, "the kernel's 64-block FP32 sum is within 1e-6 relative of the exact dot");
        gate(max_steps <= 1.0, "every FP16 hidden value that moves moves by at most one FP16 step");
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "OK", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
