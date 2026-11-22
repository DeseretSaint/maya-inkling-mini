// tests/ik2_fp8_parity.cu - IK2_FP8_FAST's parity gate on the card: all 256 E4M3FN byte values through every
// conversion route in src/ik2/ik2_kernels.cu -- the scalar path that shipped before the fast path (the
// reference), the path this build's kernels call, the 256-entry LUT, two elements at a time, and the float4
// read helper -- compared bit for bit. One block, one launch: the smallest GPU call that can produce this
// evidence, so it can be run beside a full test pass without loading a model.
#include "../src/ik2/ik2_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef IK2_FP8_FAST
#define IK2_FP8_FAST 0
#endif

namespace k = ik2::k;

namespace {
int g_fail = 0;

#define CU(x)                                                                                  \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                               \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);      \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

void expect(const char* what, bool ok) {
    std::printf("%-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

float h2f_host(uint16_t h) {   // the half the scalar path produces, as a float (device: __half2float)
    const uint32_t sign = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 0) {
        if (m == 0) {
            u = sign;
        } else {
            int sh = 0;
            uint32_t mm = m;
            while (!(mm & 1024)) mm <<= 1, ++sh;
            u = sign | ((uint32_t) (113 - sh) << 23) | ((mm & 1023) << 13);
        }
    } else if (e == 31) {
        u = sign | 0x7f800000u | (m << 13);
    } else {
        u = sign | ((e + 112) << 23) | (m << 13);
    }
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}
}  // namespace

int main() {
    std::printf("fp8 parity: IK2_FP8_FAST=%d\n", (int) IK2_FP8_FAST);
    std::vector<uint8_t> all_b(256);
    for (int i = 0; i < 256; ++i) all_b[i] = (uint8_t) i;
    uint8_t* db = nullptr;
    uint16_t *dsc = nullptr, *dsel = nullptr, *dlut = nullptr, *dx2 = nullptr;
    float* dload = nullptr;
    CU(cudaMalloc(&db, 256));
    CU(cudaMemcpy(db, all_b.data(), 256, cudaMemcpyHostToDevice));
    for (uint16_t** p : {&dsc, &dsel, &dlut, &dx2}) CU(cudaMalloc(p, 256 * sizeof(uint16_t)));
    CU(cudaMalloc(&dload, 256 * sizeof(float)));
    k::fp8_cvt_selftest(db, dsc, dsel, dlut, dx2, dload, 0);
    CU(cudaDeviceSynchronize());

    std::vector<uint16_t> sc(256), sel(256), lut(256), x2(256);
    std::vector<float> load(256);
    CU(cudaMemcpy(sc.data(), dsc, 256 * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(sel.data(), dsel, 256 * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(lut.data(), dlut, 256 * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(x2.data(), dx2, 256 * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(load.data(), dload, 256 * sizeof(float), cudaMemcpyDeviceToHost));

    bool sel_ok = true, lut_ok = true, x2_ok = true, load_ok = true;
    for (int b = 0; b < 256; ++b) {
        sel_ok = sel_ok && sel[b] == sc[b];
        lut_ok = lut_ok && lut[b] == sc[b];
        x2_ok = x2_ok && x2[b] == sc[b];
        if (sel[b] != sc[b] || x2[b] != sc[b])
            std::printf("  byte 0x%02X: scalar 0x%04X, selected 0x%04X, x2 0x%04X\n", b, sc[b], sel[b], x2[b]);
    }
    for (int b = 0; b < 256; ++b) {
        const uint16_t h = sc[b];
        const bool nan = (h & 0x7C00) == 0x7C00 && (h & 0x03FF) != 0;
        const bool ok = nan ? std::isnan(load[b]) : (load[b] == h2f_host(h));
        if (!ok) std::printf("  byte 0x%02X: float4 helper %.9g, scalar half 0x%04X\n", b, (double) load[b], h);
        load_ok = load_ok && ok;
    }
    expect("selected path == scalar, all 256 bytes", sel_ok);
    expect("LUT == scalar, all 256 bytes", lut_ok);
    expect("two-at-a-time == scalar, all 256 bytes", x2_ok);
    expect("float4 read helper == scalar, all 256 bytes", load_ok);
    cudaFree(db);
    for (uint16_t* p : {dsc, dsel, dlut, dx2}) cudaFree(p);
    cudaFree(dload);
    std::printf("%s\n", g_fail ? "fp8 parity: FAIL" : "fp8 parity: all ok");
    return g_fail ? 1 : 0;
}
