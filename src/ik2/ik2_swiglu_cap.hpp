// src/ik2/ik2_swiglu_cap.hpp - the cap-aware SwiGLU, in one place.
//
// The expert epilogue is applied in two shapes: after the gate/up product has been written to memory
// (swiglu_cap_f32 / swiglu_cap) and, on the fused path, in registers before the store.  Both must be the
// same arithmetic, so the expression lives here and every kernel calls it.  Included by ik2_kernels.cu
// (device and host passes) and by tests/ik2_swiglu_cap_test.cpp (host only, no GPU needed).
#ifndef IK2_SWIGLU_CAP_HPP
#define IK2_SWIGLU_CAP_HPP

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <cuda_fp16.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define IK2_HD __host__ __device__
#else
#define IK2_HD
#endif

namespace ik2 {

// The model's activation: cap-aware SwiGLU.  gate_cap caps the gate's magnitude through tanh, up_cap the
// up projection's, and the gate also carries the logistic factor.  The order of the multiplications is
// the one the engine has always used - keep it, a reassociation changes the float rounding.
IK2_HD inline float swiglu_cap_pair(float g, float u, float gcap, float ucap) {
    return gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap);
}

// The two conversions the epilogue needs, identical on both passes so a host test can compare them.
IK2_HD inline float h2f(uint16_t h) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __half2float(__ushort_as_half(h));
#else
    _Float16 v;
    std::memcpy(&v, &h, 2);
    return (float) v;
#endif
}

IK2_HD inline uint16_t f2h(float f) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __half_as_ushort(__float2half_rn(f));
#else
    const _Float16 v = (_Float16) f;   // round to nearest even, as __float2half_rn
    uint16_t h;
    std::memcpy(&h, &v, 2);
    return h;
#endif
}

// One hidden value from the raw gate/up sums: the expert's scales, then the activation, then FP16.
// The unfused path calls this after reading the FP32 scratch back; the fused epilogue calls it on the
// accumulators it still holds, so the only difference between the two paths is where the values lived.
IK2_HD inline uint16_t gate_up_swiglu_f16(float g_raw, float u_raw, float sg, float su, float gcap,
                                          float ucap) {
    return f2h(swiglu_cap_pair(g_raw * sg, u_raw * su, gcap, ucap));
}

// The gate/up dot the fused prefill kernel accumulates: FP16 operands, FP32 sum, in 64-value blocks and
// then left to right, so a host test can reproduce the kernel's summation order exactly.
IK2_HD inline float dot_f16_f32(const uint16_t* w, const uint16_t* x, int n) {
    float acc = 0.0f;
    int k = 0;
    for (; k + 64 <= n; k += 64) {
        float blk = 0.0f;
        for (int j = 0; j < 64; ++j) blk += h2f(w[k + j]) * h2f(x[k + j]);
        acc += blk;
    }
    for (; k < n; ++k) acc += h2f(w[k]) * h2f(x[k]);
    return acc;
}

}   // namespace ik2

#undef IK2_HD

#endif   // IK2_SWIGLU_CAP_HPP
