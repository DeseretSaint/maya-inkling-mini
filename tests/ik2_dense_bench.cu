// tests/ik2_dense_bench.cu - the decode dense-path speed gate (task t_d75ca41d): for each dense shape the decode
// loop actually issues at T=1 (measured with IK2_GEMM_LOG=1 off the engine, see runs/P4-5-dense-path.md), time the
// three ways the engine can compute one token's product:
//
//   gemm  - hipBLAS f16 x f16 GEMM, F32 accumulate (what ik2_main.cu's gemm16 does today, over a resident f16 copy)
//   mvq8  - ik2::k::mv_q8p: planar Q8_0 (int8 values + FP16 block scales), a warp a row (what the g_dense_q8 fast
//           path uses at T=1)
//   mvraw - ik2::k::mv_blocks over the GGUF's Q8_0 blocks themselves (34 bytes a 32 values)
//
// The point is bytes moved: a Q8_0 matrix is 34/32 = 1.0625 bytes a value against f16's 2, so a T=1 product over
// the blocks reads half the weight bytes - but only wins if the kernel reaches the same fraction of peak bandwidth.
//
//   ./ik2_dense_bench            # every shape, 200 reps
//   ./ik2_dense_bench 500        # 500 reps
#include "../src/ik2/ik2_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace k = ik2::k;

namespace {
int g_fail = 0;
std::mt19937 rng(20261121);

#define CU(x)                                                                                  \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                               \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);      \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)
#define CB(x)                                                                                  \
    do {                                                                                       \
        cublasStatus_t s_ = (x);                                                               \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                                     \
            std::printf("BLAS %s: %d (line %d)\n", #x, (int) s_, __LINE__);                    \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

using cuda_half_t = unsigned short;

float h2f(uint16_t h) {
    __half_raw r;
    r.x = h;
    return __half2float(__half(r));
}
uint16_t f2h(float f) {
    __half_raw r = __half_raw(__float2half(f));
    return r.x;
}

// a random matrix as the GGUF would hold it: Q8_0 blocks of [scale f16][32 int8], values ~ N(0, 1)
struct Q8Mat {
    std::vector<uint8_t> blocks;   // nblk * 34
    int N, K;
    size_t nblk() const { return (size_t) N * K / 32; }
};

Q8Mat make_q8(int N, int K) {
    Q8Mat m;
    m.N = N, m.K = K;
    m.blocks.resize(m.nblk() * 34);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    for (size_t b = 0; b < m.nblk(); ++b) {
        float v[32], amax = 0;
        for (int i = 0; i < 32; ++i) v[i] = nd(rng), amax = std::max(amax, std::fabs(v[i]));
        const float d = amax / 127.0f;
        const uint16_t dh = f2h(d);
        std::memcpy(&m.blocks[b * 34], &dh, 2);
        const float dd = h2f(dh);
        for (int i = 0; i < 32; ++i) {
            int q = (int) std::lround(v[i] / (dd ? dd : 1.0f));
            q = std::max(-128, std::min(127, q));
            m.blocks[b * 34 + 2 + i] = (uint8_t) (int8_t) q;
        }
    }
    return m;
}

void planar(const Q8Mat& m, std::vector<int8_t>& q, std::vector<uint16_t>& d) {
    q.resize((size_t) m.N * m.K);
    d.resize(m.nblk());
    for (size_t b = 0; b < m.nblk(); ++b) {
        std::memcpy(&d[b], &m.blocks[b * 34], 2);
        std::memcpy(&q[b * 32], &m.blocks[b * 34 + 2], 32);
    }
}

struct Dev {
    cuda_half_t* f16 = nullptr;
    int8_t* q = nullptr;
    uint16_t* d = nullptr;
    uint8_t* raw = nullptr;
    cuda_half_t* x16 = nullptr;
    float* x32 = nullptr;
    float* y = nullptr;
    size_t cap = 0;
};

Dev alloc(int N, int K, const Q8Mat& m, const std::vector<int8_t>& qh, const std::vector<uint16_t>& dh) {
    Dev v;
    v.cap = (size_t) N * K;
    std::vector<cuda_half_t> h(v.cap);
    for (size_t i = 0; i < v.cap; ++i) h[i] = f2h(h2f(dh[i >> 5]) * (float) qh[i]);
    CU(cudaMalloc(&v.f16, v.cap * 2));
    CU(cudaMemcpy(v.f16, h.data(), v.cap * 2, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&v.q, v.cap));
    CU(cudaMemcpy(v.q, qh.data(), v.cap, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&v.d, dh.size() * 2));
    CU(cudaMemcpy(v.d, dh.data(), dh.size() * 2, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&v.raw, m.blocks.size()));
    CU(cudaMemcpy(v.raw, m.blocks.data(), m.blocks.size(), cudaMemcpyHostToDevice));
    std::vector<cuda_half_t> xh(K);
    std::vector<float> xf(K);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int i = 0; i < K; ++i) xf[i] = nd(rng), xh[i] = f2h(xf[i]);
    CU(cudaMalloc(&v.x16, K * 2));
    CU(cudaMemcpy(v.x16, xh.data(), K * 2, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&v.x32, K * 4));
    CU(cudaMemcpy(v.x32, xf.data(), K * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&v.y, (size_t) N * 4));
    return v;
}

void free_dev(Dev& v) {
    cudaFree(v.f16), cudaFree(v.q), cudaFree(v.d), cudaFree(v.raw), cudaFree(v.x16), cudaFree(v.x32),
        cudaFree(v.y);
}

struct Res {
    double us = 0, gbs = 0;
    std::vector<float> y;
};

// time `reps` launches (one stream, one sync at the end), and keep the output of the first
template <class F>
Res timed(int reps, double bytes, F launch, int N) {
    cudaStream_t s;
    CU(cudaStreamCreate(&s));
    launch(s);   // warm
    CU(cudaStreamSynchronize(s));
    cudaEvent_t a, b;
    CU(cudaEventCreate(&a));
    CU(cudaEventCreate(&b));
    CU(cudaEventRecord(a, s));
    for (int i = 0; i < reps; ++i) launch(s);
    CU(cudaEventRecord(b, s));
    CU(cudaEventSynchronize(b));
    float ms = 0;
    CU(cudaEventElapsedTime(&ms, a, b));
    Res r;
    r.us = ms * 1e3 / reps;
    r.gbs = bytes / (r.us * 1e-6) / 1e9;
    r.y.resize(N);
    CU(cudaEventDestroy(a));
    CU(cudaEventDestroy(b));
    CU(cudaStreamDestroy(s));
    return r;
}

float rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        num += (double) (a[i] - b[i]) * (a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return (float) std::sqrt(num / std::max(1e-30, den));
}
}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 200;
    cublasHandle_t blas;
    CB(cublasCreate(&blas));
    // the engine's gemm16: CUBLAS_OP_T (W is [N][K] row-major), F32 sums, default heuristic
    auto gemm16 = [&](cublasHandle_t b, const cuda_half_t* W, int N, int K, const cuda_half_t* X, int T, float* Y,
                      int ldy) {
        const float one = 1.0f, zero = 0.0f;
        CB(cublasGemmEx(b, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, W, CUDA_R_16F, K, X, CUDA_R_16F, K, &zero, Y,
                        CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    };

    // the shapes decode issues at T=1, per token (IK2_GEMM_LOG=1 on the 128-token run)
    struct Shape {
        int N, K, per_token;
        const char* what;
    } shapes[] = {
        {8600, 2304, 20, "sliding qkvrg+sink"},
        {6552, 2304, 8, "full qrg+sink, fused gate"},
        {3480, 2304, 20, "full qrg+sink, fused q"},
        {3072, 2304, 20, "full wq"},
        {1056, 2304, 16, "wk/wv, short-conv layer"},
        {1024, 2304, 40, "wk / wv"},
        {2304, 3072, 48, "attn_output (wo)"},
        {128, 2304, 48, "router"},
        {201024, 2304, 1, "head (do not target)"},
    };

    std::printf("%-24s %6s %6s %6s | %9s %8s | %9s %8s | %9s %8s | %8s %8s\n", "shape", "N", "K", "per",
                "gemm us", "GB/s", "mvq8 us", "GB/s", "mvraw us", "GB/s", "rel mvq8", "rel raw");
    double tot_g = 0, tot_q = 0, tot_r = 0, tok_g = 0, tok_q = 0, tok_r = 0;
    for (auto& sh : shapes) {
        const int N = sh.N, K = sh.K;
        Q8Mat m = make_q8(N, K);
        std::vector<int8_t> qh;
        std::vector<uint16_t> dh;
        planar(m, qh, dh);
        Dev v = alloc(N, K, m, qh, dh);
        const double bytes_f16 = (double) N * K * 2, bytes_q8 = (double) N * K * 34.0 / 32.0;

        Res g = timed(reps, bytes_f16, [&](cudaStream_t s) {
            cublasSetStream(blas, s);
            gemm16(blas, v.f16, N, K, v.x16, 1, v.y, N);
        }, N);
        CU(cudaMemcpy(g.y.data(), v.y, (size_t) N * 4, cudaMemcpyDeviceToHost));

        Res q = timed(reps, bytes_q8, [&](cudaStream_t s) { k::mv_q8p(v.q, v.d, N, K, v.x32, v.y, s); }, N);
        CU(cudaMemcpy(q.y.data(), v.y, (size_t) N * 4, cudaMemcpyDeviceToHost));

        Res r = timed(reps, bytes_q8, [&](cudaStream_t s) { k::mv_blocks(k::kQ8_0, v.raw, N, K, v.x32, v.y, s); }, N);
        CU(cudaMemcpy(r.y.data(), v.y, (size_t) N * 4, cudaMemcpyDeviceToHost));

        const float e1 = rel_err(q.y, g.y), e2 = rel_err(r.y, g.y);
        std::printf("%-24s %6d %6d %6d | %9.1f %8.0f | %9.1f %8.0f | %9.1f %8.0f | %8.2e %8.2e\n", sh.what, N, K,
                    sh.per_token, g.us, g.gbs, q.us, q.gbs, r.us, r.gbs, e1, e2);
        tot_g += g.us * sh.per_token, tot_q += q.us * sh.per_token, tot_r += r.us * sh.per_token;
        if (N != 201024) {   // the card says: output projection is not a target
            tok_g += g.us * sh.per_token, tok_q += q.us * sh.per_token, tok_r += r.us * sh.per_token;
        }
        free_dev(v);
    }
    std::printf("\nper token, the shapes decode issues (us, summed over the calls a token makes):\n");
    std::printf("  all      : gemm %.0f  mvq8 %.0f  mvraw %.0f\n", tot_g, tot_q, tot_r);
    std::printf("  no head  : gemm %.0f  mvq8 %.0f  mvraw %.0f\n", tok_g, tok_q, tok_r);
    std::printf("  a %.1f ms/token decode: gemm %.1f%% of it -> mvq8 %.1f%%, mvraw %.1f%%\n", 19.5, tot_g / 19500 * 100,
                tot_q / 19500 * 100, tot_r / 19500 * 100);
    std::printf("%s\n", g_fail ? "FAILURES" : "all timed");
    CB(cublasDestroy(blas));
    return g_fail ? 1 : 0;
}
