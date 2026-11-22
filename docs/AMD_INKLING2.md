# Inkling 2 Mini on RDNA4: the HIP port

The Inkling engine (`src/ik2`, `maya-ik2`) was written for CUDA. This is what it takes to build
and run it on an RDNA4 card, what the matrix cores give, and what a kernel-level profile of the
decode says the remaining work is. Measured on one **RX 9070 XT (gfx1201, 16 GB)** with ROCm
10.2.0 from the AMD wheel index, clang from the same SDK, g++ 16.2.1. Nothing here was measured on
an NVIDIA card or on gfx1100/gfx1151.

## Build

```sh
R=$(realpath .rocm/lib/python3.12/site-packages/_rocm_sdk_devel)   # the wheel SDK
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_HIP_IK2_SPIKE=ON \
  -DAMDGPU_TARGETS=gfx1201 -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_HIP_COMPILER="$R/llvm/bin/clang++" -DCMAKE_PREFIX_PATH="$R"
cmake --build build-hip --target maya-ik2 ik2_kernels_test ik2_expert_bench -j
```

`STRATA_HIP_IK2_SPIKE` (`OFF` by default) adds `ik2_kernels`, `maya-ik2`, `ik2_kernels_test` and
`ik2_expert_bench` to a HIP build; without it an AMD build is the GLM/DeepSeek/MiniMax engines
only. `STRATA_PREFILL_MMQ=ON` is required by any HIP build (the GLM prompt path).

## What the port needed

Four things, in the order they fail:

1. **`cuda_fp8.h` did not exist in `include/strata/hip_compat/`.** `hip_fp8.h` exists in hipcc; the
   shim now maps `__nv_fp8_storage_t`, `__nv_fp8x2_storage_t`, `__NV_E4M3`/`__NV_E5M2` and the
   `__nv_cvt_*` conversions onto it.
2. **`__shfl_*_sync` macros defined by a source break the HIP headers.** A source that defines
   `__shfl_down_sync` before the HIP headers get their own declarations macro-mangled ("out-of-line
   definition does not match any declaration"). `intrinsics.hpp` now includes `hip_fp16.h`,
   `hip_bf16.h` and `hip_fp8.h` first, which fixes the order for every HIP source.
3. **Managed memory and capture modes were unmapped.** `cudaMallocManaged`, `cudaMemPrefetchAsync`,
   `cudaMemAdvise` with its advice enums, `cudaCpuDeviceId`, the `cudaMemLocation*` types of CUDA 13
   and `cudaStreamCaptureMode*` all have to be in `cuda_runtime.h` for the tiered expert cache and
   the graph path to compile. `tools/check_hip_compat.py` now passes over the tree — it reported 38
   unmapped names before, all of them in `src/ik2`.
4. **The flash-attention block used PTX.** One block (`mma.sync`, `ldmatrix`, `cp.async`, the FP8
   conversions) was the whole compile break, plus three runtime facts:
   - **the K/V tile buffers alternate** (`buf = (t - t0) & 1`) while gfx12 allocates one
     (`kFaBufs = 1`): every tile after the first reads unwritten LDS and exactly half the outputs
     are wrong. Use `& (kFaBufs - 1)`.
   - **`hipFuncSetAttribute(MaxDynamicSharedMemorySize, 53248)` returns `invalid argument`** on this
     card and the kernel never launches (53,248 B is under the 65,536 B default); the call is
     CUDA-only now. The kernel needs 53,248 B: 34,816 K/V + 18,432 bias.
   - the gfx12 matrix-core layout differs from `mma.sync`'s, so the fragments are loaded for wave32
     (see below) rather than staged through `ldmatrix`.

## The matrix-core path

`IK2_EXPERT_WMMA=1` (default) sends the expert GEMMs to `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`
on gfx1200/gfx1201; `=0` forces the emulated `m16n8k16` (shuffles + FP16-exact FP32 FMA) that every
other architecture runs. `IK2_FA_WMMA` does the same for the three flash-attention kernels (prefill,
one-token decode, drafter window); those are templated on the product choice with both
instantiations compiled, so the choice costs no branch in the loops.

Wave32 gfx12 fragment layout, confirmed with a basis probe rather than a comment: lane *l* holds
`A[l % 16][8 (l / 16) + j]`, `B[8 (l / 16) + j][l % 16]`, `D[8 (l / 16) + i][l % 16]`. A lane
therefore reads 8 *consecutive* k of one A row and one B row - no shuffles and no LDS staging.

An expert group is at most 8 tokens, so the 16x16x16 tile pads N with zero columns and guards every
write; the gate keeps 16 meaningful rows by holding 8 gate rows then the same 8 of up, which is why
the SwiGLU needs no cross-lane exchange. NVFP4's 36-byte block (4 ue4m3 scale bytes + 32 code bytes)
puts sub-block *t*'s 16 values in byte *i*'s low nibble (*i*) and high nibble (*i* + 8): a lane half
takes the low or high nibbles of all 8 bytes, not 4 consecutive bytes. Getting that wrong yields
plausible-looking garbage about 2x off, not a crash.

## Expert GEMM, both paths

`./build-hip/ik2_expert_bench {gate|down} <iters> <dump.bin>`, `IK2_GROUP_MMA=3`, 128 groups of 8
tokens, C = 2304, F = 1152, 200 iterations, median:

| Path | gate+up | down |
| --- | --- | --- |
| `IK2_EXPERT_WMMA=1` (matrix core) | 717.6 us (666.5 best), 15,150 GFLOP/s | 348.3 us (336.7 best), 15,606 GFLOP/s |
| `IK2_EXPERT_WMMA=0` (emulated) | 1,633.6 us (1,611.8 best), 6,655 GFLOP/s | 979.6 us (918.6 best), 5,549 GFLOP/s |

**2.4x gate+up, 2.7x down.** The dumps: **down is bit-identical** (FP32 sums, only the summation
order differs) and gate+up differs on 0.31 % of the bench's synthetic values (median relative
4e-3). The gate input is rounded to FP16 for the product, and this bench's input scale is not the
model's; the kernel suite and the engine's greedy ids are the gates that matter, and both pass (see
below). The Q8_0 grouped variants stay emulated on purpose: their scale is applied *after* the FP32
dot, so a WMMA tile would have to accumulate a 32-value block at zero and scale-add it.

## Flash attention, both arms

`--cache-gb 12`, `IK2_KV=fp8`, 32 generated tokens, 2 repetitions each, same run order:

| prompt | `IK2_FA_WMMA=1` prefill | `=0` prefill | `=1` decode | `=0` decode |
| --- | --- | --- | --- | --- |
| 512 tokens | 209.2 / 208.4 tok/s | 205.6 / 107.5 tok/s | 33.84 / 33.86 | 33.77 / 33.81 |
| 16,384 tokens | 1,190.2 / 956.3 tok/s | 910.1 / 907.0 tok/s | 43.09 / 43.13 | 43.30 / 43.29 |

The FA product is a **prefill lever, not a decode one**: the 512-token rows are inside run-to-run
noise (one emulated repetition was disturbed from outside the engine), at 16K the matrix core buys
1.05-1.31x prefill, and decode does not move. `IK2_PROF=1` at 131K depth puts attention at 0.81 s of
a 1.43 s 31-token decode (57 %) while the product swap moves the total 2.5 %: decode attention time
is in the K/V reads and the softmax/merge, not in the 16x16x16 products.

Greedy ids are not a parity oracle between the two arms - the products agree only to FP16 round-off
(`blk.0.attn_out` max 4.4e-4 at scale 17.7) and 48 layer visits amplify it. Here the two arms did
produce the same 32 ids at both prompt sizes, and each arm reproduces its own ids exactly; the real
gate is `ik2_kernels_test` in both flag states.

## Verification

`build-hip/ik2_kernels_test` (double-precision reference, no model loaded) with `IK2_GROUP_MMA=3`,
all four combinations of `IK2_EXPERT_WMMA` and `IK2_FA_WMMA`: **346 ok / 0 fail** each.

End to end: `maya-ik2 --model Inkling-2-Mini-UD-Q4_K_XL.gguf --tokens @ids --gen 32 --ctx 20480
--cache-gb 12` runs on the card; 8 engine runs at 512 tokens and 4 at 16,384 tokens produced
identical greedy id sequences within each prompt size.

## Kernel-level profile of decode (and how to read it)

`rocprofv3 --kernel-trace`, 1-token prompt + 16 generated tokens, ctx 2048, `--cache-gb 12`: 0.268 s
of GPU time over 21,549 dispatches.

**The profiling perturbs the run.** The same binary reports 16.80 tok/s decode under `rocprofv3`
against ~51 uninstrumented (28.93 vs ~51 with a 64-token prompt), because tracing serialises and adds
per-dispatch overhead. Quote shares from a trace; quote tok/s only from uninstrumented runs.

| Kernel | GPU time | Calls | Share |
| --- | --- | --- | --- |
| `experts_gate_up_swiglu_kernel<4,2>` | 45.70 ms | 1,226 | 17.1 % |
| hipBLASLt / Tensile GEMM (all shapes) | 112.1 ms | - | 41.8 % |
| `experts_down_kernel<4,4>` | 24.23 ms | 744 | 9.0 % |
| `rocclr_copyBuffer` (expert uploads) | 18.26 ms | 2,046 | 6.8 % |
| `Cijk...MT16x32x256` (one big GEMM a token) | 23.41 ms | 16 | 8.7 % |
| `route_warp_kernel` | 8.52 ms | 768 | 3.2 % |

What this says:

- **The expert kernels are bandwidth-bound, not arithmetic-bound.** A decode layer's expert launch
  streams 8 experts' 21.2 MB in 48 us - 442 GB/s of the card's 640, 69 % of peak. There is no
  headroom for matrix cores to recover.
- **The dense path was the largest remaining share, and it is fixed.** 42 % of decode was hipBLASLt
  f16 GEMMs reading 2 bytes a weight value. The per-token dequant this profile was read as is **not**
  happening - every `decode_f16_q8_kernel` call lands in the load, once - but the resident FP16 copy
  it builds is what the GEMMs read. Keeping the dense weights planar Q8_0 and taking a token's product
  straight over the blocks (`IK2_DENSE=q8`, now the default) collapses that family, see the next
  section.
- **Profile first.** Three optimistic kernel changes (expert WMMA, the flash-attention product,
  a decode expert-MMA dispatch) each measured inside noise end to end, for the same reason: the
  component was optimised before the decode was profiled. The measured nulls are the reason the
  matrix-core work above is described by its *kernel* speed and only claimed as a prefill lever.

## The dense path: planar Q8_0 (MEASURED, +20 % decode)

The fix is not a fused dequant, because there is no per-token dequant to fuse: in the decode-only
trace all 145 `decode_f16_q8_kernel` calls fall in the first 303 ms of a 4.3 s run - they are the
one-time load of the dense weights (9.1 ms of GPU time), not a per-token cost. What the profile really
shows is the consequence of that load: at T = 1 every dense product is a hipBLAS f16 x f16 GEMM over
a resident FP16 copy, reading 2 bytes a weight value, and that copy is 3.0 GiB of the card's VRAM
against 1.59 GiB planar by the GGUF's own dense census (1,609 M dense values, 646 tensors) - 1.4 GiB the
expert cache cannot use.

**The shapes were counted, not guessed.** `IK2_GEMM_LOG=1` (kept as a diagnostic) prints the
(N, K, T) table a run issues; one decode token asks for 221 dense products:

| N | K | calls/token | what |
| --- | --- | --- | --- |
| 8600 | 2304 | 20 | sliding layers' fused qkvrg + sink |
| 6552 | 2304 | 8 | full layers' fused qrgate + sink |
| 3480 | 2304 | 20 | full layers' fused qrgate + sink, gate unfused |
| 3072 | 2304 | 20 | full layers' `attn_q` |
| 1056 | 2304 | 16 | `wk`/`wv` of the short-conv layers |
| 1024 | 2304 | 40 | `attn_k` / `attn_v` |
| 2304 | 3072 | 48 | `attn_output` |
| 128 | 2304 | 48 | router |
| 201024 | 2304 | 1 | head |

`tests/ik2_dense_bench.cu` (`ik2_dense_bench`, built with the engine) times each shape three ways over
the same data, 200 reps: the hipBLAS f16 GEMM, `mv_q8p` over **planar** Q8_0 (int8 values + FP16
block scales, a warp a row), and `mv_blocks` over the GGUF's 34-byte Q8_0 blocks as they lie.

| shape (N x K) | GEMM us | mvq8 us | mvraw us | rel mvq8 |
| --- | --- | --- | --- | --- |
| 8600 x 2304 | 46.0 | **24.2** | 47.2 | 2.9e-04 |
| 6552 x 2304 | 23.6 | **13.6** | 20.4 | 2.9e-04 |
| 3480 x 2304 | 13.3 | **9.2** | 15.0 | 2.9e-04 |
| 3072 x 2304 | 12.2 | **8.6** | 14.1 | 3.0e-04 |
| 1056 x 2304 | 8.2 | **5.4** | 8.1 | 2.9e-04 |
| 1024 x 2304 | 8.2 | **5.3** | 8.3 | 2.8e-04 |
| 2304 x 3072 | 12.1 | **9.2** | 12.2 | 3.0e-04 |
| 128 x 2304 | 9.0 | **4.7** | 6.3 | 2.9e-04 |

Planar Q8_0 wins every shape by 24-48 %, and over the calls a token makes it is 1910 us against the
GEMM's 3090 us (-38 %). The raw-block form loses to the GEMM on most shapes - the blocks are read
inefficiently by a warp-per-row loop - so *planar*, not "the GGUF's blocks as they lie", is the form
to keep. The bench's GB/s column is inflated by the 64 MB Infinity Cache (the matrices fit in it): the
ranking is what transfers, not the absolute rate.

The engine already had this layout behind `IK2_DENSE`, but only switched to it when every expert fit in
VRAM, which on 16 GB never happens. It is now the default whenever every dense matrix qualifies (Q8_0
with K a multiple of 128 - the model is asked at load, not assumed), with `IK2_DENSE=f16` the opt-out.

Decode, 128-token prompt, `--ctx 4096 --cache-gb 12`, 96 generated, 2 runs an arm, interleaved:

| arm | run 1 | run 2 |
| --- | --- | --- |
| `IK2_DENSE=f16` | 52.72 | 52.87 |
| planar Q8_0 (default) | 63.73 | 63.72 |

**+20.7 % decode**, against run-to-run noise of ~1.7 % on this machine. At depth (16,384-token prompt,
`--ctx 32768`) it is 49.27 -> 58.09 tok/s, +17.9 %, at a **-5.8 % prefill cost** (1622.6 -> 1528.0):
with no FP16 copy, a T > 1 product decodes the matrix it needs on demand, a chunk at a time, which is
the price of freeing 1.4 GiB. The deep preset gains headroom rather than losing it - planar is cheaper in
VRAM, and the freed pages go to the expert cache (1867 misses against 1983 on the 16K prefill).

Parity: `ik2_kernels_test` 346 ok / 0 fail; greedy ids identical over 128 generated tokens, byte for
byte, both arms; the planar product's worst RMS difference from the GEMM is 3.0e-04, which is the f16
copy's own rounding - planar Q8_0 is exactly the value that copy was rounded from.

Kernel-level delta, `rocprofv3 --kernel-trace`, PERTURBED (1-token prompt, 16 generated, ctx 2048) -
shares only, never throughput:

| family | f16 copies | planar Q8_0 |
| --- | --- | --- |
| total GPU | 286.8 ms | **237.2 ms** |
| hipBLASLt / Tensile GEMM | 111.92 ms (39.0 %), 4,304 calls | **12.82 ms (5.4 %)**, 3,584 calls |
| `mv_q8p_kernel<4,...>` | - | 54.95 ms (23.2 %), 2,768 calls |
| expert kernels | 88.3 ms | 92.8 ms (cache variance) |
| `rocclr_copyBuffer` | 19.35 ms | 19.17 ms |

The dense GEMM family collapses by 89 %, and the traced run's total GPU time drops 17.3 %: the
mechanism is the dense products, and the +20.7 % is the untraced measurement.

## Not verified here

The CUDA build (no NVIDIA card on this machine), gfx1100 / gfx1151, cards with every expert resident
in VRAM, and any quality claim at depth: the engine runs on one 16 GB RDNA4 card and the numbers
above come from that one.
