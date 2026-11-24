# The KLD reference gate (`--dump-ref` / `--kld-eval`)

Token agreement is a weak gate on a kernel change: a kernel can be wrong in a way that keeps the argmax
identical for 128 tokens and still damages long-context quality. This is the instrument that sees the
difference — Strata's `cmd_kld_eval` (`engine/src/inkling2/ink_main.cpp:233-300`) cloned into maya-ik2. A
build dumps the teacher-forced logits of one fixed text; every later build is scored against that file by
**Mean KLD / Top-1 / PPL**.

It is built from `src/ik2/ik2_main.cu` beside `--gen` and `--serve`, and it never enters the sampler, the
prefill scheduler or the KV code: both modes return before any decoding.

## Usage

```bash
# 1. dump a reference (from the build every later build is compared to)
./build-hip-kld/maya-ik2 --model Inkling-2-Mini-UD-Q4_K_XL.gguf \
    --tokens @tools/kld/ref-512.tokens --ctx 4096 --cache-gb 4 \
    --dump-ref /tmp/ik2-ref.bin

# 2. score a build against it (drop --tokens: the ids travel in the reference)
./build-hip-kld/maya-ik2 --model Inkling-2-Mini-UD-Q4_K_XL.gguf \
    --ctx 4096 --cache-gb 4 --kld-eval /tmp/ik2-ref.bin

# 3. the same thing as a gate: exit 3 when Mean KLD is over the budget
./build-hip-kld/maya-ik2 ... --kld-eval /tmp/ik2-ref.bin --kld-max 0.01
```

Output, in Strata's shape so the numbers are comparable with theirs:

```
[kld-eval] n=512 | Mean KLD=0.000000 | Top-1=100.00% | PPL=12.345
[kld-eval] reference /tmp/ik2-ref.bin (v1, f32, rows=512, vocab=200058, chunk=2048, ids fnv1a ...);
           NLL=2.5132 over 511 targets (reference NLL=2.5132); mean|dlogit|=0.000e+00, max|dlogit|=0.000e+00
```

`n` is the number of positions compared (one per text token). KLD and Top-1 use all `n`; `PPL` is the
candidate's, over the `n-1` positions that have a next token. `mean|dlogit|`/`max|dlogit|` are the
log-probability drift, for diagnosis.

## The fixed text

`tools/kld/ref-512.txt` is the text (the opening of *Alice's Adventures in Wonderland*, public domain);
`tools/kld/ref-512.tokens` is the comma-separated token-id file the engine actually reads. Both are
committed, and neither may change: a moving reference makes every later number meaningless.

The ids were made once, with the model's own tokenizer:

```bash
env PYTHONPATH= ~/Projects/maya-inkling-mini/.venv/bin/python tools/kld/kld_ref.py ids \
    --gguf ~/Projects/inkling-2-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf \
    --text tools/kld/ref-512.txt --out tools/kld/ref-512.tokens --count 512
```

The script refuses to write the ids unless they decode back to a prefix of the text.

## The file format

Little-endian (x86-64, this engine's only target):

| field | type | meaning |
|---|---|---|
| `magic` | `char[8]` | `"IK2KLD01"` |
| `version` | `uint32` | 1 — the layout below |
| `dtype` | `uint32` | 0 = float32 logits rows |
| `rows` | `uint32` | logits rows, one per text position (must equal `n_tok`) |
| `vocab` | `uint32` | `n_vocab` of the model that dumped it |
| `n_tok` | `uint32` | the fixed text, as token ids |
| `chunk` | `uint32` | the `--chunk` it was prefilled at |
| `reserved` | `uint32` | 0 |
| `ids` | `int32[n_tok]` | the text |
| `logits` | `float[rows][vocab]` | raw logits, `W.logits` rows before any softmax |

`--kld-eval` **refuses** a reference whose magic, version, dtype, vocab, row count, token count, prefill
chunk or byte length disagrees with the run, and says which: a silently mismatched reference produces a
meaningless number that looks like a good one.

## Why the chunk is in the header

`--chunk` is not output-identical (`docs/AMD_INKLING2.md`): two chunk sizes run different prefill
schedulings, so their logits differ slightly. A reference dumped at `--chunk 2048` and evaluated at
`--chunk 4096` is not a kernel change and must not be reported as one, so a chunk mismatch is refused
rather than scored. Dump and evaluate with the same `--chunk` (the default is 2048 both ways).

## Proving the metric, not the model

A number from this tool is a claim about the metric until it is proven both ways:

1. **It can return exactly zero.** Evaluate a build against a reference *that same build* dumped:
   `Mean KLD=0.000000`, `Top-1=100.00%`. A broken metric cannot return exactly zero, so this line is the
   proof that the arithmetic is wired.
2. **It discriminates.** Perturb one row of the reference and the number must move:

   ```bash
   tools/kld/kld_ref.py perturb ref.bin ref-row17.bin --row 17 --scale 1.05
   ./build-hip-kld/maya-ik2 --model ... --ctx 4096 --kld-eval ref-row17.bin
   ```

   A perturbation that does not move Mean KLD means the metric is wrong, not that the model is fine.
   A *different numerics configuration* of the same engine (a kernel path off, another KV type) is the
   real cross-build check; `tools/kld/kld_ref.py compare a.bin b.bin` recomputes the same KLD, top-1 and
   PPL in numpy, independently of the C++.

`tools/kld/prove-gate.sh` runs all of that against one binary and exits non-zero unless it holds:

```bash
IK2_BIN=./build-hip-kld/maya-ik2 IK2_MODEL=…/Inkling-2-Mini-UD-Q4_K_XL.gguf \
  tools/kld/prove-gate.sh /tmp/ik2-gate
```

## Measured: the proving run on `ik2/kld-eval`

Evidence for the instrument, not for the model. Binary `build-hip-kld/maya-ik2` sha256
`182163fb867af4dc710d784af5ef1af5aa51bf646c293efbd730519769e78cbc` from commit `e4b6c32`; model
`Inkling-2-Mini-UD-Q4_K_XL.gguf`; `--ctx 4096 --cache-gb 4`, default `--chunk 2048`, default (f16) KV.
The reference this build dumped is `ref.bin`, sha256
`de14da3dd62b383de5edbfebf7f8905d3fe0917296dd2bb971c97bb1d73e6537`.

| run | reference | Mean KLD | Top-1 | mean\|dlogit\| |
|---|---|---|---|---|
| self-control: this build against its own dump | `ref.bin` | **0.000000** | 100.00% | 0.000e+00 |
| numpy witness of the same pair (`kld_ref.py compare`) | `ref.bin` vs `ref.bin` | **0.000000** | 100.00% | 0.000e+00 |
| row 17 scaled 1.05 | `ref.bin` perturbed | **0.000002** | 100.00% | 2.916e-03 |
| row 17 scaled 2.00 | `ref.bin` perturbed | **0.000038** | 100.00% | 5.851e-02 |
| `IK2_KV=fp8` against the f16 reference | `ref.bin` | **0.005106** | 99.22% | 3.569e-01 |

Read the last two rows together with the Top-1 column: the x2.00 perturbation moves Mean KLD by 3.8e-5
while top-1 stays at 100.00% — exactly the wrong-but-argmax-identical failure a token-agreement gate
cannot see, and the reason this instrument exists. `IK2_KV=fp8` is a real numerics change of the same
engine: 0.005106 on this 512-token text (the same order as the 0.004552 that Kintsugi's q4_0-global
cites on their own text, and as P6-1's 0.0059 on 2048 prose targets at 32K — different text, not
comparable directly).

The guards were exercised too: a reference with a wrong magic is refused (`bad magic - not an ik2 KLD
reference`), one whose header vocab is off by one is refused (`vocab 200057 but the model's is
200058`), and `--kld-max 0` on the perturbed file exits **3** (`FAIL: Mean KLD 0.000038 > the budget
0.000000`) while the self-control passes the same gate at `--kld-max 0.000001`.

Cross-build note: the P6-1 instrument (`~/Projects/maya-ik2-ppl/build-ppl/maya-ik2`
`854b8430d4173788…`, built from `5ad408b` + `--ref-window`/`--cand-out`) dumped the same 512 tokens to
**byte-identical logits** (sha256 of the logits block `19808afd59cf83aa…`), so the two independent
implementations of this format and this forward path agree exactly.

## Per class, not one aggregate

The file carries its own token ids, so a class split needs no new code: dump one reference per class
and score each. Keep `ref-512.txt` fixed as the common baseline and add e.g. `ref-toolcall.txt` /
`ref-code.txt` next to it, each with its committed `.tokens`. An aggregate over mixed text hides a
class-specific regression: P6-1 measured in-domain code ~10x cleaner than prose at 32K/128K
(`runs/P6-1-ppl-longctx.md`), and user/tool text past ~1,000 tokens is the class that degrades first.
