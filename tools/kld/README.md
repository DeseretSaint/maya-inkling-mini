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
