#!/usr/bin/env python3
"""tools/kld/kld_ref.py - the KLD reference file, outside the engine.

maya-ik2's `--dump-ref` / `--kld-eval` (src/ik2/ik2_main.cu) are the gate; this script is what makes a
reference reproducible and what a reviewer can hold the engine's number against:

    ids       tokenize the fixed text into the committed token-id file (once, never again)
    info      read a reference's header and digest
    perturb   copy a reference with one row scaled - the negative control for the gate
    compare   compute KLD / top-1 / PPL between two reference files, in numpy, independently of the C++

The file format (little-endian)::

    char   magic[8]   "IK2KLD01"
    uint32 version    1
    uint32 dtype      0 = float32 logits rows
    uint32 rows       logits rows (one per text position)
    uint32 vocab      n_vocab of the model that dumped it
    uint32 n_tok      the fixed text, as token ids
    uint32 chunk      the --chunk it was prefilled at
    uint32 reserved   0
    int32  ids[n_tok]
    float  logits[rows][vocab]

Why the tokens are committed as ids and not as text: maya-ik2 takes ids, so the text is only as fixed as
the ids.  `ids` records the text it came from, the digest of the ids and the model it tokenized with, and
the ids are then a file under version control - a moving reference makes every later number meaningless.

Usage:
    kld_ref.py ids --gguf MODEL.gguf --text tools/kld/ref-512.txt --out tools/kld/ref-512.tokens
    kld_ref.py info REF.bin
    kld_ref.py perturb IN.bin OUT.bin --row 17 --scale 1.05
    kld_ref.py compare A.bin B.bin            # A = reference, B = the run being scored
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import struct
import sys

MAGIC = b"IK2KLD01"
VERSION = 1
DTYPE_F32 = 0
HEADER = struct.Struct("<8s7I")


def _digest(ids) -> str:
    h = hashlib.sha256()
    for v in ids:
        h.update(struct.pack("<i", int(v)))
    return h.hexdigest()


def read_ref(path):
    raw = pathlib.Path(path).read_bytes()
    if len(raw) < HEADER.size:
        raise SystemExit(f"{path}: shorter than a reference header - not an ik2 KLD reference")
    magic, version, dtype, rows, vocab, n_tok, chunk, _res = HEADER.unpack_from(raw, 0)
    if magic != MAGIC:
        raise SystemExit(f"{path}: bad magic {magic!r} - expected {MAGIC!r}")
    if version != VERSION:
        raise SystemExit(f"{path}: version {version}, this tool reads {VERSION}")
    if dtype != DTYPE_F32:
        raise SystemExit(f"{path}: dtype {dtype}, this tool reads float32 ({DTYPE_F32})")
    if rows != n_tok:
        raise SystemExit(f"{path}: {rows} logits rows for {n_tok} tokens - header disagrees with itself")
    want = HEADER.size + n_tok * 4 + rows * vocab * 4
    if len(raw) != want:
        raise SystemExit(f"{path}: {len(raw)} bytes, the header says {want} - truncated or padded")
    at = HEADER.size
    ids = list(struct.unpack_from(f"<{n_tok}i", raw, at))
    at += n_tok * 4
    import numpy as np

    logits = np.frombuffer(raw, dtype="<f4", count=rows * vocab, offset=at).reshape(rows, vocab)
    return {"version": version, "dtype": dtype, "rows": rows, "vocab": vocab, "n_tok": n_tok,
            "chunk": chunk, "ids": ids, "logits": logits}


def write_ref(path, meta, ids, logits) -> None:
    import numpy as np

    arr = np.ascontiguousarray(logits, dtype="<f4")
    head = HEADER.pack(MAGIC, VERSION, DTYPE_F32, int(arr.shape[0]), int(arr.shape[1]), len(ids),
                       int(meta["chunk"]), 0)
    with open(path, "wb") as f:
        f.write(head)
        f.write(struct.pack(f"<{len(ids)}i", *ids))
        f.write(arr.tobytes())


def cmd_ids(args) -> int:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from strata_tokenizer import Tokenizer

    text = pathlib.Path(args.text).read_text(encoding="utf-8")
    tk = Tokenizer.from_gguf(args.gguf)
    ids = tk.encode(text)
    if len(ids) < args.count:
        raise SystemExit(f"{args.text}: only {len(ids)} tokens, need {args.count} - lengthen the text")
    ids = ids[: args.count]
    # The ids must decode to a prefix of the committed text, or the reference is not the text it claims.
    back = tk.decode(ids)
    if not text.startswith(back):
        at = next((i for i, (a, b) in enumerate(zip(text, back)) if a != b), min(len(text), len(back)))
        raise SystemExit(f"{args.text}: the first {args.count} ids decode to something else at char {at}: "
                         f"{back[max(0, at - 20):at + 20]!r}")
    out = pathlib.Path(args.out)
    out.write_text(",".join(str(v) for v in ids) + "\n", encoding="utf-8")
    print(f"{out}: {len(ids)} ids, sha256(ids) {_digest(ids)}")
    print(f"  first 80 chars of the decoded text: {back[:80]!r}")
    return 0


def cmd_info(args) -> int:
    r = read_ref(args.ref)
    print(f"{args.ref}: v{r['version']} dtype={r['dtype']} rows={r['rows']} vocab={r['vocab']} "
          f"n_tok={r['n_tok']} chunk={r['chunk']}")
    print(f"  sha256(ids) {_digest(r['ids'])}")
    print(f"  sha256(logits) {hashlib.sha256(r['logits'].tobytes()).hexdigest()}")
    return 0


def cmd_perturb(args) -> int:
    import numpy as np

    r = read_ref(args.src)
    logits = r["logits"].copy()
    if not 0 <= args.row < logits.shape[0]:
        raise SystemExit(f"--row {args.row} is outside 0..{logits.shape[0] - 1}")
    before = logits[args.row].copy()
    logits[args.row] = logits[args.row] * np.float32(args.scale) + np.float32(args.add)
    print(f"row {args.row}: scaled by {args.scale} (added {args.add}); "
          f"max|logit| {float(np.abs(before).max()):.4f} -> {float(np.abs(logits[args.row]).max()):.4f}")
    write_ref(args.dst, r, r["ids"], logits)
    print(f"{args.dst}: written ({r['rows']} x {r['vocab']} f32)")
    return 0


def _scores(ref_logits, cand_logits, ids):
    import numpy as np

    p = ref_logits.astype(np.float64)
    q = cand_logits.astype(np.float64)
    lp = p - (p.max(axis=1, keepdims=True) + np.log(np.exp(p - p.max(axis=1, keepdims=True)).sum(axis=1,
                                                                                           keepdims=True)))
    lq = q - (q.max(axis=1, keepdims=True) + np.log(np.exp(q - q.max(axis=1, keepdims=True)).sum(axis=1,
                                                                                           keepdims=True)))
    kld = float((np.exp(lp) * (lp - lq)).sum(axis=1).mean())
    top1 = float((p.argmax(axis=1) == q.argmax(axis=1)).mean()) * 100.0
    tgt = np.asarray(ids[1:], dtype=np.int64)
    nll = float(-lq[:-1][np.arange(len(tgt)), tgt].mean())
    return kld, top1, float(np.exp(nll)), nll


def cmd_compare(args) -> int:
    import numpy as np

    a, b = read_ref(args.a), read_ref(args.b)
    if a["vocab"] != b["vocab"] or a["rows"] != b["rows"]:
        raise SystemExit(f"shape mismatch: {a['rows']}x{a['vocab']} against {b['rows']}x{b['vocab']}")
    if a["ids"] != b["ids"]:
        raise SystemExit("the two files are logits over different token ids")
    kld, top1, ppl, nll = _scores(a["logits"], b["logits"], a["ids"])
    d = np.abs(a["logits"].astype(np.float64) - b["logits"].astype(np.float64))
    print(f"n={a['rows']} | Mean KLD={kld:.6f} | Top-1={top1:.2f}% | PPL={ppl:.3f}")
    print(f"mean|dlogit|={d.mean():.3e} max|dlogit|={d.max():.3e} NLL={nll:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ids", help="tokenize the fixed text into a committed token-id file")
    p.add_argument("--gguf", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=512)
    p.set_defaults(fn=cmd_ids)

    p = sub.add_parser("info", help="read a reference's header and digests")
    p.add_argument("ref")
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("perturb", help="copy a reference with one logits row scaled (the gate's negative control)")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--add", type=float, default=0.0)
    p.set_defaults(fn=cmd_perturb)

    p = sub.add_parser("compare", help="KLD / top-1 / PPL between two references, independently of the engine")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(fn=cmd_compare)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
