#!/usr/bin/env bash
# tools/kld/prove-gate.sh - prove the KLD gate, not the model.
#
# A quality number from --kld-eval is worthless until the instrument has been shown both ways:
#   1. it can return exactly zero  (a build scored against a reference that same build dumped)
#   2. it discriminates            (one perturbed logits row must move Mean KLD off zero)
#   3. it refuses a broken file    (bad magic, mismatched vocab, wrong chunk ...)
# This script runs those three on one binary and exits 0 only if all of them hold. It builds nothing:
# point it at a binary that has --dump-ref/--kld-eval.
#
# Usage:
#   IK2_BIN=./build-hip-kld/maya-ik2 \
#   IK2_MODEL=~/Projects/inkling-2-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf \
#     tools/kld/prove-gate.sh [workdir]
#
# Env:
#   IK2_PYTHON  python with numpy for the perturb/compare witness (default: python3)
#   IK2_RUNNER  optional wrapper executed as "$IK2_RUNNER" BIN ARGS... - for this board that is
#               ~/Projects/inkling-2-mini/tools/run-ik2-outside-cgroup.sh, which moves the engine out
#               of a worker's 4 GiB memory cgroup (the experts are pinned in host memory and swap-spin
#               inside it). Without it the engine runs in place.
#   IK2_KV      forwarded to the engine (e.g. IK2_KV=fp8); leave unset for the default f16 KV.
set -euo pipefail

BIN="${IK2_BIN:?set IK2_BIN to a maya-ik2 built with --dump-ref/--kld-eval}"
MODEL="${IK2_MODEL:?set IK2_MODEL to the GGUF the reference is dumped for}"
PY="${IK2_PYTHON:-python3}"
WORK="${1:-${TMPDIR:-/tmp}/ik2-kld-gate.$$}"
CTX="${IK2_CTX:-4096}"
CACHE_GB="${IK2_CACHE_GB:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$WORK"

engine() {  # engine ARGS... - adds --model/--ctx/--cache-gb, or hands them to the wrapper that does
    if [ -n "${IK2_RUNNER:-}" ]; then
        IK2_MODEL="$MODEL" IK2_BIN="$BIN" IK2_CWD="$WORK" "$IK2_RUNNER" --ctx "$CTX" --cache-gb "$CACHE_GB" "$@"
    else
        "$BIN" --model "$MODEL" --ctx "$CTX" --cache-gb "$CACHE_GB" "$@"
    fi
}
field() {  # field FILE -> Mean KLD, from either the engine's [kld-eval] line or kld_ref.py compare's line
    awk 'index($0,"Mean KLD=") > 0 { sub(/^.*Mean KLD=/, ""); sub(/[ |].*$/, ""); print; exit }' "$1"
}
drift() {  # drift FILE -> mean|dlogit|, from the engine's reference line or kld_ref.py compare's line
    awk 'index($0,"mean|dlogit|=") > 0 { sub(/^.*mean\|dlogit\|=/, ""); sub(/[ ,].*$/, ""); print; exit }' "$1"
}
fail() { echo "prove-gate: FAIL: $*" >&2; exit 1; }

echo "prove-gate: bin $BIN ($(sha256sum "$BIN" | cut -d' ' -f1))"
echo "prove-gate: model $MODEL ($(sha256sum "$MODEL" | cut -d' ' -f1))"
echo "prove-gate: work $WORK  (ctx $CTX, cache-gb $CACHE_GB, chunk default)"

# --- 1. dump a reference with this build, and score this build against it: it must be exactly zero.
REF="$WORK/ref.bin"
engine --tokens "@$HERE/ref-512.tokens" --dump-ref "$REF" > "$WORK/dump.log" 2>&1 \
    || fail "the dump failed: see $WORK/dump.log"
grep -q '^\[dump-ref\]' "$WORK/dump.log" || fail "no [dump-ref] line: see $WORK/dump.log"
engine --kld-eval "$REF" --kld-max 0.000001 > "$WORK/self.log" 2>&1 \
    || fail "the self-evaluation did not pass --kld-max 0.000001: see $WORK/self.log"
kld0=$(field "$WORK/self.log"); d0=$(drift "$WORK/self.log")
[ "$kld0" = "0.000000" ] || fail "self Mean KLD is $kld0, not 0.000000 - the metric or the path is not exact"
[ "$d0" = "0.000e+00" ] || fail "self mean|dlogit| is $d0, not 0.000e+00"
grep -q 'Top-1=100.00%' "$WORK/self.log" || fail "self Top-1 is not 100.00%"
echo "prove-gate: [1] self-control: Mean KLD=0.000000, Top-1=100.00%, mean|dlogit|=0.000e+00"

# --- 2. perturb one row: the gate must move, and the independent numpy witness must agree.
PERT="$WORK/ref-row17x200.bin"
"$PY" "$HERE/kld_ref.py" perturb "$REF" "$PERT" --row 17 --scale 2.00 > "$WORK/perturb.log" 2>&1 \
    || fail "perturb failed ($PY needs numpy): see $WORK/perturb.log"
set +e
engine --kld-eval "$PERT" > "$WORK/pert.log" 2>&1; rc_pert=$?
set -e
[ "$rc_pert" = 0 ] || fail "scoring the perturbed reference exited $rc_pert: see $WORK/pert.log"
kldp=$(field "$WORK/pert.log"); dp=$(drift "$WORK/pert.log")
awk -v v="$kldp" 'BEGIN { exit !(v + 0 > 0) }' || fail "the perturbed reference scored Mean KLD=$kldp - a perturbation that does not move the metric means the metric is wrong"
set +e
engine --kld-eval "$PERT" --kld-max 0 > "$WORK/gate.log" 2>&1; rc_gate=$?
set -e
[ "$rc_gate" = 3 ] || fail "--kld-max 0 on the perturbed reference exited $rc_gate, expected 3 (the gate)"
"$PY" "$HERE/kld_ref.py" compare "$PERT" "$REF" > "$WORK/compare.log" 2>&1 \
    || fail "the numpy comparison failed: see $WORK/compare.log"
kld_np=$(field "$WORK/compare.log"); d_np=$(drift "$WORK/compare.log")
[ "$kld_np" = "$kldp" ] || fail "numpy Mean KLD $kld_np disagrees with the engine's $kldp"
[ "$d_np" = "$dp" ] || fail "numpy mean|dlogit| $d_np disagrees with the engine's $dp"
echo "prove-gate: [2] discrimination: perturbed row 17 x2.00 -> Mean KLD=$kldp (numpy $kld_np, mean|dlogit| $dp), gate exit 3"

# --- 3. a broken reference must be refused, not scored.
BAD="$WORK/ref-badmagic.bin"
"$PY" - "$REF" "$BAD" <<'PY' || fail "could not write the bad-magic reference"
import pathlib, sys
raw = bytearray(pathlib.Path(sys.argv[1]).read_bytes())
raw[0:8] = b"IK2KLD02"
pathlib.Path(sys.argv[2]).write_bytes(raw)
PY
set +e
engine --kld-eval "$BAD" > "$WORK/bad.log" 2>&1; rc_bad=$?
set -e
[ "$rc_bad" != 0 ] || fail "a reference with a bad magic was scored instead of refused"
grep -q 'bad magic' "$WORK/bad.log" || fail "the refusal does not say the magic is wrong: see $WORK/bad.log"
echo "prove-gate: [3] guard: bad magic refused (exit $rc_bad): $(grep -o 'bad magic.*' "$WORK/bad.log" | head -1)"

echo "prove-gate: PASS - self-control 0.000000, perturbation $kldp, numpy agrees, broken file refused"
echo "prove-gate: reference $REF ($(sha256sum "$REF" | cut -d' ' -f1))"
