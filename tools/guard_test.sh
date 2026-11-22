#!/usr/bin/env bash
# S1 - the logits finiteness guard.
#
# Two claims, both from a real run of the engine on an idle GPU:
#
#   1. the happy path is unchanged: the same prompt, greedy, the same token ids as the binary built before the
#      guard, so the check costs nothing that shows up in the output;
#   2. a non-finite logits row fails the request: IK2_TEST_NAN=<position> writes a NaN into that position's row,
#      and the engine must report it on stderr with the position, the count of non-finite logits and the first
#      offending index, print no token chosen from that row, and exit non-zero.
#
# The model is exclusive: this script refuses to start while another maya-ik2 is resident (as
# tools/run-ik2-quiet.sh does), because two copies cannot both fit in VRAM.
#
# Usage:
#   IK2_PRE_BIN=<pre-change maya-ik2> tools/guard_test.sh [--post <guard maya-ik2>] [--gen 64] [--prompt <ids file>]
#
# The prompt file is comma-separated token ids (maya-ik2's --tokens format; a space-separated file collapses to one
# id and the run silently becomes a 1-token prompt).  Defaults: the model the research tree uses, a 32-id prompt, and
# 64 greedy tokens - the card's acceptance run.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${IK2_MODEL:-$HOME/Projects/inkling-2-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf}"
R="${IK2_ROCM_ROOT:-$HOME/Projects/maya-inkling-mini/.rocm/lib/python3.12/site-packages/_rocm_sdk_devel}"
PRE="${IK2_PRE_BIN:-}"                  # the build before the guard (the measurement lane's build-hip/maya-ik2)
POST="${IK2_POST_BIN:-$ROOT/build-hip-guard/maya-ik2}"
GEN=64
PROMPT=""
CTX=4096
CACHE_GB=8
WORK="${IK2_WORK:-$(mktemp -d)}"

while [ $# -gt 0 ]; do
    case "$1" in
        --pre) PRE="$2"; shift 2 ;;
        --post) POST="$2"; shift 2 ;;
        --gen) GEN="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --ctx) CTX="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --work) WORK="$2"; shift 2 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "guard_test: unknown argument $1" >&2; exit 64 ;;
    esac
done

if [ -z "$PRE" ] || [ ! -x "$PRE" ]; then
    echo "guard_test: --pre <maya-ik2 built before the guard> is required (or IK2_PRE_BIN=...)" >&2
    exit 64
fi
[ -x "$POST" ] || { echo "guard_test: no post-change binary at $POST" >&2; exit 64; }
[ -r "$MODEL" ] || { echo "guard_test: no model at $MODEL" >&2; exit 64; }
mkdir -p "$WORK"

if [ -z "$PROMPT" ]; then
    PROMPT="$WORK/prompt.txt"
    # 32 comma-separated ids: a fixed prompt, so the two builds see the same text
    : >"$PROMPT"
    for i in $(seq 1000 1031); do
        [ -s "$PROMPT" ] && printf ',' >>"$PROMPT"
        printf '%d' "$i" >>"$PROMPT"
    done
fi
NPROMPT=$(( $(tr -cd ',' <"$PROMPT" | wc -c) + 1 ))

if pgrep -x maya-ik2 >/dev/null 2>&1; then
    echo "guard_test: another maya-ik2 is resident - refusing to start (the GPU is exclusive)" >&2
    exit 3
fi

export LD_LIBRARY_PATH="$R/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export IK2_KV="${IK2_KV:-fp8}"

run() {   # run <binary> <out> <err> <extra env assignments...> -- <engine args...>
    local bin="$1" out="$2" err="$3"; shift 3
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    env "${envs[@]}" "$bin" --model "$MODEL" --tokens "@$PROMPT" --ctx "$CTX" --cache-gb "$CACHE_GB" "$@" \
        >"$out" 2>"$err"
}

tokens_of() {   # the token line the engine prints as it chooses (comma separated; the last one has no comma)
    grep -E '^[0-9]+(,[0-9]+)*$' "$1" | tail -1
}

fail=0
say() { printf '%-58s %s\n' "$1" "$2"; }

echo "guard_test: pre=$PRE"
echo "guard_test: post=$POST"
echo "guard_test: model=$MODEL prompt=$NPROMPT ids gen=$GEN work=$WORK"

# ---- 1. the happy path: the same 64 greedy token ids as the pre-change build
run "$PRE" "$WORK/pre.out" "$WORK/pre.err" -- --gen "$GEN"
rc_pre=$?
run "$POST" "$WORK/post.out" "$WORK/post.err" -- --gen "$GEN"
rc_post=$?
tok_pre="$(tokens_of "$WORK/pre.out")"
tok_post="$(tokens_of "$WORK/post.out")"
n_pre=$(tr -cd , <<<"$tok_pre" | wc -c)
n_post=$(tr -cd , <<<"$tok_post" | wc -c)
say "pre-change run (exit $rc_pre, $((n_pre + 1)) tokens)" "$([ $rc_pre -eq 0 ] && echo ok || echo FAILED)"
say "guarded run (exit $rc_post, $((n_post + 1)) tokens)" "$([ $rc_post -eq 0 ] && echo ok || echo FAILED)"
if [ "$tok_pre" = "$tok_post" ] && [ -n "$tok_pre" ] && [ $((n_pre + 1)) -eq "$GEN" ]; then
    say "greedy $GEN tokens, bit-identical to the pre-change build" "PASS"
else
    say "greedy $GEN tokens, bit-identical to the pre-change build" "FAIL"
    echo "  pre : $tok_pre"
    echo "  post: $tok_post"
    fail=1
fi

# ---- 2. a forced non-finite row: report it, choose nothing from it, exit non-zero
NAN_AT=$(( NPROMPT + 8 ))                      # the 9th generated token's position
run "$POST" "$WORK/nan.out" "$WORK/nan.err" IK2_TEST_NAN="$NAN_AT" -- --gen 16
rc_nan=$?
n_nan=$(( $(tr -cd , <<<"$(tokens_of "$WORK/nan.out")" | wc -c) + 1 ))
say "IK2_TEST_NAN=$NAN_AT run (exit $rc_nan, $n_nan tokens)" "$([ $rc_nan -ne 0 ] && echo "non-zero, ok" || echo FAILED)"
if [ $rc_nan -eq 2 ]; then say "  the request failed with the documented status 2" "PASS"
else say "  the request failed with the documented status 2" "FAIL (got $rc_nan)"; fail=1; fi
if grep -q "logits guard: token position $NAN_AT: 3 non-finite logits, first at index 5" "$WORK/nan.err"; then
    say "  stderr names the position, the count and the first index" "PASS"
else
    say "  stderr names the position, the count and the first index" "FAIL"
    sed -n '1,20p' "$WORK/nan.err"
    fail=1
fi
if [ "$n_nan" -le $(( NAN_AT - NPROMPT )) ]; then
    say "  no token was chosen from the bad row" "PASS"
else
    say "  no token was chosen from the bad row" "FAIL ($n_nan tokens printed)"
    fail=1
fi

# ---- 3. IK2_GUARD=0 restores the old behaviour: the injection is ignored (the same tokens as the pre-change build)
run "$POST" "$WORK/off.out" "$WORK/off.err" IK2_GUARD=0 IK2_TEST_NAN="$NAN_AT" -- --gen 16
rc_off=$?
tok_off="$(tokens_of "$WORK/off.out")"
tok_pre16="$(run "$PRE" "$WORK/pre16.out" "$WORK/pre16.err" -- --gen 16; tokens_of "$WORK/pre16.out")"
if [ $rc_off -eq 0 ] && [ "$tok_off" = "$tok_pre16" ]; then
    say "IK2_GUARD=0 samples anyway (the old behaviour, for bisection)" "PASS"
else
    say "IK2_GUARD=0 samples anyway (the old behaviour, for bisection)" "FAIL"
    echo "  off    : $tok_off (exit $rc_off)"
    echo "  pre-16 : $tok_pre16"
    fail=1
fi

[ $fail -eq 0 ] && echo "guard_test: PASS" || echo "guard_test: FAIL"
exit $fail
