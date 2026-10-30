#!/usr/bin/env bash
# One-factor ablation, end to end with the official judges.
#
# base = verbatim text and images + image descriptions + BM25 + dense + neighbouring exchanges (+ whole memory when
# it fits). Each other arm adds exactly one component to base; "full" is the current default configuration.
# Image-budget arms vary only how many original images are returned and at what size (img0, img8, img2x768 vs full);
# compare them against full: ARMS="full img0 img8 img2x768" BASE_ARM=full eval/ablate.sh OUT
# All arms share one state directory per dataset, so images are described and texts embedded only once; arms
# differ only at Search time. Compare an arm with: python -m eval.compare OUT/base.answers.jsonl OUT/<arm>.answers.jsonl
#
# usage: eval/ablate.sh OUT_DIR [N_PER_TYPE]      (needs model keys in the environment, e.g. from .env)
#        ARMS="base full" DATASETS="memlens256k" eval/ablate.sh OUT 10     (a subset)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?output dir}; N=${2:-20}
mkdir -p "$OUT/state"
PY=.venv/bin/python

BASE="EXPAND_N=0 TIME_WEIGHT=0 RECENCY_WEIGHT=0 OPTIONS_WEIGHT=0 RERANK=0"
arm_env() {  # settings of each arm (plain function: macOS ships bash 3.2, which has no associative arrays)
  case $1 in
    base)    echo "$BASE" ;;
    rerank)  echo "$BASE RERANK=1" ;;
    expand)  echo "$BASE EXPAND_N=3" ;;
    time)    echo "$BASE TIME_WEIGHT=1" ;;
    recency) echo "$BASE RECENCY_WEIGHT=0.5" ;;
    options) echo "$BASE OPTIONS_WEIGHT=0.25" ;;
    full)    echo "" ;;
    img0)    echo "MAX_IMAGES=0" ;;
    img8)    echo "MAX_IMAGES=8" ;;
    img2x768) echo "MAX_IMAGES=2 RETURN_IMAGE_SIDE=768" ;;
  esac
}
ARMS="${ARMS:-base rerank expand time recency options full}"
DATASETS="${DATASETS:-memlens32k memlens256k memeye mobilemem}"

run() {  # run <arm> <dataset tag> <command...>
  local arm=$1 tag=$2; shift 2
  echo "== $tag / $arm"
  env $(arm_env "$arm") EVAL_STATE_DIR="$OUT/state/$tag" "$@" --out "$OUT/$tag.$arm.json" 2>>"$OUT/$tag.$arm.log" \
    | grep -E "^answer|^ALL|judge|score=" || true
}

for arm in $ARMS; do
  for tag in $DATASETS; do
    case $tag in
      memlens32k)  run "$arm" $tag $PY -m eval.replay_memlens _data/memlens --split 32k  --answer "$N" --only-answered ;;
      memlens256k) run "$arm" $tag $PY -m eval.replay_memlens _data/memlens --split 256k --answer "${N256:-$N}" --only-answered ;;
      memeye)      run "$arm" $tag $PY -m eval.replay_memgallery _data/memeye/data --glob '[A-Z]*.json' --grader memeye --answer "${NME:-$N}" ;;
      mobilemem)   run "$arm" $tag $PY -m eval.replay_mobilemem _data/mobilemem --users 3 --answer "$N" ;;
    esac
  done
done

echo; echo "paired comparisons against ${BASE_ARM:-base}:"
for tag in $DATASETS; do
  for arm in $ARMS; do
    [ "$arm" = "${BASE_ARM:-base}" ] && continue
    echo "--- $tag: $arm vs ${BASE_ARM:-base}"
    $PY -m eval.compare "$OUT/$tag.${BASE_ARM:-base}.answers.jsonl" "$OUT/$tag.$arm.answers.jsonl" || true
  done
done
