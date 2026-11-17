#!/usr/bin/env bash
# Stage 6 overnight queue (runs one job at a time on the GPU).
#  1. ablation: body-vs-wall vision with the stage 5 scoring (food - 2 x deaths),
#     to separate the effect of the new vision from the per-life scoring
#  2-3. repeats of the stage 6 setup with ES seeds 1 and 2 (different noise, different games)
# Each run is diagnosed on the same 64 unseen games afterwards.
# Usage: PY=path/to/python bash scripts/stage6_queue.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOG=results/stage6_queue.log
mkdir -p results

# wait for the stage 6 evaluation (started separately) to finish, so jobs don't share the GPU
while [ ! -f results/stage6/evaluation.json ] && ! grep -q "Traceback" results/eval_v6.log 2>/dev/null; do
  sleep 60
done
echo "$(date) evaluation finished, starting queue" >> "$LOG"

COMMON="--controller fly --generations 40 --decisions 600 --regular_input --readout_from_ms 12 --sigma0 0.15 --init_from results/evolution/fly_v5_regular_late12/champion.npy"

# Safe to stop at any time and re-run: evolve.py resumes each run from its last
# finished generation (results/evolution/<run>/checkpoint.npz), and runs that are
# already trained and diagnosed are skipped.
run() {  # name, extra args
  # done = all 40 generations in history.csv (header + 40 rows) AND diagnosed.
  # (A run stopped mid-way must not count as done even if a diagnosis exists.)
  local h="results/evolution/$1/history.csv"
  if [ -f "results/stage5/death_diagnosis_$1.json" ] && [ -f "$h" ] && [ "$(wc -l < "$h")" -ge 41 ]; then
    echo "$(date) skip $1 (already done)" >> "$LOG"; return
  fi
  echo "$(date) start/resume $1" >> "$LOG"
  $PY -m experiments.evolve $COMMON --run "$1" ${@:2} >> "results/evolve_$1.log" 2>&1
  echo "$(date) trained $1 (exit $?)" >> "$LOG"
  $PY -m experiments.diagnose_deaths --run "$1" --quick > "results/diag_$1.log" 2>&1
  echo "$(date) diagnosed $1 (exit $?)" >> "$LOG"
}

run fly_v6_body_penalty2 --fitness penalty --death_penalty 2 --seed 0
run fly_v6_body_perlife_s1 --fitness per_life --seed 1
run fly_v6_body_perlife_s2 --fitness per_life --seed 2
echo "$(date) queue done" >> "$LOG"
