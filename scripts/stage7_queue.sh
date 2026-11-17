#!/usr/bin/env bash
# Stage 7 queue (one job at a time on the GPU). Safe to stop and re-run at any time:
# evolve.py resumes each run from its last finished generation, and a run counts as
# done only when its history.csv has every generation AND it has been diagnosed.
#
#  1. ann_v7_odour     ANN control that also smells
#  2. fly_v7_learn     calibrated brain + smell + mushroom-body vision + dopamine learning
#  3. fly_v7_control   identical, but synapses never change (learning forced off)
#  then a full evaluation of both flies (unseen games, full games, A*, 50x50)
#
# Usage: PY=path/to/python bash scripts/stage7_queue.sh   (or double-click resume_stage7.bat)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOG=results/stage7_queue.log
mkdir -p results/stage7

FLY="--controller fly --learning --generations 40 --decisions 600 --regular_input --window_ms 72 --readout_from_ms 12 --theta_scale 8 --sigma0 0.15 --fitness penalty --death_penalty 2 --init_from results/evolution/fly_v6_body_penalty2/champion.npy --seed 0"
ANN="--controller ann --odour --generations 80 --decisions 600 --fitness penalty --death_penalty 2 --seed 0"

done_run() {  # name generations
  local h="results/evolution/$1/history.csv"
  [ -f "$h" ] && [ "$(wc -l < "$h")" -ge $(( $2 + 1 )) ]
}

train() {  # name generations args...
  local name=$1 gens=$2; shift 2
  if done_run "$name" "$gens"; then echo "$(date) skip training $name (done)" >> "$LOG"; return; fi
  echo "$(date) start/resume $name" >> "$LOG"
  $PY -m experiments.evolve "$@" --run "$name" >> "results/evolve_$name.log" 2>&1
  echo "$(date) trained $name (exit $?)" >> "$LOG"
}

diagnose() {  # name generations
  if ! done_run "$1" "$2"; then echo "$(date) not diagnosing $1 (training unfinished)" >> "$LOG"; return; fi
  if [ -f "results/stage5/death_diagnosis_$1.json" ]; then return; fi
  $PY -m experiments.diagnose_deaths --run "$1" --quick > "results/diag_$1.log" 2>&1
  echo "$(date) diagnosed $1 (exit $?)" >> "$LOG"
}

evaluate() {  # fly stage
  if ! done_run "$1" 40; then return; fi
  if [ -f "results/$2/evaluation.json" ]; then echo "$(date) skip evaluation $2 (done)" >> "$LOG"; return; fi
  echo "$(date) start evaluation $2" >> "$LOG"
  $PY -m experiments.evaluate_evolution --fly "$1" --ann ann_v7_odour --shuffled none --stage "$2" --big --skip_dt_check > "results/eval_$2.log" 2>&1
  echo "$(date) evaluated $2 (exit $?)" >> "$LOG"
}

train ann_v7_odour 80 $ANN
train fly_v7_learn 40 $FLY
diagnose fly_v7_learn 40
train fly_v7_control 40 $FLY --no_plasticity
diagnose fly_v7_control 40
evaluate fly_v7_learn stage7
evaluate fly_v7_control stage7_control
echo "$(date) queue done" >> "$LOG"
