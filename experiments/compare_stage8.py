"""Stage 8 summary: learning fly vs matched no-learning control (and the stage 7 pair).

Reads the outputs of evaluate_evolution.py and diagnose_deaths.py; plays nothing.
Full-game comparisons are paired by seed (same 64 games), with bootstrap 95 % CIs.

Run:  python -m experiments.compare_stage8
Writes results/stage8/comparison.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
FLY = "fly (evolved)"
PAIRS = {"stage8": ("stage8", "stage8_control", "fly_v8_learn", "fly_v8_control"),
         "stage7": ("stage7", "stage7_control", "fly_v7_learn", "fly_v7_control")}


def ci(x, n=4000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return [round(float(v), 2) for v in np.percentile(m, [2.5, 97.5])]


def paired(a_dir, b_dir, fname):
    a = pd.read_csv(R / a_dir / fname)
    b = pd.read_csv(R / b_dir / fname)
    a = a[a.agent == FLY].set_index("seed").food_eaten
    b = b[b.agent == FLY].set_index("seed").food_eaten
    d = (a - b.reindex(a.index)).to_numpy()
    return {"learn": round(float(a.mean()), 2), "control": round(float(b.mean()), 2),
            "learn_minus_control": round(float(d.mean()), 2), "ci95": ci(d),
            "learn_better_games": int((d > 0).sum()), "control_better_games": int((d < 0).sum())}


def held_out(stage):
    e = json.loads((R / stage / "evaluation.json").read_text())["fly_held_out_progress"]
    return {"start_genome": e[next(k for k in e if k.startswith("gen 0"))], "final": e["final"]}


def diag(run):
    p = R / "stage5" / f"death_diagnosis_{run}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())["fly_as_trained"]
    keys = ("food", "deaths", "by_outcome", "fraction_avoidable", "mean_length_at_death")
    out = {k: d.get(k) for k in keys}
    return out


def main():
    out = {}
    for name, (sa, sb, ra, rb) in PAIRS.items():
        if not (R / sa / "evaluation.json").exists():
            continue
        out[name] = {"held_out_300_moves": {"learn": held_out(sa), "control": held_out(sb)},
                     "full_games_20x20": paired(sa, sb, "episodes.csv"),
                     "full_games_50x50": paired(sa, sb, "episodes_50x50.csv"),
                     "death_diagnosis": {"learn": diag(ra), "control": diag(rb)}}
    e = json.loads((R / "stage8" / "evaluation.json").read_text())
    out["references_full_games_20x20"] = {k: e["full_episodes"][k]["food_mean"] for k in e["full_episodes"]}
    out["references_held_out"] = {k: {"food": v["food_mean"], "deaths": v["deaths_mean"]}
                                  for k, v in e["controls_held_out"].items()}
    (R / "stage8" / "comparison.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
