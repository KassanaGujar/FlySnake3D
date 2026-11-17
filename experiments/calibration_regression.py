"""Does the stage 7 calibration change the earlier (vision-only) results?

Replays the stage 5 champion (fly_v5_regular_late12) on the stage 5 held-out
games (seeds 500000-500063, 300 moves) with the published model and with the
stage 7 calibration (adaptation + raised Kenyon cell threshold).

Run:  python -m experiments.calibration_regression
Writes results/stage7/calibration_regression.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agents.policies import FlyPolicy
from evolution.rollout import play_budget
from experiments.evaluate_evolution import summarise
from experiments.evolve import train_config

ROOT = Path(__file__).resolve().parents[1]
RUN = "fly_v5_regular_late12"


def main():
    from brain.connectome_loader import load_connectome
    conn = load_connectome()
    rc = json.loads((ROOT / "results" / "evolution" / RUN / "config.json").read_text())
    g = np.repeat(np.load(ROOT / "results" / "evolution" / RUN / "champion.npy")[None], 64, 0)
    seeds = 500_000 + np.arange(64)
    res = {}
    for name, cal in [("published model", False), ("stage 7 calibration", True)]:
        fly = FlyPolicy(conn, g, dt=rc["dt"], window_ms=rc["window_ms"], readout_from_ms=rc["readout_from_ms"],
                        regular_input=rc["regular_input"], seed=1, calibrated=cal)
        res[name] = summarise(play_budget(fly, train_config(20), seeds, 300))
        print(name, res[name], flush=True)
        del fly
    out = ROOT / "results" / "stage7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration_regression.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
