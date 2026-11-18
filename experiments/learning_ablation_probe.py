"""Does dopamine learning matter WITHIN a life of the evolved learning fly?

Same brain (fly_v7_learn champion), same held-out games (seeds 500000-500063, 300 moves),
learning on vs learning switched off (rate 0). Also reports how far the plastic
Kenyon cell -> MBON synapses moved from their FlyWire strength by the end.
The comparison is paired: each seed is played by both versions.

Run:  python -m experiments.learning_ablation_probe
Writes results/stage7/learning_ablation_probe.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agents.policies import FlyPolicy
from evolution.rollout import play_budget
from experiments.evolve import train_config

ROOT = Path(__file__).resolve().parents[1]
RUN = "fly_v7_learn"


def paired_ci(d, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, (n_boot, len(d))).mean(1)
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def main():
    from brain.connectome_loader import load_connectome
    conn = load_connectome()
    rc = json.loads((ROOT / "results" / "evolution" / RUN / "config.json").read_text())
    g = np.repeat(np.load(ROOT / "results" / "evolution" / RUN / "champion.npy")[None], 64, 0)
    seeds = 500_000 + np.arange(64)
    raw, out = {}, {}
    for name, plastic in [("learning on", True), ("learning off", False)]:
        fly = FlyPolicy(conn, g, dt=rc["dt"], window_ms=rc["window_ms"], readout_from_ms=rc["readout_from_ms"],
                        regular_input=rc["regular_input"], seed=1, learning=True, plasticity=plastic,
                        theta_scale=rc["theta_scale"])
        r = play_budget(fly, train_config(20), seeds, 300)
        ps = fly.sim.pscale.float().cpu().numpy()          # (n_plastic, B), 1 = FlyWire strength
        raw[name] = r
        out[name] = {"food_mean": round(float(r["food"].mean()), 3), "deaths_mean": round(float(r["deaths"].mean()), 3),
                     "plastic_synapses": int(ps.shape[0]),
                     "mean_depression": round(float((1 - ps).mean()), 5),
                     "frac_below_0.9": round(float((ps < 0.9).mean()), 5),
                     "frac_below_0.5": round(float((ps < 0.5).mean()), 5),
                     "min_scale": round(float(ps.min()), 4)}
        print(name, out[name], flush=True)
        del fly
    for k in ("food", "deaths"):
        d = raw["learning on"][k] - raw["learning off"][k]
        out[f"paired_{k}_on_minus_off"] = {"mean": round(float(d.mean()), 3), "ci95": paired_ci(d),
                                           "seeds_on_better": int((d > 0).sum() if k == "food" else (d < 0).sum()),
                                           "seeds_equal": int((d == 0).sum())}
    print(out, flush=True)
    (ROOT / "results" / "stage7" / "learning_ablation_probe.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
