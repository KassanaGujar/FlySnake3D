"""How many steering spikes does each decision rest on?

Plays a champion on 64 unseen games for 100 moves and records, per decision,
the readout spike counts the decoder sees (left and right, all five types) and
how far the weighted turn signal is from the threshold. If decisions rest on a
handful of spikes, chance alone will flip many of them.

Measurement only; nothing is changed.
Run:  python -m experiments.readout_spike_probe [--run fly_v5_regular_late12]
Writes results/stage6/readout_spike_probe_<run>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agents.policies import FlyPolicy
from brain.motor_decoder import N_OUT, THETA_SCALE, READOUT_TYPES
from brain.visual_encoder import N_FEAT, view_features
from evolution.rollout import LIFE_SEED_STEP
from experiments.evolve import train_config
from game.environment import SnakeGame

ROOT = Path(__file__).resolve().parents[1]
B, N_DEC = 64, 100


def main():
    from brain.connectome_loader import load_connectome
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="fly_v5_regular_late12")
    a = ap.parse_args()
    rc = json.loads((ROOT / "results" / "evolution" / a.run / "config.json").read_text())
    champ = np.load(ROOT / "results" / "evolution" / a.run / "champion.npy")
    conn = load_connectome()
    fly = FlyPolicy(conn, np.repeat(champ[None], B, 0), dt=rc["dt"], window_ms=rc["window_ms"],
                    readout_from_ms=rc.get("readout_from_ms", 0.0), regular_input=rc.get("regular_input", False))
    cfg = train_config(20)
    seeds = np.arange(800_000, 800_000 + B)
    games = [SnakeGame(cfg, int(s)) for s in seeds]
    prev, lives = [None] * B, np.zeros(B, int)
    feats = np.zeros((B, 2, N_FEAT), np.float32)
    fly.reset(np.ones(B, bool))
    totals, margins, per_type = [], [], []
    w = fly.weights.cpu().numpy()[0]
    th = float(fly.theta.cpu().numpy()[0]) * THETA_SCALE
    for _ in range(N_DEC):
        for b, g in enumerate(games):
            feats[b], prev[b] = view_features(g, prev[b])
        acts = fly.act(feats)
        c = fly.last_counts.view(N_OUT, 2, B).cpu().numpy()          # type, side, game
        totals.append(c.sum((0, 1)))
        per_type.append(c.sum(1))
        turn = ((c[:, 1] - c[:, 0]).T * w).sum(1)
        margins.append(np.abs(np.abs(turn) - th))
        restart = np.zeros(B, bool)
        for b, g in enumerate(games):
            g.step(int(acts[b]))
            if g.done:
                lives[b] += 1
                games[b] = SnakeGame(cfg, int(seeds[b]) + LIFE_SEED_STEP * int(lives[b]))
                prev[b], restart[b] = None, True
        if restart.any():
            fly.reset(restart)
    tot = np.concatenate(totals)
    mar = np.concatenate(margins)
    pt = np.stack(per_type).mean((0, 2))
    res = {"run": a.run, "decisions": int(tot.size),
           "readout_spikes_per_decision_mean": round(float(tot.mean()), 2),
           "readout_spikes_per_decision_median": float(np.median(tot)),
           "share_of_decisions_with_0_spikes": round(float((tot == 0).mean()), 3),
           "share_with_at_most_3_spikes": round(float((tot <= 3).mean()), 3),
           "mean_spikes_per_type": dict(zip(READOUT_TYPES, np.round(pt, 2).tolist())),
           "weights": dict(zip(READOUT_TYPES, np.round(w, 3).tolist())), "threshold": round(th, 3),
           "share_of_decisions_within_1_weighted_spike_of_threshold": round(float((mar < 1).mean()), 3)}
    out = ROOT / "results" / "stage6"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"readout_spike_probe_{a.run}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
