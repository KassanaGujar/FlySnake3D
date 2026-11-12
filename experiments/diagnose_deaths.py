"""Why does the evolved fly controller die more often than the ANN control?

Plays the saved champion (one genome in every batch column, different games)
and measures, per death: what killed it, whether a safe move existed, and what
it did instead. Then tests three candidate causes:

  noise      the same view is shown to 64 freshly reset brains; how often do
             they agree on the action?
  leftover   same games, but every brain is reset before every decision
  window     same games, 48 ms of brain time per decision instead of 24 ms

Nothing here changes the controller; it only measures it.

Run:  python -m experiments.diagnose_deaths
Writes results/stage5/death_diagnosis.json
"""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np

from agents.policies import AnnPolicy, FlyPolicy
from brain.visual_encoder import FEATURES, N_FEAT, view_features
from evolution.rollout import DEATHS, LIFE_SEED_STEP
from experiments.evolve import train_config
from game.environment import SnakeGame

ROOT = Path(__file__).resolve().parents[1]
EVO = ROOT / "results" / "evolution"
OUT = ROOT / "results" / "stage5"
B, N_DEC = 64, 300
SEEDS = np.arange(700_000, 700_000 + B)        # never used in training
F = {n: i for i, n in enumerate(FEATURES)}
ACT = ("left", "straight", "right")


def safe_actions(g: SnakeGame) -> list[int]:
    """Actions that do not kill on this move (one-step check, measurement only)."""
    ok = []
    for a in range(3):
        h = copy.deepcopy(g)
        h.step(a)
        if not (h.done and h.outcome in DEATHS and h.outcome.name != "STARVED"):
            ok.append(a)
    return ok


def play_logged(policy, cfg, reset_each: bool = False):
    games = [SnakeGame(cfg, int(s)) for s in SEEDS]
    prev = [None] * B
    lives = np.zeros(B, np.int64)
    food = deaths = 0
    log, views = [], []
    policy.reset(np.ones(B, bool))
    feats = np.zeros((B, 2, N_FEAT), np.float32)
    for t in range(N_DEC):
        for b, g in enumerate(games):
            feats[b], prev[b] = view_features(g, prev[b])
        if reset_each:
            policy.reset(np.ones(B, bool))
        acts = policy.act(feats)
        if t % 12 == 0:
            views.append(feats[t % B].copy())
        restart = np.zeros(B, bool)
        for b, g in enumerate(games):
            safe = safe_actions(g)
            a = int(acts[b])
            g.step(a)
            if g.done:
                food += g.food_eaten
                if g.outcome in DEATHS:
                    deaths += 1
                    log.append({"outcome": g.outcome.name, "action": ACT[a],
                                "safe": [ACT[s] for s in safe], "length": g.length,
                                "feats": feats[b].copy()})
                lives[b] += 1
                games[b] = SnakeGame(cfg, int(SEEDS[b]) + LIFE_SEED_STEP * int(lives[b]))
                prev[b] = None
                restart[b] = True
        if restart.any():
            policy.reset(restart)
    food += sum(g.food_eaten for g in games)
    return {"food": food / B, "deaths": deaths / B}, log, views


def summarise(log) -> dict:
    n = max(len(log), 1)
    crashes = [d for d in log if d["outcome"] != "STARVED"]
    avoidable = [d for d in crashes if d["safe"]]
    return {
        "deaths": len(log),
        "by_outcome": dict(Counter(d["outcome"] for d in log)),
        "crashes_with_a_safe_move": len(avoidable),
        "crashes_with_no_safe_move": len(crashes) - len(avoidable),
        "avoidable_crash_action": dict(Counter(d["action"] for d in avoidable)),
        "avoidable_straight_into_adjacent_obstacle": sum(
            d["action"] == "straight" and d["feats"][0, F["obstacle_front"]] >= 1.0 for d in avoidable),
        "avoidable_turned_into_adjacent_obstacle": sum(d["action"] != "straight" for d in avoidable),
        "mean_length_at_death": round(float(np.mean([d["length"] for d in log])), 2) if log else None,
        "fraction_avoidable": round(len(avoidable) / n, 3),
    }


def agreement(policy, views) -> dict:
    """Same view to 64 fresh brains: share choosing the most common action."""
    fr = []
    for v in views:
        policy.reset(np.ones(B, bool))
        a = policy.act(np.repeat(v[None], B, 0))
        fr.append(np.bincount(a, minlength=3).max() / B)
    fr = np.array(fr)
    return {"views": len(fr), "mean_agreement": round(float(fr.mean()), 3),
            "min_agreement": round(float(fr.min()), 3),
            "views_below_90pct": int((fr < 0.9).sum())}


def main():
    from brain.connectome_loader import load_connectome
    cfg = train_config(20)
    champ = np.load(EVO / "fly_main" / "champion.npy")
    genomes = np.repeat(champ[None], B, 0)
    conn = load_connectome()
    res = {}

    fly = FlyPolicy(conn, genomes)
    score, log, views = play_logged(fly, cfg)
    res["fly_24ms"] = score | summarise(log)
    fatal = [d["feats"] for d in log if d["safe"]][:24]
    res["fly_agreement_random_views"] = agreement(fly, views)
    if fatal:
        res["fly_agreement_views_before_avoidable_crash"] = agreement(fly, fatal)
    score, log, _ = play_logged(fly, cfg, reset_each=True)
    res["fly_24ms_reset_every_decision"] = score | summarise(log)
    del fly

    fly48 = FlyPolicy(conn, genomes, window_ms=48.0)
    score, log, _ = play_logged(fly48, cfg)
    res["fly_48ms"] = score | summarise(log)
    del fly48

    ann = AnnPolicy(np.repeat(np.load(EVO / "ann_main" / "champion.npy")[None], B, 0))
    score, log, _ = play_logged(ann, cfg)
    res["ann"] = score | summarise(log)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "death_diagnosis.json").write_text(json.dumps(res, indent=1, default=int))
    print(json.dumps(res, indent=1, default=int))


if __name__ == "__main__":
    main()
