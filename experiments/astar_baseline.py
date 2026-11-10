"""Stage 2: A* baselines on Snake, and reproduction of long-body self-trapping.

For each planner in the ladder, play N seeded episodes and record per-episode
results (CSV) plus a few replays of self-trap deaths for viewing in Godot.
The same seeds are used for every planner, so food placement is matched.

Run:  python -m experiments.astar_baseline [--episodes 100] [--size 20]
Writes results/stage2/{episodes.csv, summary.json, by_length.csv}
       results/replays/stage2/*.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents.astar_agent import AGENTS
from game.environment import GameConfig
from game.runner import run_episode

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--replays", type=int, default=3)
    a = ap.parse_args()
    cfg = GameConfig(width=a.size, height=a.size, max_steps=50 * a.size * a.size)
    out = ROOT / "results" / "stage2"
    rep = ROOT / "results" / "replays" / "stage2"
    out.mkdir(parents=True, exist_ok=True)

    rows, deaths = [], []
    for name, cls in AGENTS.items():
        saved = 0
        for seed in range(a.episodes):
            r = run_episode(cls(), cfg, seed)
            rows.append(r.summary())
            # where in the snake's life does it die, and was it trapped first?
            if r.outcome in ("SELF", "WALL"):
                deaths.append({"agent": name, "length_at_death": r.final_length,
                               "self_trap": r.self_trap_death})
            if r.self_trap_death and saved < a.replays:
                run_episode(cls(), cfg, seed, replay_path=rep / f"{name}_seed{seed}.jsonl")
                saved += 1
        print(f"{name:18s} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "episodes.csv", index=False)
    cells = a.size * a.size
    summary = {}
    for name, d in df.groupby("agent", sort=False):
        died = d.outcome.isin(["SELF", "WALL"])
        summary[name] = {
            "episodes": len(d),
            "food_mean": round(d.food_eaten.mean(), 1),
            "food_median": float(d.food_eaten.median()),
            "food_ci95": [round(float(x), 1) for x in
                          np.percentile([np.random.default_rng(i).choice(d.food_eaten, len(d)).mean()
                                         for i in range(2000)], [2.5, 97.5])],
            "final_length_mean": round(d.final_length.mean(), 1),
            "max_length": int(d.final_length.max()),
            "board_fill_mean_%": round(100 * d.final_length.mean() / cells, 1),
            "outcomes": d.outcome.value_counts().to_dict(),
            "self_collision_rate": round((d.outcome == "SELF").mean(), 3),
            "wall_collision_rate": round((d.outcome == "WALL").mean(), 3),
            "self_trap_death_rate": round(d.self_trap_death.mean(), 3),
            "trap_to_death_moves_median": (float(d.moves_trap_to_death.dropna().median())
                                           if d.moves_trap_to_death.notna().any() else None),
            "steps_mean": round(d.steps.mean(), 1),
            "decision_ms_mean": round(d.mean_decision_ms.mean(), 3),
        }
    dd = pd.DataFrame(deaths)
    if len(dd):
        bins = [0, 25, 50, 100, 150, 200, 300, cells + 1]
        dd["length_bin"] = pd.cut(dd.length_at_death, bins, right=False).astype(str)
        dd.groupby(["agent", "length_bin"], sort=False).agg(
            deaths=("self_trap", "size"), self_trap_fraction=("self_trap", "mean")
        ).reset_index().to_csv(out / "by_length.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"config": vars(a), "game": cfg.__dict__ | {"obstacles": []},
                                                  "agents": summary}, indent=2, default=str))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
