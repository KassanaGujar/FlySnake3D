"""Evolve a Snake controller. The same loop, budget and features for every controller.

  --controller fly            FlyWire connectome between encoder and decoder
  --controller fly_shuffled   same, but the wiring is shuffled (control: does
                              the real wiring matter, or would any network of
                              these neurons do?)
  --controller ann            small tanh network on the same features (control)

Every generation all candidates play the same seeds (common random numbers),
so differences between candidates are not due to luckier food. The champion is
re-evaluated every generation on fresh seeds (see evolution/es.py).

Run:     python -m experiments.evolve --controller fly --generations 60 --run fly_main
Resume:  same command; it continues from the last checkpoint in results/evolution/<run>/
Writes   results/evolution/<run>/{config.json, history.csv, checkpoint.npz, champion_genXXXX.npy}
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from agents.policies import ANN_INIT_MEAN, FLY_INIT_MEAN, AnnPolicy, FlyPolicy
from evolution.es import ES
from evolution.rollout import play_budget
from game.environment import GameConfig

ROOT = Path(__file__).resolve().parents[1]
SHUFFLE_SEED = 12345


def shuffled_wiring(W, seed: int = SHUFFLE_SEED):
    """Give every neuron's inputs to a random other neuron (row permutation).

    Each presynaptic neuron keeps its number of outgoing synapses, their
    strengths and signs; the network keeps its in-degree distribution. Only
    WHO receives the input changes, which destroys the specific pathways.
    """
    perm = np.random.default_rng(seed).permutation(W.shape[0])
    return W.tocsr()[perm]


def train_config(size: int) -> GameConfig:
    # Short starvation cap during training: a snake that circles is restarted
    # after 100 moves instead of 400, so less GPU time is spent on it.
    return GameConfig(width=size, height=size, max_steps_without_food=100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["fly", "fly_shuffled", "ann"], default="fly")
    ap.add_argument("--run", default=None)
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--decisions", type=int, default=300)
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.6)
    ap.add_argument("--window_ms", type=float, default=24.0)
    ap.add_argument("--sigma0", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--death_penalty", type=float, default=0.5,
                    help="fitness = food - death_penalty * deaths")
    ap.add_argument("--init_from", default=None,
                    help="start the search at a saved genome (.npy) instead of the default")
    a = ap.parse_args()
    run = ROOT / "results" / "evolution" / (a.run or a.controller)
    run.mkdir(parents=True, exist_ok=True)
    cfg = train_config(a.size)

    is_fly = a.controller.startswith("fly")
    init = np.load(a.init_from) if a.init_from else (FLY_INIT_MEAN if is_fly else ANN_INIT_MEAN)
    es = ES(init, sigma0=a.sigma0, pop=a.pop, seed=a.seed)
    hist_path, ck_path = run / "history.csv", run / "checkpoint.npz"
    history = []
    if ck_path.exists():
        es.load(np.load(ck_path, allow_pickle=False))
        history = pd.read_csv(hist_path).to_dict("records")[: es.generation]
        print(f"resumed {run.name} at generation {es.generation}", flush=True)
    else:
        (run / "config.json").write_text(json.dumps(vars(a) | {"genome_size": len(es.mean)}, indent=1))

    policy = None
    if is_fly:
        from brain.connectome_loader import load_connectome
        conn = load_connectome()
        W = shuffled_wiring(conn.W) if a.controller == "fly_shuffled" else None
        policy = FlyPolicy(conn, es.ask(), dt=a.dt, window_ms=a.window_ms, device=a.device,
                           seed=a.seed, W=W)

    while es.generation < a.generations:
        gen = es.generation
        t0 = time.time()
        genomes = es.ask()
        if is_fly:
            policy.set_genomes(genomes)
        else:
            policy = AnnPolicy(genomes)
        seeds = np.full(a.pop, 10_000 + a.seed * 1_000_000 + gen)   # same games for everyone
        res = play_budget(policy, cfg, seeds, a.decisions)
        fit = res["food"] - a.death_penalty * res["deaths"]
        info = es.tell(fit)
        best = int(np.argmax(fit))
        row = {"generation": gen, **info,
               "champion_food": float(res["food"][0]), "champion_deaths": float(res["deaths"][0]),
               "best_food": float(res["food"][best]), "best_deaths": float(res["deaths"][best]),
               "mean_food": float(res["food"].mean()), "mean_deaths": float(res["deaths"].mean()),
               "max_length": int(res["max_length"].max()),
               "frac_left": float(res["action_fractions"][0]),
               "frac_straight": float(res["action_fractions"][1]),
               "frac_right": float(res["action_fractions"][2]),
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        pd.DataFrame(history).to_csv(hist_path, index=False)
        np.savez(ck_path, **es.state())
        if row["champion_replaced"] or gen % 10 == 0:
            np.save(run / f"champion_gen{gen:04d}.npy", es.champion)
        np.save(run / "champion.npy", es.champion)
        print(f"gen {gen:3d}  champ {row['champion_fitness']:6.1f}  best {row['best_fitness']:6.1f}  "
              f"mean {row['mean_fitness']:6.1f}  food(mean) {row['mean_food']:5.1f}  "
              f"deaths(mean) {row['mean_deaths']:5.1f}  sigma {row['sigma_mean']:.3f}  "
              f"{row['seconds']:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
