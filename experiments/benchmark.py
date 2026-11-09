"""Benchmark the whole-brain simulator and extrapolate evolution cost.

Measures wall-clock seconds per simulated second for batches of B brains on
this machine, then projects the cost of an evolutionary run from explicit,
editable assumptions (ASSUMPTIONS below). The projection is arithmetic, not
a guess: change the assumptions and re-run.

Run:  python -m experiments.benchmark
Writes results/milestone0/benchmark.json
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from brain.connectome_loader import load_connectome
from brain.neural_simulator import LIFParams, TorchLIF

OUT = Path(__file__).resolve().parents[1] / "results" / "milestone0"

ASSUMPTIONS = {
    "neural_ms_per_game_tick": 50,       # brain time simulated per Snake move
    "ticks_per_episode": 500,            # typical early episode on 20x20 (cap is configurable)
    "episodes_per_candidate": 5,         # needed to average out food placement luck
    "population_size": 64,
}


def bench(c, B, n_steps=500, warm=50):
    import torch
    sim = TorchLIF(c.W, LIFParams(), batch=B, seed=0)
    stim = c.indices_of_type("LPLC2", "left")
    sim.set_poisson_input(stim, 150.0)
    for _ in range(warm):
        sim.step()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n_steps):
        sim.step()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t
    sim_s = n_steps * sim.p.dt / 1000
    mem = torch.cuda.max_memory_allocated() / 1e9
    del sim
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return {"batch": B, "wall_s_per_sim_s_total": wall / sim_s,
            "wall_s_per_sim_s_per_brain": wall / sim_s / B, "peak_gpu_GB": mem}


def project(per_brain_s_per_sim_s, a=ASSUMPTIONS):
    sim_s_per_candidate = a["neural_ms_per_game_tick"] / 1000 * a["ticks_per_episode"] * a["episodes_per_candidate"]
    s_per_gen = sim_s_per_candidate * a["population_size"] * per_brain_s_per_sim_s
    out = {"simulated_brain_seconds_per_candidate": sim_s_per_candidate,
           "wall_s_per_generation": s_per_gen}
    for g in (1_000, 10_000, 100_000, 1_000_000):
        out[f"wall_days_{g}_generations"] = s_per_gen * g / 86400
    return out


def main():
    import torch
    c = load_connectome()
    rows = []
    for B in (1, 4, 16, 32, 64):
        try:
            r = bench(c, B)
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch": B, "error": "OOM"}); break
        rows.append(r)
        print(r, flush=True)
    best = min(r["wall_s_per_sim_s_per_brain"] for r in rows if "error" not in r)
    res = {"device": torch.cuda.get_device_name(0), "cpu": platform.processor(),
           "network": c.source, "dt_ms": LIFParams().dt, "results": rows,
           "best_wall_s_per_sim_s_per_brain": best,
           "assumptions": ASSUMPTIONS, "projection_at_best_batch": project(best)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "benchmark.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
