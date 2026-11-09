"""Validate the PyTorch LIF backend against the published Brian2 model.

Both backends are stochastic (Poisson input), so we cannot expect identical
spike trains. Following the approach used by eonsystemspbc/fly-brain, we
compare *trial-averaged firing rates*:

  * Jaccard overlap of the sets of neurons that fire at all
  * Pearson correlation of per-neuron mean spike counts
  * ratio of total spike counts

Stimulus: left LPLC2 (looming-sensitive visual projection neurons) at 150 Hz.

Run:  python -m experiments.validate_backends  [--trials 5] [--t_ms 200]
Writes results/milestone0/backend_validation.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from brain.connectome_loader import load_connectome
from brain.neural_simulator import LIFParams, TorchLIF, brian2_reference_counts

OUT = Path(__file__).resolve().parents[1] / "results" / "milestone0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--t_ms", type=float, default=200.0)
    ap.add_argument("--stim", default="LPLC2")
    ap.add_argument("--side", default="left")
    ap.add_argument("--rate", type=float, default=150.0)
    a = ap.parse_args()

    c = load_connectome()
    stim = c.indices_of_type(a.stim, side=a.side)
    p = LIFParams()
    n_steps = int(round(a.t_ms / p.dt))

    t0 = time.time()
    sim = TorchLIF(c.W, p, batch=a.trials, seed=1)
    sim.set_poisson_input(stim, a.rate)
    torch_counts = sim.run(n_steps).astype(float)          # (N, trials)
    t_torch = time.time() - t0

    b2 = []
    t0 = time.time()
    for k in range(a.trials):
        b2.append(brian2_reference_counts(c.W, stim, a.rate, a.t_ms, p, seed=100 + k))
        print(f"brian2 trial {k+1}/{a.trials} done ({time.time()-t0:.0f}s)", flush=True)
    t_b2 = time.time() - t0
    b2 = np.stack(b2, 1).astype(float)

    mt, mb = torch_counts.mean(1), b2.mean(1)
    act_t, act_b = mt > 0, mb > 0
    union = act_t | act_b
    res = {
        "stimulus": f"{a.stim} ({a.side}) @ {a.rate} Hz, {len(stim)} neurons",
        "t_ms": a.t_ms, "trials": a.trials,
        "active_torch": int(act_t.sum()), "active_brian2": int(act_b.sum()),
        "jaccard_active": float((act_t & act_b).sum() / max(union.sum(), 1)),
        "pearson_rates_union": float(np.corrcoef(mt[union], mb[union])[0, 1]),
        "total_spike_ratio_torch_over_brian2": float(mt.sum() / max(mb.sum(), 1e-9)),
        "wall_s_torch_all_trials_batched": t_torch,
        "wall_s_brian2_numpy_all_trials": t_b2,
    }
    # Same comparison restricted to neurons that are non-stimulated (pure network response)
    mask = union.copy(); mask[stim] = False
    res["pearson_rates_network_only"] = float(np.corrcoef(mt[mask], mb[mask])[0, 1]) if mask.sum() > 2 else None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "backend_validation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
