"""Diagnose how far photoreceptor stimulation propagates in the LIF model.

Stimulates all left R1-6 (or R7/R8) and reports the mean firing rate of each
downstream visual cell type, ordered by the canonical visual hierarchy.
Writes results/milestone0/retina_propagation.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from brain.connectome_loader import load_connectome
from brain.neural_simulator import LIFParams, TorchLIF

OUT = Path(__file__).resolve().parents[1] / "results" / "milestone0"
CHAIN = ["R1-6", "R7", "R8", "L1", "L2", "L3", "L4", "L5", "C2", "C3", "Mi1", "Tm3", "Mi4", "Mi9",
         "Tm1", "Tm2", "Tm4", "Tm9", "T4a", "T5a", "LPLC2", "LC4", "LPLC1", "LC9", "LC11", "DNp01"]

def main(t_ms=200.0, rate=150.0):
    c = load_connectome(); p = LIFParams()
    conds = {"R1-6": c.indices_of_type("R1-6", "left"),
             "R7R8": np.r_[c.indices_of_type("R7", "left"), c.indices_of_type("R8", "left")],
             "L1L2L3": np.r_[c.indices_of_type("L1", "left"), c.indices_of_type("L2", "left"), c.indices_of_type("L3", "left")]}
    B = len(conds)
    sim = TorchLIF(c.W, p, batch=B, seed=3)
    for k, idx in enumerate(conds.values()):
        r = np.zeros((len(idx), B), np.float32); r[:, k] = rate
        sim.set_poisson_input(idx, r)
    counts = sim.run(int(t_ms / p.dt)) / (t_ms / 1000)
    res = {}
    for k, name in enumerate(conds):
        row = {}
        for t in CHAIN:
            try:
                idx = c.indices_of_type(t, "left")
            except KeyError:
                continue
            row[t] = round(float(counts[idx, k].mean()), 1)
        row["_total_active_neurons"] = int((counts[:, k] > 0).sum())
        res[f"stim_{name}_left"] = row
    # synaptic sign of the first relay (data, not assumption)
    for pre, post in [("R1-6", "L1"), ("R1-6", "L2"), ("R1-6", "L3"), ("L1", "Mi1"), ("L2", "Tm1"), ("L3", "Mi9")]:
        res[f"signed_syn_{pre}->{post}"] = c.synapses_between(pre, post)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "retina_propagation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
