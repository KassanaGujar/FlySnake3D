"""Do the pathways a learning fly needs actually carry signal in this model?

Before building dopamine learning and smell into the agent, check in the
unmodified connectome (LIF model, dt 0.6 ms, 240 ms, 3 seeds) that:

  smell   olfactory receptor neurons (ORNs) -> projection neurons (ALPN) -> Kenyon cells
  vision  visual projection neurons (the encoder's 8 types)    -> Kenyon cells
  output  Kenyon cells -> mushroom body output neurons (MBON)  -> steering readout DNs
  reward  PAM / PPL1 dopamine neurons fire when driven

Also counts direct synapses between these groups, straight from the wiring.
Stage 0 found that photoreceptor input never reaches the central brain, so
none of this is assumed.

Run:  python -m experiments.learning_pathway_probe
Writes results/stage7/learning_pathway_probe.json
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from brain.connectome_loader import load_connectome
from brain.event_simulator import EventLIF
from brain.motor_decoder import READOUT_TYPES
from brain.neural_simulator import LIFParams
from brain.visual_encoder import POPULATIONS

OUT = Path(__file__).resolve().parents[1] / "results" / "stage7"


def groups(c):
    a = c.annotations
    cc, ct = a.cell_class.fillna(""), a.cell_type.fillna("")
    idx = lambda m: np.flatnonzero(m.to_numpy())
    return {
        "ORN": idx(ct.str.startswith("ORN_")),
        "ALPN": idx(cc == "ALPN"),
        "KC": idx(cc == "Kenyon_Cell"),
        "MBON": idx(cc == "MBON"),
        "PAM": idx(ct.str.startswith("PAM")),
        "PPL1": idx(ct.str.startswith("PPL1")),
        "visual_in": np.concatenate([c.indices_of_type(p) for p in POPULATIONS]),
        "readout_DN": np.concatenate([c.indices_of_type(d) for d in READOUT_TYPES]),
    }


def main():
    c = load_connectome()
    G = groups(c)
    W = c.W.tocsr()            # W[post, pre]? check orientation below
    res = {"group_sizes": {k: int(len(v)) for k, v in G.items()}}

    # Direct synapse weights between groups (sum of |signed synapse counts|).
    def syn(pre, post):
        sub = c.W[np.ix_(G[post], G[pre])] if c.W.shape[0] == c.W.shape[1] else None
        return float(abs(sub).sum())
    # Find orientation: EventLIF treats column j as presynaptic neuron j, so W[post, pre].
    pairs = [("ORN", "ALPN"), ("ALPN", "KC"), ("visual_in", "KC"), ("KC", "MBON"),
             ("PAM", "MBON"), ("PPL1", "MBON"), ("PAM", "KC"), ("PPL1", "KC"),
             ("MBON", "readout_DN"), ("visual_in", "readout_DN")]
    res["direct_synapses"] = {f"{a}->{b}": syn(a, b) for a, b in pairs}
    print(res["direct_synapses"], flush=True)

    stimuli = {"smell (all ORNs 200 Hz)": G["ORN"], "vision (encoder neurons 200 Hz)": G["visual_in"],
               "Kenyon cells 50 Hz": G["KC"], "PAM 100 Hz": G["PAM"], "PPL1 100 Hz": G["PPL1"]}
    rates = {"Kenyon cells 50 Hz": 50.0, "PAM 100 Hz": 100.0, "PPL1 100 Hz": 100.0}
    res["activity"] = {}
    for name, ii in stimuli.items():
        rows = []
        for seed in (0, 1, 2):
            s = EventLIF(c.W, replace(LIFParams(), dt=0.6), batch=1, seed=seed)
            s.set_poisson_input(ii, rates.get(name, 200.0))
            cnt = s.run(400)[:, 0]
            rows.append({k: int(cnt[v].sum()) for k, v in G.items()} |
                        {"KC_active": int((cnt[G["KC"]] > 0).sum()),
                         "MBON_active": int((cnt[G["MBON"]] > 0).sum())})
        res["activity"][name] = rows
        print(name, rows, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "learning_pathway_probe.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
