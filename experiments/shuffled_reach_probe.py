"""Does visual input reach the steering readout on real vs shuffled wiring?

Every encoder input neuron (all eight visual populations, both eyes) is driven
at 200 Hz for 240 ms (dt = 0.6 ms, the setting evolution uses), on the real
FlyWire wiring and on the shuffled wiring used by the fly_shuffled control
(row permutation, see experiments/evolve.py). We count how many neurons fire
and how many spikes reach the decoder's descending neurons.

If the readout receives nothing, the shuffled control cannot steer whatever
evolution does to the interface, and its flat learning curve says that the
specific wiring is what links vision to steering, not that shuffled wiring
"steers worse".

Run:  python -m experiments.shuffled_reach_probe
Writes results/stage5/shuffled_reach_probe.json
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
from experiments.evolve import shuffled_wiring

OUT = Path(__file__).resolve().parents[1] / "results" / "stage5"


def main():
    c = load_connectome()
    dn = np.concatenate([c.indices_of_type(d) for d in READOUT_TYPES])
    vis = np.concatenate([c.indices_of_type(p) for p in POPULATIONS])
    res = {"stimulus": "all encoder input neurons at 200 Hz, 240 ms, dt 0.6 ms", "seeds": [0, 1, 2]}
    for name, W in [("real", c.W), ("shuffled", shuffled_wiring(c.W))]:
        rows = []
        for seed in res["seeds"]:
            s = EventLIF(W, replace(LIFParams(), dt=0.6), batch=1, seed=seed)
            s.set_poisson_input(vis, 200.0)
            cnt = s.run(400)[:, 0]
            rows.append({"neurons_active": int((cnt > 0).sum()), "total_spikes": int(cnt.sum()),
                         "readout_spikes": int(cnt[dn].sum())})
        res[name] = rows
        print(name, rows, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shuffled_reach_probe.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
