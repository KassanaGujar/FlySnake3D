"""Before evolving: does each visual population reach the steering readout at all?

Stimulate one encoder population in the LEFT eye at 150 Hz and record the
decoder's readout neurons on both sides, with the settings evolution uses
(dt = 0.6 ms). One condition per batch column. This tells us, without any
evolution, which inputs can move the decoder and in which direction; a
population that reaches no readout neuron cannot contribute to steering.

Also repeats the measurement at the published dt = 0.1 ms as a check that the
coarse step does not change the picture.

Run:  python -m experiments.interface_probe
Writes results/stage3/interface_probe.csv
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from brain.connectome_loader import load_connectome
from brain.event_simulator import EventLIF
from brain.motor_decoder import READOUT_TYPES
from brain.neural_simulator import LIFParams
from brain.visual_encoder import POPULATIONS

OUT = Path(__file__).resolve().parents[1] / "results" / "stage3"
T_MS, RATE, REPS = 240.0, 150.0, 4


def probe(c, dt: float) -> pd.DataFrame:
    B = len(POPULATIONS) * REPS
    sim = EventLIF(c.W, replace(LIFParams(), dt=dt), batch=B, seed=0)
    for k, pop in enumerate(POPULATIONS):
        idx = c.indices_of_type(pop, side="left")
        r = np.zeros((len(idx), B), np.float32)
        r[:, k * REPS:(k + 1) * REPS] = RATE
        sim.set_poisson_input(idx, r)
    counts = sim.run(int(round(T_MS / dt)))
    rows = []
    for k, pop in enumerate(POPULATIONS):
        row = {"stimulated_left": pop, "dt_ms": dt}
        for d in READOUT_TYPES:
            for side in ("left", "right"):
                ii = c.indices_of_type(d, side=side)
                hz = counts[ii][:, k * REPS:(k + 1) * REPS].mean() / (T_MS / 1000.0)
                row[f"{d}_{side[0].upper()}"] = round(float(hz), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    c = load_connectome()
    df = pd.concat([probe(c, 0.6), probe(c, 0.1)])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "interface_probe.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
