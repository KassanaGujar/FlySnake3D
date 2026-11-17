"""Olfactory lock-up in the unmodified LIF model (stage 7 finding).

Drives olfactory receptor neurons (one glomerulus, or the six food-odour
glomeruli) or the stage 3-6 visual inputs for 48 ms at 2-200 Hz, switches the
input off, waits 120 ms, and counts active Kenyon cells during and after.
Result (3 seeds): any olfactory input, even 2 Hz into DM1 alone, leaves ~3,840
of 5,177 Kenyon cells, the antennal lobe, APL and DPM firing at the model's
maximum rate indefinitely; visual input never does. Removing all KC->KC
synapses does not prevent it. See docs/DEVLOG.md (17 Nov 2026).

Run:  python -m experiments.olfactory_lockup_probe
"""
from dataclasses import replace
import numpy as np
from brain.connectome_loader import load_connectome
from brain.event_simulator import EventLIF
from brain.neural_simulator import LIFParams
from brain.mushroom_body import ODOUR_GLOMERULI, mushroom_body_wiring, _type_index
from brain.visual_encoder import POPULATIONS

c = load_connectome()
mb = mushroom_body_wiring(c)
kc = mb["kc"]
P = replace(LIFParams(), dt=0.6)
one = np.concatenate([_type_index(c, "ORN_DM1", s) for s in ("left", "right")])
six = np.concatenate([_type_index(c, "ORN_" + t, s) for t in ODOUR_GLOMERULI for s in ("left", "right")])
vis = np.concatenate([c.indices_of_type(p) for p in POPULATIONS])
OUT = __import__("pathlib").Path(__file__).resolve().parents[1] / "results" / "stage7"
rows = {}
print("stimulus, rate -> KCs active during (first 48 ms) | 120 ms after input off, over 3 seeds")
for label, idx in [("DM1 only", one), ("6 food glomeruli", six), ("vision (stage 3-6 inputs)", vis)]:
    for rate in (2.0, 5.0, 10.0, 200.0):
        out = []
        for seed in range(3):
            s = EventLIF(c.W, P, batch=1, seed=seed)
            s.set_poisson_input(idx, rate)
            d = s.run(80)[:, 0]
            s.clear_inputs()
            s.run(200)
            after = s.run(80)[:, 0]
            out.append(f"{int((d[kc] > 0).sum())}|{int((after[kc] > 0).sum())}")
        print(f"  {label:26s} {rate:5.0f} Hz:", "  ".join(out), flush=True)
        rows[f"{label} @ {rate:.0f} Hz"] = out
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "olfactory_lockup_probe.json").write_text(__import__("json").dumps({"format": "KCs active during | after", "rows": rows}, indent=1))
