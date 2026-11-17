"""Stage 7 calibration: adaptation + Kenyon cell sparseness.

The unmodified LIF model locks the olfactory system at maximum firing after any
odour input (experiments/olfactory_lockup_probe.py). Two disclosed changes are
calibrated here, both NOT part of Shiu et al. 2024:

  adaptation        every neuron: +ADAPT_MV threshold per spike, decaying with ADAPT_TAU_MS
  KC threshold      extra threshold (mV) for Kenyon cells only
  APL inhibition    factor on APL -> Kenyon cell synapses

Target (real flies; from memory, needs checking: Turner et al. 2008; Honegger
et al. 2011; Lin et al. 2014): about 5-10% of Kenyon cells respond to an
odour, different odours recruit largely different cells, and activity stops
when the odour stops. Measured with single glomeruli (DM1, VA2, DM2) at
50 Hz for 48 ms, then 120 ms without input, 2 seeds.

Run:  python -m experiments.kc_sparseness_calibration
Writes results/stage7/kc_sparseness_calibration.json
"""
from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from brain.connectome_loader import load_connectome
from brain.event_simulator import EventLIF
from brain.motor_decoder import READOUT_TYPES
from brain.mushroom_body import ODOUR_GLOMERULI, _type_index, mushroom_body_wiring
from brain.neural_simulator import LIFParams
from brain.visual_encoder import POPULATIONS

ADAPT_MV, ADAPT_TAU_MS = 2.0, 1000.0
OUT = Path(__file__).resolve().parents[1] / "results" / "stage7"


def build(c, kc, apl, kc_mv, apl_x, seed):
    s = EventLIF(c.W, replace(LIFParams(), dt=0.6), batch=1, seed=seed)
    s.enable_adaptation(ADAPT_MV, ADAPT_TAU_MS)
    if kc_mv:
        s.set_threshold_offset(kc, kc_mv)
    if apl_x != 1.0:
        s.scale_synapses(apl, kc, apl_x)
    return s


def main():
    c = load_connectome()
    mb = mushroom_body_wiring(c)
    kc, mbon = mb["kc"], mb["mbon"]
    apl = _type_index(c, "APL")
    glom = {g: np.concatenate([_type_index(c, "ORN_" + g, s) for s in ("left", "right")])
            for g in ("DM1", "VA2", "DM2")}
    six = np.concatenate([_type_index(c, "ORN_" + t, s) for t in ODOUR_GLOMERULI for s in ("left", "right")])
    vis = np.concatenate([c.indices_of_type(p) for p in POPULATIONS])
    dn = np.concatenate([c.indices_of_type(d) for d in READOUT_TYPES])
    rows = []
    for kc_mv, apl_x in itertools.product([0, 4, 8, 12, 16], [1.0, 4.0]):
        r = {"kc_threshold_mv": kc_mv, "apl_factor": apl_x}
        act, after, sets = [], [], {}
        for seed in (0, 1):
            for g, idx in glom.items():
                s = build(c, kc, apl, kc_mv, apl_x, seed)
                s.set_poisson_input(idx, 50.0)
                d = s.run(80)[:, 0]
                s.clear_inputs(); s.run(200)
                aft = s.run(80)[:, 0]
                on = set(np.flatnonzero(d[kc] > 0))
                act.append(len(on)); after.append(int((aft[kc] > 0).sum()))
                sets[(seed, g)] = on
        jac = []
        for seed in (0, 1):
            for g1, g2 in itertools.combinations(glom, 2):
                a_, b_ = sets[(seed, g1)], sets[(seed, g2)]
                jac.append(len(a_ & b_) / max(1, len(a_ | b_)))
        s = build(c, kc, apl, kc_mv, apl_x, 0)
        s.set_poisson_input(six, 50.0)
        d6 = s.run(80)[:, 0]
        s = build(c, kc, apl, kc_mv, apl_x, 0)
        s.set_poisson_input(vis, 200.0)
        dv = s.run(80)[:, 0]
        r |= {"kc_active_pct_single_glomerulus": round(100 * float(np.mean(act)) / len(kc), 1),
              "kc_active_after_off": float(np.mean(after)),
              "jaccard_between_glomeruli": round(float(np.mean(jac)), 2),
              "kc_active_pct_six_glomeruli": round(100 * float((d6[kc] > 0).sum()) / len(kc), 1),
              "mbon_spikes_six": int(d6[mbon].sum()), "dn_spikes_six": int(d6[dn].sum()),
              "dn_spikes_vision": int(dv[dn].sum())}
        rows.append(r)
        print(r, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kc_sparseness_calibration.json").write_text(json.dumps(
        {"adapt_mv": ADAPT_MV, "adapt_tau_ms": ADAPT_TAU_MS, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
