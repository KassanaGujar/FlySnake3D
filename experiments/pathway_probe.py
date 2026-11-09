"""Milestone 0: can the connectome be stimulated, simulated and read?

Three probes, all on the unmodified FlyWire v783 / Shiu LIF model:

A. Visual population -> descending neurons (DNs).
   Stimulate one visual population on one side; record all 1,303 DNs.
   Tells us which DN types respond to which visual input and whether the
   response is lateralised (left stimulus -> different DNs than right), which
   is what a left/right turn decoder will need.

B. Dose-response: LPLC2 (left) rate -> giant fiber DNp01 rate.
   The prototype of the "stimulus intensity vs neural response" curve that the
   looming/evasion analysis (research question 3, section 11) needs.

C. Latency: time from stimulus onset to first DNp01 spike.

Every condition is one column of the batch, so all conditions run in parallel
on the GPU in a single simulation.

Run:  python -m experiments.pathway_probe [--reps 3] [--t_ms 200]
Writes results/milestone0/pathway_probe.json and dn_responses.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from brain.connectome_loader import load_connectome
from brain.neural_simulator import LIFParams, TorchLIF

OUT = Path(__file__).resolve().parents[1] / "results" / "milestone0"

# Visual populations to probe. These are published FlyWire cell types; the
# biological role in the comment is from the cited literature, NOT assumed to
# carry over to Snake.
POPULATIONS = {
    "R1-6": ["R1-6"],                         # outer photoreceptors (motion/luminance)
    "R7R8": ["R7", "R8"],                     # inner photoreceptors (colour/UV)
    "T4T5": ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"],  # direction-selective
    "LPLC2": ["LPLC2"],                       # looming (Klapoetke 2017; Ache 2019)
    "LC4": ["LC4"],                           # looming / escape timing (von Reyn 2017)
    "LPLC1": ["LPLC1"],                       # object/looming, collision avoidance (Tanaka & Clark 2022)
    "LC9": ["LC9"],
}


def build_conditions(c):
    conds = [("baseline", None, None)]
    for name, types in POPULATIONS.items():
        for side in ("left", "right"):
            idx = np.concatenate([c.indices_of_type(t, side=side) for t in types])
            conds.append((f"{name}_{side}", idx, side))
    return conds


def probe_A(c, reps, t_ms, rate, p):
    conds = build_conditions(c)
    B = len(conds) * reps
    sim = TorchLIF(c.W, p, batch=B, seed=7)
    for k, (name, idx, _) in enumerate(conds):
        if idx is None:
            continue
        r = np.zeros((len(idx), B), np.float32)
        r[:, k * reps:(k + 1) * reps] = rate
        sim.set_poisson_input(idx, r)
    dn = c.indices_of_super_class("descending")
    counts = sim.run(int(t_ms / p.dt), record=dn)                 # (n_dn, B)
    rates = counts.reshape(len(dn), len(conds), reps).mean(2) / (t_ms / 1000)  # Hz
    a = c.annotations.iloc[dn]
    df = pd.DataFrame(rates, columns=[n for n, _, _ in conds])
    df.insert(0, "root_id", a.root_id.values)
    df.insert(1, "cell_type", a.cell_type.values)
    df.insert(2, "side", a.side.values)

    summary = {}
    for name, idx, side in conds[1:]:
        col = df[name] - df["baseline"]
        top = df.assign(dr=col).sort_values("dr", ascending=False).head(8)
        summary[name] = {
            "n_stimulated": int(len(idx)),
            "n_DN_responding_>5Hz": int((col > 5).sum()),
            "top_DNs": [f"{r.cell_type}({r.side[0] if isinstance(r.side, str) else '?'}):{r.dr:.0f}Hz"
                        for r in top.itertuples()],
        }
    # Lateralisation per population: DNs whose response differs most L vs R stim.
    lat = {}
    for name in POPULATIONS:
        L, R = df[f"{name}_left"], df[f"{name}_right"]
        li = (L - R)
        order = np.argsort(-np.abs(li.values))[:6]
        lat[name] = [f"{df.cell_type.iloc[i]}({str(df.side.iloc[i])[0]}): L-R={li.iloc[i]:+.0f}Hz"
                     for i in order if abs(li.iloc[i]) > 1]
    return df, summary, lat


def probe_B_C(c, reps, t_ms, p):
    rates = [0, 10, 25, 50, 75, 100, 150, 200]
    B = len(rates) * reps
    stim = c.indices_of_type("LPLC2", side="left")
    gf = np.r_[c.indices_of_type("DNp01", "left"), c.indices_of_type("DNp01", "right")]
    sim = TorchLIF(c.W, p, batch=B, seed=11)
    r = np.repeat(np.array(rates, np.float32), reps)[None, :].repeat(len(stim), 0)
    sim.set_poisson_input(stim, r)
    import torch
    gf_t = torch.as_tensor(gf, device=sim.device)
    n_steps = int(t_ms / p.dt)
    first = np.full((2, B), np.nan)
    counts = np.zeros((2, B))
    for s in range(n_steps):
        spk = sim.step()[gf_t].cpu().numpy()
        counts += spk
        new = spk & np.isnan(first)
        first[new] = s * p.dt
    hz = counts / (t_ms / 1000)
    dose = {f"{rt}Hz": {"DNp01_left_Hz": float(hz[0, i*reps:(i+1)*reps].mean()),
                        "DNp01_right_Hz": float(hz[1, i*reps:(i+1)*reps].mean()),
                        "first_spike_ms_left": float(np.nanmean(first[0, i*reps:(i+1)*reps]))
                        if not np.all(np.isnan(first[0, i*reps:(i+1)*reps])) else None}
            for i, rt in enumerate(rates)}
    return dose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--t_ms", type=float, default=200.0)
    ap.add_argument("--rate", type=float, default=150.0)
    a = ap.parse_args()
    p = LIFParams()
    c = load_connectome()
    df, summary, lat = probe_A(c, a.reps, a.t_ms, a.rate, p)
    dose = probe_B_C(c, a.reps, a.t_ms, p)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "dn_responses.csv", index=False)
    res = {"params": p.as_dict(), "t_ms": a.t_ms, "reps": a.reps, "stim_rate_Hz": a.rate,
           "A_population_to_DN": summary, "A_lateralisation": lat,
           "B_C_LPLC2_left_dose_response_DNp01": dose}
    (OUT / "pathway_probe.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
