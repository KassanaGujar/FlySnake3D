"""Record whole-brain spiking for the Godot brain view.

Runs the unmodified FlyWire v783 / Shiu LIF model (Milestone 0) with a scripted
visual stimulus, and writes every spike of every neuron, binned, in a compact
binary format that viewer/brain.gd plays back.

The brain is NOT yet connected to the Snake game (that needs the visual encoder
and motor decoder, stages 3-4). Until then this records the Milestone 0 probe
conditions so the view shows real model activity, not an animation:

  0-50 ms     no input. The model has no spontaneous activity, so it is silent.
  50-250 ms   LPLC2 + LC4, left eye, 150 Hz  (published looming detectors)
  250-350 ms  no input (activity dies away)
  350-550 ms  LC9, right eye, 150 Hz         (visual projection type that drove
                                              left DNa01 in the Milestone 0 probe)
  550-700 ms  no input

Neuron positions are the FlyWire anchor points (pos_x/y/z, one point per neuron
on its backbone, 4x4x40 nm voxels, per the flywire_annotations README). They
show where each neuron is, not its full shape.

Run:  python -m experiments.record_brain_activity [--out results/brain/looming_demo]
Writes  results/brain/neurons_pos.f32    N x 3 float32, micrometres, centred,
                                         axes: x = left-right, y = up, z = front-back
        results/brain/neurons_class.u8   N uint8 super-class id (255 = no position)
        results/brain/neurons.json       class names, labelled cell groups
        <out>/activity.json              stimuli, bins, per-bin summaries
        <out>/spikes_idx.i32, spikes_off.i32  neurons that spiked in each bin
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from brain.connectome_loader import load_connectome
from brain.neural_simulator import LIFParams, TorchLIF

ROOT = Path(__file__).resolve().parents[1]
BRAIN = ROOT / "results" / "brain"
VOXEL_NM = np.array([4.0, 4.0, 40.0])

# Stimulus script: (label, cell types, side, rate Hz, start ms, end ms)
STIMULI = [
    ("Looming detectors, left eye (LPLC2 + LC4)", ["LPLC2", "LC4"], "left", 150.0, 50, 250),
    ("Object detector LC9, right eye", ["LC9"], "right", 150.0, 350, 550),
]
DURATION_MS = 700
BIN_MS = 2.0

# Groups labelled in the viewer and reported per bin. All are published
# FlyWire cell types already used in Milestone 0.
LABELS = [
    ("LPLC2 (looming)", "LPLC2"), ("LC4 (looming)", "LC4"), ("LC9", "LC9"),
    ("DNp01 giant fiber (escape)", "DNp01"), ("DNa01 (steering)", "DNa01"),
    ("DNa02 (steering)", "DNa02"),
]


def write_neurons(c) -> np.ndarray:
    ann = pd.read_csv(ROOT / "data" / "neuron_annotations.tsv", sep="\t",
                      usecols=["root_id", "pos_x", "pos_y", "pos_z", "super_class"],
                      dtype={"root_id": np.int64}, low_memory=False)
    ann = ann.drop_duplicates("root_id").set_index("root_id").reindex(c.root_ids)
    vox = ann[["pos_x", "pos_y", "pos_z"]].to_numpy(np.float64)
    has = ~np.isnan(vox).any(axis=1)
    um = vox * VOXEL_NM / 1000.0
    centre = np.nanmean(um[has], axis=0)
    um = um - centre
    # FlyWire y grows ventrally; flip so +y is dorsal ("up") in Godot.
    pos = np.zeros((len(um), 3), np.float32)
    pos[has] = np.stack([um[has, 0], -um[has, 1], um[has, 2]], axis=1)
    classes = sorted(ann.super_class.dropna().unique())
    cls = np.full(len(um), 255, np.uint8)
    for i, name in enumerate(classes):
        cls[(ann.super_class == name).to_numpy() & has] = i

    labels = []
    for text, ctype in LABELS:
        for side in ("left", "right"):
            idx = c.indices_of_type(ctype, side=side)
            idx = idx[has[idx]]
            if len(idx):
                labels.append({"text": f"{text} {side[0].upper()}", "type": ctype, "side": side,
                               "pos": pos[idx].mean(axis=0).round(2).tolist(), "n": int(len(idx)),
                               "indices": idx.tolist()})
    BRAIN.mkdir(parents=True, exist_ok=True)
    pos.tofile(BRAIN / "neurons_pos.f32")
    cls.tofile(BRAIN / "neurons_class.u8")
    (BRAIN / "neurons.json").write_text(json.dumps({
        "n": int(len(pos)), "with_position": int(has.sum()), "classes": classes,
        "units": "micrometres", "source": "flywire_annotations pos_x/y/z (4x4x40 nm voxels)",
        "axes": "x left-right, y dorsal(up), z anterior-posterior; centred on the mean",
        "labels": labels}, indent=1))
    print(f"neurons: {len(pos)} ({has.sum()} with positions)")
    return has


def record(c, out: Path, seed: int, device: str):
    p = LIFParams()
    sim = TorchLIF(c.W, p, batch=1, device=device, seed=seed)
    steps_per_bin = int(round(BIN_MS / p.dt))
    n_bins = int(DURATION_MS / BIN_MS)
    stim_idx = [np.concatenate([c.indices_of_type(t, side=s) for t in types])
                for _, types, s, *_ in STIMULI]
    types = c.annotations.cell_type.fillna(c.annotations.hemibrain_type).fillna("?").to_numpy()

    idx_chunks, offsets, bins = [], [0], []
    torch = sim.torch
    t0 = time.time()
    for b in range(n_bins):
        t_ms = b * BIN_MS
        sim.clear_inputs()
        active = []
        for k, (label, _, side, rate, start, end) in enumerate(STIMULI):
            if start <= t_ms < end:
                sim.set_poisson_input(stim_idx[k], rate)
                active.append(k)
        counts = torch.zeros(sim.N, device=sim.device, dtype=torch.int32)
        for _ in range(steps_per_bin):
            counts += sim.step()[:, 0]
        fired = torch.nonzero(counts).flatten().cpu().numpy().astype(np.int32)
        idx_chunks.append(fired)
        offsets.append(offsets[-1] + len(fired))
        top = pd.Series(types[fired]).value_counts().head(6) if len(fired) else pd.Series(dtype=int)
        bins.append({"t_ms": t_ms, "n_spiking": int(len(fired)), "stimuli": active,
                     "top_types": [[k, int(v)] for k, v in top.items()]})
    print(f"simulated {DURATION_MS} ms in {time.time() - t0:.1f} s")

    out.mkdir(parents=True, exist_ok=True)
    np.concatenate(idx_chunks).astype(np.int32).tofile(out / "spikes_idx.i32")
    np.asarray(offsets, np.int32).tofile(out / "spikes_off.i32")
    (out / "activity.json").write_text(json.dumps({
        "title": "Milestone 0 probe: looming left, then object right (brain not yet playing Snake)",
        "model": "Shiu et al. 2024 LIF on FlyWire v783, PyTorch backend", "params": p.as_dict(),
        "seed": seed, "bin_ms": BIN_MS, "n_bins": n_bins, "duration_ms": DURATION_MS,
        "stimuli": [{"label": l, "types": t, "side": s, "rate_hz": r, "start_ms": a, "end_ms": e}
                    for l, t, s, r, a, e in STIMULI],
        "total_spiking_neuron_bins": int(offsets[-1]), "bins": bins}, indent=0))
    print(f"wrote {out} ({offsets[-1]} neuron-bins with spikes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BRAIN / "looming_demo"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    c = load_connectome()
    write_neurons(c)
    record(c, Path(a.out), a.seed, a.device)


if __name__ == "__main__":
    main()
