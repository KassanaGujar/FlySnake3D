"""Brain view: orientation of the exported positions, and headless playback in Godot."""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BRAIN = ROOT / "results" / "brain"
GODOT = os.environ.get(
    "GODOT", r"C:\Users\User\tools\godot-4.7.2\Godot_v4.7.2-stable_win64_console.exe")
needs_recording = pytest.mark.skipif(
    not (BRAIN / "looming_demo" / "activity.json").exists(),
    reason="run python -m experiments.record_brain_activity first")


@needs_recording
def test_exported_positions_are_anatomically_oriented():
    """Viewer axes: +x = fly's right, +y = dorsal, -z = anterior (right-handed, not mirrored).

    Checked against landmarks, not assumed: left-annotated groups have lower x
    than their right partners, and the looming detectors' cell groups sit in
    the optic lobes, far from the midline.
    """
    meta = json.loads((BRAIN / "neurons.json").read_text())
    pos = {(l["type"], l["side"]): np.array(l["pos"]) for l in meta["labels"]}
    for t in ("LPLC2", "LC4", "LC9"):
        assert pos[(t, "left")][0] < -50 < 50 < pos[(t, "right")][0], t
    xyz = np.fromfile(BRAIN / "neurons_pos.f32", np.float32).reshape(-1, 3)
    assert len(xyz) == meta["n"]
    assert 600 < np.ptp(xyz[:, 0]) < 1000      # brain width in micrometres, with optic lobes
    assert 150 < np.ptp(xyz[:, 2]) < 400       # front-to-back depth


@needs_recording
@pytest.mark.skipif(not Path(GODOT).exists(), reason="Godot not installed")
def test_brain_view_plays_every_spike():
    act = json.loads((BRAIN / "looming_demo" / "activity.json").read_text())
    out = subprocess.run([GODOT, "--headless", "--path", str(ROOT / "viewer"), "--",
                          "--brain", "--check"], capture_output=True, text=True, timeout=300)
    line = next(s for s in out.stdout.splitlines() if s.startswith("BRAIN_CHECK "))
    check = json.loads(line.split(" ", 1)[1])
    assert check["ok"]
    assert check["spiking_neuron_bins"] == act["total_spiking_neuron_bins"]
    assert check["bins"] == act["n_bins"]
