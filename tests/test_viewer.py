"""The Godot viewer loads, validates and draws every frame of a replay (headless).

Skipped when Godot is not installed. Set GODOT to the console executable.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

from agents.astar_agent import AStarGreedy
from game.environment import GameConfig
from game.runner import run_episode

GODOT = os.environ.get(
    "GODOT", r"C:\Users\User\tools\godot-4.7.2\Godot_v4.7.2-stable_win64_console.exe")
VIEWER = Path(__file__).resolve().parents[1] / "viewer"


@pytest.mark.skipif(not Path(GODOT).exists(), reason="Godot not installed")
def test_viewer_loads_and_validates_replay(tmp_path):
    p = tmp_path / "ep.jsonl"
    res = run_episode(AStarGreedy(), GameConfig(10, 10), seed=3, replay_path=p)
    out = subprocess.run([GODOT, "--headless", "--path", str(VIEWER), "--",
                          f"--replay={p}", "--check"],
                         capture_output=True, text=True, timeout=180)
    line = next(s for s in out.stdout.splitlines() if s.startswith("VIEWER_CHECK "))
    check = json.loads(line.split(" ", 1)[1])
    assert check["ok"], check["problems"]
    assert check["frames"] == res.steps + 1
    assert check["drawn_segments_last"] == res.final_length
    assert out.returncode == 0
