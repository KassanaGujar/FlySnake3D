"""Run episodes, measure every decision, and write replays.

Per-decision measurements (research brief section 9). They are computed by the
*experiment*, never shown to the agent:

  length            snake length before the move
  nearest_body      Manhattan distance from head to the nearest non-neck segment
  free_neighbours   number of non-fatal moves available
  reachable         cells reachable from the head (flood fill, tail counts as free)
  trapped           reachable < length: the body can no longer fit in the
                    space the head can reach. This is a conservative proxy;
                    the tail moving can still open an escape, so it is
                    reported together with whether death actually followed.
  food_reachable    a time-aware path from head to food exists

Episode-level self-trapping:
  first_trap_step   first decision with trapped == True (None if never)
  self_trap_death   died (self/wall) after having become trapped
  moves_trap_to_death
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from game.environment import HEADINGS, GameConfig, Outcome, SnakeGame, flood_fill

DEATHS = (Outcome.WALL, Outcome.SELF, Outcome.OBSTACLE)


@dataclass
class EpisodeResult:
    agent: str
    seed: int
    board: str
    outcome: str
    food_eaten: int
    final_length: int
    steps: int
    first_trap_step: int | None
    self_trap_death: bool
    moves_trap_to_death: int | None
    mean_decision_ms: float
    decisions: list = field(default_factory=list, repr=False)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("decisions")
        return d


def _measure(g: SnakeGame) -> dict:
    hx, hy = g.head
    body = list(g.body)
    nearest = min((abs(hx - x) + abs(hy - y) for x, y in body[2:]), default=None)
    free = sum(not g.is_blocked((hx + dx, hy + dy)) for dx, dy in HEADINGS
               if (hx + dx, hy + dy) != (body[1] if len(body) > 1 else None))
    reach = flood_fill(g, g.head)
    return {"length": g.length, "nearest_body": nearest, "free_neighbours": free,
            "reachable": reach, "trapped": reach < g.length}


def run_episode(agent, cfg: GameConfig, seed: int, replay_path: Path | None = None,
                measure: bool = True) -> EpisodeResult:
    g = SnakeGame(cfg, seed)
    frames = [g.snapshot()] if replay_path else None
    decisions = []
    first_trap = None
    t_total = 0.0
    while not g.done:
        m = _measure(g) if measure else {}
        t0 = time.perf_counter()
        a = agent.act(g)
        t_total += time.perf_counter() - t0
        if measure:
            m.update(step=g.steps, action=int(a), reason=getattr(agent, "last_reason", ""),
                     food_reachable=bool(getattr(agent, "last_path", None))
                     and getattr(agent, "last_reason", "").endswith("food"))
            if m["trapped"] and first_trap is None:
                first_trap = g.steps
            decisions.append(m)
        g.step(a)
        if frames is not None:
            fr = g.snapshot()
            fr["plan"] = [list(c) for c in getattr(agent, "last_path", [])]
            fr["reason"] = getattr(agent, "last_reason", "")
            frames.append(fr)
    died = g.outcome in DEATHS
    res = EpisodeResult(
        agent=getattr(agent, "name", type(agent).__name__), seed=seed,
        board=f"{cfg.width}x{cfg.height}", outcome=g.outcome.name, food_eaten=g.food_eaten,
        final_length=g.length, steps=g.steps, first_trap_step=first_trap,
        self_trap_death=died and first_trap is not None,
        moves_trap_to_death=(g.steps - first_trap) if died and first_trap is not None else None,
        mean_decision_ms=1000 * t_total / max(g.steps, 1), decisions=decisions)
    if replay_path:
        write_replay(replay_path, cfg, res, frames)
    return res


# ---------------------------------------------------------------- replays
def write_replay(path: Path, cfg: GameConfig, res: EpisodeResult, frames: list[dict]):
    """JSONL: line 1 = header (config + result), then one line per tick."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"type": "header", "config": asdict(cfg), "result": res.summary()}) + "\n")
        for fr in frames:
            f.write(json.dumps({"type": "frame", **fr}) + "\n")


def read_replay(path: Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    return lines[0], lines[1:]


def verify_replay(path: Path) -> bool:
    """Re-simulate from the header's config/seed and check every frame matches.

    Actions are recovered from consecutive headings, so this also proves the
    replay is a faithful record of the deterministic game.
    """
    from game.environment import action_to
    header, frames = read_replay(path)
    cfg_d = dict(header["config"])
    cfg_d["obstacles"] = tuple(tuple(o) for o in cfg_d["obstacles"])
    g = SnakeGame(GameConfig(**cfg_d), header["result"]["seed"])
    for prev, cur in zip(frames, frames[1:]):
        if g.snapshot()["body"] != prev["body"]:
            return False
        a = action_to(tuple(prev["heading"]), tuple(cur["heading"]))
        g.step(a)
    return g.snapshot()["body"] == frames[-1]["body"] and g.outcome.name == frames[-1]["outcome"]
