"""Play B games in parallel with one batched policy.

Two modes:

  budget    (training)   every column plays for exactly ``n_decisions`` moves.
                         When a snake dies or starves its game restarts with a
                         new seed and its brain is reset. Fitness uses total
                         food and number of deaths. This keeps the GPU busy
                         and averages several lives per candidate.
  episodes  (evaluation) every column plays ONE episode to its end, with the
                         same per-move measurements as the A* benchmark
                         (game/runner.py), so results are directly comparable.
"""
from __future__ import annotations

import numpy as np

from brain.visual_encoder import N_FEAT, view_features
from game.environment import GameConfig, Outcome, SnakeGame
from game.runner import _measure

DEATHS = (Outcome.WALL, Outcome.SELF, Outcome.OBSTACLE, Outcome.STARVED)
LIFE_SEED_STEP = 1_000_003
DEATH_PENALTY = 0.5


def fitness(food: np.ndarray, deaths: np.ndarray) -> np.ndarray:
    """Food eaten minus a penalty per death (crash or starvation)."""
    return food - DEATH_PENALTY * deaths


def play_budget(policy, cfg: GameConfig, seeds, n_decisions: int) -> dict:
    B = len(seeds)
    games = [SnakeGame(cfg, int(s)) for s in seeds]
    prev = [None] * B
    lives = np.zeros(B, np.int64)
    food = np.zeros(B)
    deaths = np.zeros(B)
    max_len = np.zeros(B, np.int64)
    acts_hist = np.zeros(3, np.int64)
    policy.reset(np.ones(B, bool))
    feats = np.zeros((B, 2, N_FEAT), np.float32)
    for _ in range(n_decisions):
        for b, g in enumerate(games):
            feats[b], prev[b] = view_features(g, prev[b])
        acts = policy.act(feats)
        acts_hist += np.bincount(acts, minlength=3)
        restart = np.zeros(B, bool)
        for b, g in enumerate(games):
            g.step(int(acts[b]))
            if g.done:
                food[b] += g.food_eaten
                max_len[b] = max(max_len[b], g.length)
                deaths[b] += g.outcome in DEATHS
                lives[b] += 1
                games[b] = SnakeGame(cfg, int(seeds[b]) + LIFE_SEED_STEP * int(lives[b]))
                prev[b] = None
                restart[b] = True
        if restart.any():
            policy.reset(restart)
    for b, g in enumerate(games):
        food[b] += g.food_eaten
        max_len[b] = max(max_len[b], g.length)
    return {"fitness": fitness(food, deaths), "food": food, "deaths": deaths,
            "max_length": max_len, "action_fractions": acts_hist / acts_hist.sum()}


def play_episodes(policy, cfg: GameConfig, seeds, max_decisions: int, name: str | None = None,
                  record_col: int | None = None) -> tuple[list[dict], list[dict] | None]:
    """One full episode per column. Returns (per-episode summaries, frames of record_col)."""
    B = len(seeds)
    games = [SnakeGame(cfg, int(s)) for s in seeds]
    prev = [None] * B
    first_trap = [None] * B
    alive = np.ones(B, bool)
    frames = [games[record_col].snapshot()] if record_col is not None else None
    policy.reset(np.ones(B, bool))
    feats = np.zeros((B, 2, N_FEAT), np.float32)
    for _ in range(max_decisions):
        feats[:] = 0
        for b, g in enumerate(games):
            if alive[b]:
                feats[b], prev[b] = view_features(g, prev[b])
                if first_trap[b] is None and _measure(g)["trapped"]:
                    first_trap[b] = g.steps
        acts = policy.act(feats)
        for b, g in enumerate(games):
            if alive[b]:
                g.step(int(acts[b]))
                if b == record_col:
                    frames.append(g.snapshot())
                if g.done:
                    alive[b] = False
        if not alive.any():
            break
    out = []
    for b, g in enumerate(games):
        died = g.outcome in (Outcome.WALL, Outcome.SELF, Outcome.OBSTACLE)
        ft = first_trap[b]
        out.append({
            "agent": name or getattr(policy, "name", "policy"), "seed": int(seeds[b]),
            "board": f"{cfg.width}x{cfg.height}",
            "outcome": g.outcome.name if g.done else "DECISION_CAP",
            "food_eaten": g.food_eaten, "final_length": g.length, "steps": g.steps,
            "first_trap_step": ft, "self_trap_death": bool(died and ft is not None),
            "moves_trap_to_death": (g.steps - ft) if died and ft is not None else None})
    return out, frames
