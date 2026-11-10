"""Authoritative, headless Snake simulation.

This module is the single source of truth for the game. Godot only *displays*
states produced here; agents only *read* them. Keeping the rules in one small,
deterministic, rendering-free module means:

  * evolution can run millions of moves without drawing anything,
  * every episode is exactly reproducible from (config, seed, actions),
  * A* and the fly brain are judged by identical rules.

Coordinates: (x, y) integer cells, 0 <= x < width, 0 <= y < height, y grows
"down" (north = (0, -1)). The body is stored head-first.

Move rules (standard Snake):
  1. The snake moves one cell per tick in its current heading.
  2. Actions are *relative*: LEFT / STRAIGHT / RIGHT. Reversing is impossible
     by construction (no action maps to the opposite heading).
  3. If the new head lands on food the snake grows (the tail stays);
     otherwise the tail cell is vacated in the same tick, so moving into the
     cell the tail is leaving is legal.
  4. Death: leaving the board (wall), entering an obstacle, or entering a
     body cell that is still occupied after the tail moves (self).
  5. Optional starvation cap: an episode ends after ``max_steps_without_food``
     ticks without eating (prevents agents from circling forever).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

Cell = tuple[int, int]

NORTH, EAST, SOUTH, WEST = (0, -1), (1, 0), (0, 1), (-1, 0)
HEADINGS = [NORTH, EAST, SOUTH, WEST]  # clockwise order


class Action(IntEnum):
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2


class Outcome(IntEnum):
    RUNNING = 0
    WALL = 1
    SELF = 2
    OBSTACLE = 3
    STARVED = 4
    WON = 5          # board completely filled
    STEP_LIMIT = 6   # hard episode cap reached while alive


def turn(heading: Cell, action: Action) -> Cell:
    i = HEADINGS.index(heading)
    return HEADINGS[(i + (int(action) - 1)) % 4]  # LEFT=-1, STRAIGHT=0, RIGHT=+1


def action_to(heading: Cell, new_heading: Cell) -> Action | None:
    """Relative action that turns ``heading`` into ``new_heading`` (None if reverse)."""
    for a in Action:
        if turn(heading, a) == new_heading:
            return a
    return None


@dataclass
class GameConfig:
    width: int = 20
    height: int = 20
    start_length: int = 3
    obstacles: tuple[Cell, ...] = ()
    max_steps_without_food: int | None = None   # default: width * height
    max_steps: int | None = None                # hard cap on episode length
    random_start: bool = False                  # False: centre, heading east

    def starvation_cap(self) -> int:
        return self.max_steps_without_food or self.width * self.height


@dataclass
class SnakeGame:
    cfg: GameConfig = field(default_factory=GameConfig)
    seed: int = 0

    def __post_init__(self):
        self.reset(self.seed)

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        c = self.cfg
        self.obstacles = frozenset(c.obstacles)
        if c.random_start:
            self.heading = HEADINGS[int(self.rng.integers(4))]
            hx = int(self.rng.integers(c.start_length, c.width - c.start_length))
            hy = int(self.rng.integers(c.start_length, c.height - c.start_length))
        else:
            self.heading = EAST
            hx, hy = c.width // 2, c.height // 2
        dx, dy = self.heading
        self.body: deque[Cell] = deque((hx - k * dx, hy - k * dy) for k in range(c.start_length))
        self.occupied: set[Cell] = set(self.body)
        if self.occupied & self.obstacles:
            raise ValueError("start position overlaps an obstacle")
        self.steps = 0
        self.steps_since_food = 0
        self.food_eaten = 0
        self.outcome = Outcome.RUNNING
        self.food: Cell | None = None
        self._spawn_food()
        return self

    def _spawn_food(self):
        free = [(x, y) for y in range(self.cfg.height) for x in range(self.cfg.width)
                if (x, y) not in self.occupied and (x, y) not in self.obstacles]
        if not free:
            self.food = None
            self.outcome = Outcome.WON
            return
        self.food = free[int(self.rng.integers(len(free)))]

    # ------------------------------------------------------------- queries
    @property
    def head(self) -> Cell:
        return self.body[0]

    @property
    def tail(self) -> Cell:
        return self.body[-1]

    @property
    def length(self) -> int:
        return len(self.body)

    @property
    def done(self) -> bool:
        return self.outcome != Outcome.RUNNING

    def in_bounds(self, c: Cell) -> bool:
        return 0 <= c[0] < self.cfg.width and 0 <= c[1] < self.cfg.height

    def is_blocked(self, c: Cell, tail_moves: bool = True) -> bool:
        """Would entering ``c`` next tick kill the snake? (ignores food growth)"""
        if not self.in_bounds(c) or c in self.obstacles:
            return True
        if c in self.occupied:
            return not (tail_moves and c == self.tail and c != self.food)
        return False

    def next_head(self, action: Action) -> Cell:
        dx, dy = turn(self.heading, action)
        return (self.head[0] + dx, self.head[1] + dy)

    # ---------------------------------------------------------------- step
    def step(self, action: Action) -> Outcome:
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        action = Action(action)
        self.heading = turn(self.heading, action)
        new = (self.head[0] + self.heading[0], self.head[1] + self.heading[1])
        self.steps += 1

        if not self.in_bounds(new):
            self.outcome = Outcome.WALL
            return self.outcome
        if new in self.obstacles:
            self.outcome = Outcome.OBSTACLE
            return self.outcome
        eats = new == self.food
        if not eats:  # tail leaves first, so the head may enter the old tail cell
            self.occupied.discard(self.body.pop())
        if new in self.occupied:
            self.outcome = Outcome.SELF
            return self.outcome
        self.body.appendleft(new)
        self.occupied.add(new)

        if eats:
            self.food_eaten += 1
            self.steps_since_food = 0
            self._spawn_food()
        else:
            self.steps_since_food += 1
            if self.steps_since_food >= self.cfg.starvation_cap():
                self.outcome = Outcome.STARVED
        if self.cfg.max_steps and self.steps >= self.cfg.max_steps and not self.done:
            self.outcome = Outcome.STEP_LIMIT
        return self.outcome

    # ----------------------------------------------------------- serialise
    def snapshot(self) -> dict:
        """Plain-JSON state, used for replays and for the Godot viewer."""
        return {"step": self.steps, "body": [list(c) for c in self.body],
                "heading": list(self.heading), "food": list(self.food) if self.food else None,
                "food_eaten": self.food_eaten, "outcome": self.outcome.name}


# ---------------------------------------------------------------- analysis
def flood_fill(game: SnakeGame, start: Cell, extra_blocked: frozenset = frozenset()) -> int:
    """Number of cells reachable from ``start`` through currently free cells.

    The tail cell counts as free (it will move). Used for *measurement*
    (self-trapping analysis) and by the safety-checked A* variants. It is never
    given to the fly brain.
    """
    seen = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in HEADINGS:
            n = (x + dx, y + dy)
            if n in seen or n in extra_blocked or game.is_blocked(n):
                continue
            seen.add(n)
            q.append(n)
    return len(seen)
