"""A* baselines for Snake: a ladder from naive to safety-checked.

Comparing the fly brain against a single weak baseline proves little, so we
define four planners. Each adds exactly one idea to the previous one:

  A0  AStarGreedy      Textbook greedy A*. Each tick: A* from head to food,
                       treating the *current* body as static walls; take the
                       first step. This is the PRIMARY baseline. It has the
                       documented weakness: the path is valid as a snapshot,
                       but greedily entering pockets lets the body seal the
                       snake in a few moves later.
  A1  AStarTimeAware   Same, but body occupancy is dynamic: the segment i
                       cells from the tail is free after i+1 moves, so paths
                       may run through space the tail is about to vacate.
  A2  AStarTailCheck   A1 + "virtual snake" safety check (the idea used by
                       chuyangliu/snake's graph-search solver): simulate
                       eating along the path; only commit if the virtual head
                       can still reach its own tail afterwards. Otherwise
                       chase the tail by the longest available route, and as a
                       last resort move to the neighbour farthest from food.
  A3  AStarSpaceCheck  A2 + reachable-space veto on every move: never step
                       into a region (flood fill) smaller than the body.

None of these is optimal play (that needs Hamiltonian-cycle strategies,
deliberately excluded because they never take risks and are not "pathfinding").
They are written for this project from the descriptions above, not copied.

Interface: ``agent.act(game) -> Action``. After each call ``agent.last_path``
(list of cells, may be empty) and ``agent.last_reason`` (str) describe the
decision, for logging and the Godot overlay.
"""
from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass

from game.environment import HEADINGS, Action, Cell, SnakeGame, action_to

# ---------------------------------------------------------------------------
# A lightweight immutable board for "what if" simulation inside the planners.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Board:
    body: tuple[Cell, ...]          # head first
    food: Cell | None
    obstacles: frozenset
    w: int
    h: int

    @staticmethod
    def of(g: SnakeGame) -> "Board":
        return Board(tuple(g.body), g.food, g.obstacles, g.cfg.width, g.cfg.height)

    @property
    def head(self) -> Cell:
        return self.body[0]

    def inside(self, c: Cell) -> bool:
        return 0 <= c[0] < self.w and 0 <= c[1] < self.h and c not in self.obstacles

    def neighbours(self, c: Cell):
        for dx, dy in HEADINGS:
            n = (c[0] + dx, c[1] + dy)
            if self.inside(n):
                yield n

    def legal_moves(self) -> list[Cell]:
        """Cells the head can enter next tick without dying."""
        body_set = set(self.body)
        out = []
        for n in self.neighbours(self.head):
            eats = n == self.food
            if n in body_set and not (n == self.body[-1] and not eats):
                continue
            if len(self.body) > 1 and n == self.body[1]:
                continue  # reverse
            out.append(n)
        return out

    def move(self, c: Cell) -> "Board":
        eats = c == self.food
        body = (c,) + (self.body if eats else self.body[:-1])
        return Board(body, None if eats else self.food, self.obstacles, self.w, self.h)

    def free_after(self) -> dict[Cell, int]:
        """cell -> number of moves after which it is vacated (no growth assumed)."""
        L = len(self.body)
        return {c: L - i for i, c in enumerate(self.body)}

    def flood(self, start: Cell) -> int:
        body_set = set(self.body[:-1])  # tail cell will move
        seen, q = {start}, deque([start])
        while q:
            for n in self.neighbours(q.popleft()):
                if n not in seen and n not in body_set:
                    seen.add(n)
                    q.append(n)
        return len(seen)


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(board: Board, start: Cell, goal: Cell, time_aware: bool,
          goal_is_tail: bool = False) -> list[Cell]:
    """A* on the 4-connected grid. Returns the path excluding ``start`` ([] if none).

    Static mode: every body cell except ``start`` is a wall (the tail only if
    ``goal_is_tail`` and it is the goal). Time-aware mode: body cell i is
    passable when reached at step t >= free_after[i].
    Ties are broken deterministically (insertion order) so runs are reproducible.
    """
    free_after = board.free_after()
    counter = 0
    open_ = [(manhattan(start, goal), 0, counter, start)]
    came: dict[Cell, Cell] = {}
    g = {start: 0}
    while open_:
        _, gc, _, cur = heapq.heappop(open_)
        if cur == goal:
            path = [cur]
            while path[-1] in came:
                path.append(came[path[-1]])
            path.pop()  # drop start
            return path[::-1]
        if gc > g.get(cur, 1 << 30):
            continue
        for n in board.neighbours(cur):
            t = gc + 1
            if n in free_after and n != start:
                if not (n == goal and goal_is_tail):
                    if not time_aware or t < free_after[n]:
                        continue
            if t < g.get(n, 1 << 30):
                g[n] = t
                came[n] = cur
                counter += 1
                heapq.heappush(open_, (t + manhattan(n, goal), t, counter, n))
    return []


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class _Base:
    name = "base"

    def __init__(self):
        self.last_path: list[Cell] = []
        self.last_reason = ""

    def _to_action(self, game: SnakeGame, cell: Cell) -> Action:
        h = game.head
        a = action_to(game.heading, (cell[0] - h[0], cell[1] - h[1]))
        assert a is not None, "planner proposed a reverse move"
        return a

    def _no_plan(self, game: SnakeGame, board: Board) -> Action:
        """Fallback when no food path exists: first non-fatal move (straight, left, right)."""
        legal = set(board.legal_moves())
        for a in (Action.STRAIGHT, Action.LEFT, Action.RIGHT):
            if game.next_head(a) in legal:
                self.last_reason = "no path: first safe move"
                return a
        self.last_reason = "no safe move"
        return Action.STRAIGHT


class AStarGreedy(_Base):
    """A0 - the primary baseline: greedy A* to food on a static body snapshot."""
    name = "astar_greedy"
    time_aware = False

    def act(self, game: SnakeGame) -> Action:
        b = Board.of(game)
        self.last_path = astar(b, b.head, b.food, self.time_aware) if b.food else []
        if self.last_path:
            self.last_reason = "path to food"
            return self._to_action(game, self.last_path[0])
        return self._no_plan(game, b)


class AStarTimeAware(AStarGreedy):
    """A1 - greedy A* with dynamic (time-aware) body occupancy."""
    name = "astar_time_aware"
    time_aware = True


class AStarTailCheck(_Base):
    """A2 - A1 plus virtual-snake tail-reachability check and tail chasing."""
    name = "astar_tail_check"
    space_check = False

    def _safe_after(self, b: Board) -> bool:
        """After arriving, can the head still reach its own tail?"""
        if len(b.body) < 3:
            return True
        return bool(astar(b, b.head, b.body[-1], time_aware=True, goal_is_tail=True))

    def _space_ok(self, b: Board) -> bool:
        return not self.space_check or b.flood(b.head) >= len(b.body)

    def act(self, game: SnakeGame) -> Action:
        b = Board.of(game)
        legal = [m for m in b.legal_moves() if self._space_ok(b.move(m))]
        if not legal:
            legal = b.legal_moves()  # every option is bad; still avoid instant death
        if not legal:
            self.last_path, self.last_reason = [], "no safe move"
            return Action.STRAIGHT

        # 1) shortest path to food, accepted only if the virtual snake stays safe
        if b.food:
            path = astar(b, b.head, b.food, time_aware=True)
            if path and path[0] in legal:
                v = b
                for c in path:
                    v = v.move(c)
                if self._safe_after(v):
                    self.last_path, self.last_reason = path, "safe path to food"
                    return self._to_action(game, path[0])

        # 2) chase the tail along the longest available route
        best, best_len = None, -1
        for m in legal:
            nb = b.move(m)
            p = (astar(nb, nb.head, nb.body[-1], time_aware=True, goal_is_tail=True)
                 if len(nb.body) > 2 else [nb.body[-1]])
            if p and len(p) > best_len:
                best, best_len, best_path = m, len(p), [m] + p
        if best is not None:
            self.last_path, self.last_reason = best_path, "chase tail (food unsafe)"
            return self._to_action(game, best)

        # 3) move to the neighbour with most space, ties -> farthest from food
        def key(m):
            return (b.move(m).flood(m), manhattan(m, b.food) if b.food else 0)
        m = max(legal, key=key)
        self.last_path, self.last_reason = [m], "maximise space"
        return self._to_action(game, m)


class AStarSpaceCheck(AStarTailCheck):
    """A3 - A2 plus a reachable-space veto on every move."""
    name = "astar_space_check"
    space_check = True


AGENTS = {a.name: a for a in (AStarGreedy, AStarTimeAware, AStarTailCheck, AStarSpaceCheck)}
