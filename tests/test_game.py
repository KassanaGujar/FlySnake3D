from collections import deque

import pytest

from game.environment import EAST, NORTH, SOUTH, WEST, Action, GameConfig, Outcome, SnakeGame, turn
from game.runner import run_episode, verify_replay


def place(g: SnakeGame, body, heading, food):
    g.body = deque(body)
    g.occupied = set(body)
    g.heading = heading
    g.food = food
    return g


def test_turns_are_relative_and_never_reverse():
    assert turn(NORTH, Action.LEFT) == WEST
    assert turn(NORTH, Action.RIGHT) == EAST
    assert turn(WEST, Action.STRAIGHT) == WEST
    for h in (NORTH, EAST, SOUTH, WEST):
        for a in Action:
            assert turn(h, a) != (-h[0], -h[1])


def test_start_state():
    g = SnakeGame(GameConfig(width=20, height=20, start_length=3), seed=1)
    assert list(g.body) == [(10, 10), (9, 10), (8, 10)]
    assert g.food not in g.occupied


def test_move_and_tail_follows():
    g = place(SnakeGame(GameConfig(10, 10)), [(5, 5), (4, 5), (3, 5)], EAST, (0, 0))
    g.step(Action.STRAIGHT)
    assert list(g.body) == [(6, 5), (5, 5), (4, 5)] and g.length == 3


def test_growth_on_food_and_food_respawns_on_free_cell():
    g = place(SnakeGame(GameConfig(10, 10)), [(5, 5), (4, 5), (3, 5)], EAST, (6, 5))
    g.step(Action.STRAIGHT)
    assert g.length == 4 and g.food_eaten == 1
    assert g.food is not None and g.food not in g.occupied


def test_wall_death():
    g = place(SnakeGame(GameConfig(10, 10)), [(9, 5), (8, 5), (7, 5)], EAST, (0, 0))
    assert g.step(Action.STRAIGHT) == Outcome.WALL


def test_self_collision():
    # head at (5,5) heading north; turning left -> west into (4,5), a body cell
    body = [(5, 5), (5, 6), (4, 6), (4, 5), (4, 4)]
    g = place(SnakeGame(GameConfig(10, 10)), body, NORTH, (0, 0))
    assert g.step(Action.LEFT) == Outcome.SELF


def test_moving_into_vacating_tail_is_legal():
    # 2x2 loop: head (5,5) north-bound, tail at (4,5) leaves this tick
    body = [(5, 5), (5, 6), (4, 6), (4, 5)]
    g = place(SnakeGame(GameConfig(10, 10)), body, NORTH, (0, 0))
    assert g.step(Action.LEFT) == Outcome.RUNNING
    assert g.head == (4, 5)


def test_obstacles_kill():
    g = SnakeGame(GameConfig(10, 10, obstacles=((6, 5),)))
    assert g.step(Action.STRAIGHT) == Outcome.OBSTACLE


def test_starvation_cap():
    g = SnakeGame(GameConfig(10, 10, max_steps_without_food=4))
    g.food = (0, 0)
    outs = [g.step(a) for a in (Action.LEFT, Action.LEFT, Action.LEFT, Action.LEFT)]
    assert outs[-1] == Outcome.STARVED


def test_determinism():
    a = SnakeGame(GameConfig(), seed=42)
    b = SnakeGame(GameConfig(), seed=42)
    for act in [1, 1, 0, 1, 2, 2, 1, 0] * 3:
        if a.done:
            break
        a.step(act); b.step(act)
        assert a.snapshot() == b.snapshot()


def test_replay_roundtrip(tmp_path):
    from agents.astar_agent import AStarGreedy
    p = tmp_path / "ep.jsonl"
    res = run_episode(AStarGreedy(), GameConfig(10, 10), seed=3, replay_path=p)
    assert p.exists() and res.steps > 0
    assert verify_replay(p)
