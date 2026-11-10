from collections import deque

import pytest

from agents.astar_agent import AGENTS, AStarGreedy, AStarTimeAware, Board, astar
from game.environment import EAST, NORTH, GameConfig, Outcome, SnakeGame
from game.runner import run_episode


def place(g, body, heading, food):
    g.body = deque(body); g.occupied = set(body); g.heading = heading; g.food = food
    return g


def test_astar_finds_shortest_path_on_empty_board():
    b = Board(((0, 0),), None, frozenset(), 10, 10)
    p = astar(b, (0, 0), (3, 4), time_aware=False)
    assert len(p) == 7 and p[-1] == (3, 4)


def test_static_astar_treats_body_as_wall_time_aware_does_not():
    # Snake curls around; the food lies "behind" the tail.
    # Head (2,2) facing north, body runs down then right: tail at (4,3).
    body = [(2, 2), (2, 3), (3, 3), (4, 3)]
    b = Board(tuple(body), (5, 3), frozenset(), 6, 6)
    static = astar(b, b.head, (5, 3), time_aware=False)
    aware = astar(b, b.head, (5, 3), time_aware=True)
    assert static and aware
    assert len(aware) <= len(static)


def test_agents_never_reverse_and_play_legal_games():
    for name, cls in AGENTS.items():
        res = run_episode(cls(), GameConfig(10, 10), seed=0)
        assert res.steps > 0, name


def test_greedy_eats_food_in_open_board():
    res = run_episode(AStarGreedy(), GameConfig(20, 20), seed=1)
    assert res.food_eaten >= 5


@pytest.mark.parametrize("name", list(AGENTS))
def test_agent_reproducible(name):
    a = run_episode(AGENTS[name](), GameConfig(12, 12), seed=5)
    b = run_episode(AGENTS[name](), GameConfig(12, 12), seed=5)
    assert a.summary() == {**b.summary(), "mean_decision_ms": a.mean_decision_ms}


def test_safety_ladder_improves_mean_score():
    """The safety-checked planners should beat greedy on average (sanity, not research result)."""
    def mean(cls):
        return sum(run_episode(cls(), GameConfig(10, 10), seed=s, measure=False).food_eaten
                   for s in range(8)) / 8
    assert mean(AGENTS["astar_tail_check"]) > mean(AStarGreedy)
