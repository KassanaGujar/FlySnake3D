"""Event simulator, encoder, decoder, rollout and ES. All on CPU with small synthetic
networks, so these run without the connectome or a GPU."""
from collections import deque

import numpy as np
import pytest
import scipy.sparse as sp

from agents.policies import ANN_GENOME, AnnPolicy, RandomPolicy, ReactivePolicy
from brain.event_simulator import EventLIF
from brain.neural_simulator import LIFParams, TorchLIF
from brain.visual_encoder import FEATURES, N_FEAT, view_features
from evolution.es import ES
from evolution.rollout import play_budget, play_episodes
from game.environment import EAST, NORTH, GameConfig, SnakeGame

F = {n: i for i, n in enumerate(FEATURES)}


def random_net(n=300, density=0.05, seed=0):
    rng = np.random.default_rng(seed)
    W = sp.random(n, n, density=density, random_state=rng, format="csr",
                  data_rvs=lambda k: rng.integers(-6, 12, k).astype(np.float32))
    W.setdiag(0)
    W.eliminate_zeros()
    return W


# ------------------------------------------------------------ event simulator
def test_event_simulator_matches_dense_simulator():
    W = random_net()
    stim = np.arange(40)
    a = TorchLIF(W, batch=3, device="cpu", seed=7)
    e = EventLIF(W, batch=3, device="cpu", seed=7)
    a.set_poisson_input(stim, 120.0)
    e.set_poisson_input(stim, 120.0)
    ca, ce = a.run(1500), e.run(1500)
    assert ca.sum() > 500                       # the network is really active
    # Same model, same random numbers; only float rounding order differs.
    assert np.abs(ca - ce).sum() <= 0.01 * ca.sum()
    assert np.corrcoef(ca.ravel(), ce.ravel())[0, 1] > 0.999


def test_event_simulator_silent_without_input_and_columns_independent():
    W = random_net()
    e = EventLIF(W, batch=2, device="cpu", seed=1)
    r = np.zeros((40, 2), np.float32)
    r[:, 1] = 150.0
    e.set_poisson_input(np.arange(40), r)
    c = e.run(600)
    assert c[:, 0].sum() == 0 and c[:, 1].sum() > 0


def test_reset_columns_only_resets_selected_brains():
    W = random_net()
    e = EventLIF(W, batch=2, device="cpu", seed=1)
    e.set_poisson_input(np.arange(40), 150.0)
    e.run(300)
    g1 = e.g[:, 1].clone()
    e.reset_columns(np.array([True, False]))
    assert float(e.u[:, 0].abs().sum()) == 0 and float(e.g[:, 0].abs().sum()) == 0
    assert bool((e.g[:, 1] == g1).all())
    e.clear_inputs()
    assert e.run(100)[:, 0].sum() == 0          # no pending spikes survive the reset


# ------------------------------------------------------------ visual features
def place(g, body, heading, food):
    g.body = deque(body); g.occupied = set(body); g.heading = heading; g.food = food
    return g


def test_features_are_egocentric_and_lateralised():
    # Head (5,5) heading north on 10x10. Wall is 6 cells ahead (y = -1).
    g = place(SnakeGame(GameConfig(10, 10)), [(5, 5), (5, 6), (5, 7)], NORTH, (8, 5))
    f, front = view_features(g)
    assert front == pytest.approx(1 / 6)
    assert f[0, F["obstacle_side"]] == pytest.approx(1 / 6)      # west wall, x = -1
    assert f[1, F["obstacle_side"]] == pytest.approx(1 / 5)      # east wall, x = 10
    assert f[0, F["food_side"]] == 0 and f[1, F["food_side"]] > 0   # food due east = right eye
    assert f[0, F["food_front"]] == 0
    # Same scene rotated: heading east, food straight ahead -> frontal, both eyes.
    g = place(SnakeGame(GameConfig(10, 10)), [(5, 5), (4, 5), (3, 5)], EAST, (8, 5))
    f, _ = view_features(g)
    assert f[0, F["food_front"]] == f[1, F["food_front"]] > 0
    assert f[0, F["food_side"]] == f[1, F["food_side"]] == 0


def test_looming_signals_approach_only():
    g = place(SnakeGame(GameConfig(10, 10)), [(5, 3), (5, 4), (5, 5)], NORTH, (0, 9))
    _, prev = view_features(g)
    g.step(1)
    f, _ = view_features(g, prev)
    assert f[0, F["looming_front"]] > 0                          # wall got closer
    f2, _ = view_features(g, 1.0)
    assert f2[0, F["looming_front"]] == 0                        # receding: no looming


# ------------------------------------------------------------ rollout / policies
def test_budget_rollout_is_deterministic_and_counts_deaths():
    cfg = GameConfig(10, 10, max_steps_without_food=30)
    r1 = play_budget(RandomPolicy(8, seed=3), cfg, np.full(8, 5), 120)
    r2 = play_budget(RandomPolicy(8, seed=3), cfg, np.full(8, 5), 120)
    assert (r1["fitness"] == r2["fitness"]).all()
    assert r1["deaths"].sum() > 0


def test_reactive_control_beats_random():
    cfg = GameConfig(12, 12, max_steps_without_food=60)
    seeds = np.arange(16)
    rnd = play_budget(RandomPolicy(16, seed=0), cfg, seeds, 200)["fitness"].mean()
    rea = play_budget(ReactivePolicy(16), cfg, seeds, 200)["fitness"].mean()
    assert rea > rnd + 3


def test_episode_rollout_reports_runner_style_results():
    res, frames = play_episodes(ReactivePolicy(4), GameConfig(10, 10), np.arange(4), 500,
                                record_col=0)
    assert len(res) == 4 and frames[0]["step"] == 0
    assert all(r["steps"] > 0 and r["food_eaten"] + 2 <= r["final_length"] <= r["food_eaten"] + 3 for r in res)
    assert len(frames) == res[0]["steps"] + 1


# ------------------------------------------------------------ evolution strategy
def test_es_improves_and_never_loses_champion():
    target = np.linspace(0.1, 0.9, 12)
    es = ES(np.full(12, 0.5), sigma0=0.2, pop=32, seed=0)
    champ_scores = []
    for _ in range(40):
        x = es.ask()
        assert (x[0] == es.champion).all()                       # champion is always re-entered
        fit = -((x - target) ** 2).sum(1)
        info = es.tell(fit)
        champ_scores.append(-((es.champion - target) ** 2).sum())
    assert champ_scores[-1] > champ_scores[0]
    assert all(b >= a - 1e-12 for a, b in zip(champ_scores, champ_scores[1:]))   # noise-free: monotone
    st = es.state()
    es2 = ES(np.full(12, 0.5), pop=32, seed=99)
    es2.load(st)
    assert (es2.ask() == es.ask()).all()                         # checkpoint restores the RNG too


def test_ann_policy_evolves_on_small_board():
    cfg = GameConfig(10, 10, max_steps_without_food=40)
    es = ES(np.full(ANN_GENOME, 0.5), pop=32, seed=1)
    first = last = None
    for gen in range(12):
        res = play_budget(AnnPolicy(es.ask()), cfg, np.full(32, 100 + gen), 120)
        info = es.tell(res["fitness"])
        first = info["mean_fitness"] if first is None else first
        last = info["mean_fitness"]
    assert last > first
