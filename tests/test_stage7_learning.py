"""Stage 7: dopamine plasticity, adaptation, smell, learning genomes.

Simulator tests use a small random network so they run in seconds.
"""
from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

from brain.event_simulator import EventLIF  # noqa: E402
from brain.neural_simulator import LIFParams  # noqa: E402

N = 300
PRE, POST, MOD = np.arange(0, 60), np.arange(60, 70), np.arange(70, 80)


def net(seed=0):
    rng = np.random.default_rng(seed)
    W = sp.random(N, N, density=0.05, random_state=rng, data_rvs=lambda k: rng.integers(1, 6, k)).tocsr()
    W = W.astype(np.float32)
    # make sure the plastic pathway exists: every PRE neuron synapses onto every POST neuron
    W = W.tolil()
    for j in POST:
        W[j, PRE] = 3.0
    return W.tocsr()


def run(W, plastic=False, rate=0.0, dopamine=0.0, adapt=0.0, seed=1, steps=300):
    s = EventLIF(W, replace(LIFParams(), dt=0.6), batch=2, seed=seed)
    if adapt:
        s.enable_adaptation(adapt, 200.0)
    if plastic:
        s.enable_plasticity(PRE, POST, MOD, np.ones((len(POST), len(MOD)), np.float32))
        s.set_learning([rate, rate], [0.0, 0.0], [50.0, 50.0], [50.0, 50.0])
    s.set_poisson_input(PRE, 150.0)
    if dopamine:
        r = np.zeros((len(MOD), 2), np.float32); r[:, 1] = dopamine     # dopamine only in brain 1
        s.set_poisson_input(MOD, r)
    return s, s.run(steps)


def test_plasticity_with_zero_rate_changes_nothing():
    W = net()
    _, a = run(W)
    _, b = run(W, plastic=True, rate=0.0)
    assert (a == b).all()


def test_dopamine_weakens_only_where_it_arrives():
    s, _ = run(net(), plastic=True, rate=1e-2, dopamine=200.0)
    ps = s.pscale.cpu().numpy()
    assert ps.shape[0] == len(PRE) * len(POST)
    assert ps[:, 0].min() == 1.0                 # brain 0 got no dopamine
    assert ps[:, 1].mean() < 0.99                # brain 1 did, and its synapses weakened
    assert ps.min() >= 0.0


def test_reset_plasticity_restores_published_strength():
    s, _ = run(net(), plastic=True, rate=1e-2, dopamine=200.0)
    s.reset_plasticity(np.array([False, True]))
    assert float(s.pscale.min()) == 1.0


def test_adaptation_is_off_by_default_and_reduces_firing_when_on():
    W = net()
    _, a = run(W)
    _, b = run(W, adapt=0.0)
    assert (a == b).all()
    _, c = run(W, adapt=3.0)
    assert c.sum() < a.sum()


def test_threshold_offset_silences_only_its_neurons():
    W = net()
    s = EventLIF(W, replace(LIFParams(), dt=0.6), batch=1, seed=1)
    s.set_threshold_offset(POST, 1000.0)
    s.set_poisson_input(PRE, 150.0)
    cnt = s.run(300)[:, 0]
    assert cnt[POST].sum() == 0 and cnt[PRE].sum() > 0


def test_odour_is_stronger_on_the_side_of_the_food():
    from brain.mushroom_body import odour_features
    from game.environment import GameConfig, SnakeGame
    g = SnakeGame(GameConfig(width=20, height=20), 0)
    hx, hy = g.head
    f = g.heading
    from game.environment import HEADINGS
    i = HEADINGS.index(f)
    right = HEADINGS[(i + 1) % 4]
    g.food = (hx + 3 * right[0], hy + 3 * right[1])
    left_c, right_c = odour_features(g)
    assert right_c > left_c > 0
    g.food = (hx - 3 * right[0], hy - 3 * right[1])
    left_c, right_c = odour_features(g)
    assert left_c > right_c > 0


def test_learning_genome_extends_old_genomes_and_keeps_their_genes():
    from agents.policies import FLY_GENOME, FLY_GENOME_LEARN, FLY_GENOME_V1, to_learning_genome
    old = np.random.default_rng(0).random((2, FLY_GENOME)).astype(np.float32)
    new = to_learning_genome(old)
    assert new.shape == (2, FLY_GENOME_LEARN)
    assert np.array_equal(new[:, :FLY_GENOME], old)
    v1 = np.random.default_rng(1).random((2, FLY_GENOME_V1)).astype(np.float32)
    assert to_learning_genome(v1).shape == (2, FLY_GENOME_LEARN)
