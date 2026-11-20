"""Stage 8: per-move teaching signals and MBON readout (no connectome needed)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from brain.motor_decoder import N_OUT, READOUT_TYPES, Decoder  # noqa: E402
from brain.mushroom_body import food_distance, move_signals  # noqa: E402
from game.environment import HEADINGS, GameConfig, SnakeGame  # noqa: E402


def test_danger_only_when_the_cell_straight_ahead_is_blocked():
    g = SnakeGame(GameConfig(width=20, height=20), 0)
    seen_clear = False
    while True:                              # drive straight until the wall is next
        ahead = (g.head[0] + g.heading[0], g.head[1] + g.heading[1])
        danger, _ = move_signals(g, None)
        if not g.in_bounds(ahead):
            assert danger
            break
        assert not danger
        seen_clear = True
        g.step(1)
        assert not g.done
    assert seen_clear
    g.step(1)                                # into the wall: game over, no move signal
    assert g.done and move_signals(g, None) == (False, False)


def test_closer_only_when_distance_to_food_drops():
    g = SnakeGame(GameConfig(width=20, height=20), 3)
    hx, hy = g.head
    f = g.heading
    g.food = (hx + 5 * f[0], hy + 5 * f[1])
    if not g.in_bounds(g.food):
        g.food = (hx - 5 * f[0], hy - 5 * f[1])
    d = food_distance(g)
    toward = 1 if g.food == (hx + 5 * f[0], hy + 5 * f[1]) else None
    if toward is None:
        pytest.skip("food behind the head for this seed")
    g.step(1)
    _, closer = move_signals(g, d)
    assert closer
    d = food_distance(g)
    g.step(0)                                # turn away: no longer closer
    _, closer = move_signals(g, d)
    assert not closer


def test_v8_genome_starts_with_zero_mbon_weights_and_keeps_old_genes():
    from agents.policies import FLY_GENOME_LEARN, FLY_GENOME_V8, to_v8_genome
    from brain.mushroom_body import N_MBON
    old = np.random.default_rng(0).random((3, FLY_GENOME_LEARN)).astype(np.float32)
    new = to_v8_genome(old)
    assert new.shape == (3, FLY_GENOME_V8)
    assert np.array_equal(new[:, :FLY_GENOME_LEARN], old)
    assert np.allclose(new[:, FLY_GENOME_LEARN:FLY_GENOME_LEARN + N_MBON] * 2 - 1, 0)


class _FakeConn:
    """Readout DN types get neurons 0..9 (left, right per type); 2 extra groups after."""

    def __init__(self):
        import scipy.sparse as sp
        self.W = sp.csr_matrix((20, 20))
        self.map = {}
        for k, t in enumerate(READOUT_TYPES):
            self.map[(t, "left")] = np.array([2 * k])
            self.map[(t, "right")] = np.array([2 * k + 1])

    def indices_of_type(self, t, side=None):
        return self.map[(t, side)]


def test_zero_extra_weights_leave_the_decision_unchanged_and_extra_groups_steer():
    conn = _FakeConn()
    extra = [("M1", np.array([12]), np.array([13])), ("M2", np.array([14]), np.array([15]))]
    plain = Decoder(conn, torch, "cpu")
    ext = Decoder(conn, torch, "cpu", extra=extra)
    B = 4
    rng = np.random.default_rng(0)
    pre_dn = torch.as_tensor(rng.integers(0, 10, 200))
    b_dn = torch.as_tensor(rng.integers(0, B, 200))
    pre_mb = torch.as_tensor(np.full(50, 13))          # M1 right fires a lot
    b_mb = torch.as_tensor(rng.integers(0, B, 50))
    c0, c1 = plain.new_counts(B), ext.new_counts(B)
    plain.accumulate(c0, pre_dn, b_dn, B)
    ext.accumulate(c1, torch.cat([pre_dn, pre_mb]), torch.cat([b_dn, b_mb]), B)
    w = torch.as_tensor(rng.uniform(-1, 1, (B, N_OUT)), dtype=torch.float32)
    theta = torch.full((B,), 0.3)
    a0 = plain.actions(c0, w, theta, B, 8.0)
    a1 = ext.actions(c1, torch.cat([w, torch.zeros(B, 2)], 1), theta, B, 8.0)
    assert np.array_equal(a0, a1)
    # A strong positive weight on M1 makes its right-side spikes turn right.
    w2 = torch.cat([torch.zeros(B, N_OUT), torch.tensor([[1.0, 0.0]]).repeat(B, 1)], 1)
    a2 = ext.actions(c1, w2, torch.zeros(B), B, 8.0)
    hit = np.bincount(b_mb.numpy(), minlength=B) > 0
    assert (a2[hit] == 2).all()


def test_overlapping_readout_groups_are_rejected():
    conn = _FakeConn()
    with pytest.raises(AssertionError):
        Decoder(conn, torch, "cpu", extra=[("bad", np.array([0]), np.array([19]))])
