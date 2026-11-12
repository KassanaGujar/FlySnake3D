"""Batched controllers that all see exactly the same visual features.

Every policy here maps ``feats`` (B, 2 eyes, N_FEAT) to B actions, so the fly
brain and its controls differ ONLY in what sits between features and action:

  FlyPolicy       features -> visual neurons -> FlyWire connectome (LIF) ->
                  descending neurons -> action.        Evolved: 54 numbers.
  AnnPolicy       features -> small tanh network -> action. The "ordinary
                  machine-learning model" control.     Evolved: 131 numbers.
  ReactivePolicy  hand-written reflex (control, not evolved).
  RandomPolicy    uniform random actions (control).

Genomes are vectors in [0, 1]^D ("gene space"); each policy documents how
genes map to its parameters.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from brain.motor_decoder import N_OUT, Decoder
from brain.neural_simulator import LIFParams
from brain.visual_encoder import N_FEAT, N_POP, Encoder

# ----------------------------------------------------------------- fly brain
FLY_GENOME = N_FEAT * N_POP + N_OUT + 1
FLY_INIT_MEAN = np.concatenate([np.full(N_FEAT * N_POP, 0.25), np.full(N_OUT, 0.5), [0.25]])


def fly_genome_names() -> list[str]:
    from brain.motor_decoder import READOUT_TYPES
    from brain.visual_encoder import FEATURES, POPULATIONS
    return ([f"gain[{f}->{p}]" for f in FEATURES for p in POPULATIONS]
            + [f"weight[{d}]" for d in READOUT_TYPES] + ["theta"])


class FlyPolicy:
    """B fly brains, one per column, each with its own evolved interface.

    genes[: F*P]      encoder gains in [0, 1]        (feature f -> population p)
    genes[F*P : -1]   decoder weights, 2*g - 1 in [-1, 1]
    genes[-1]         decoder threshold theta in [0, 1]

    The connectome itself (wiring, synapse counts, signs, LIF constants) is
    never changed. Brains run continuously: state carries over from one
    decision to the next and is reset only when that column's game restarts.
    """
    name = "fly"

    def __init__(self, conn, genomes: np.ndarray, dt: float = 0.6, window_ms: float = 24.0,
                 device: str | None = None, seed: int = 0, W=None, record_col: int | None = None):
        import torch

        from brain.event_simulator import EventLIF
        self.t = torch
        self.B = len(genomes)
        self.sim = EventLIF(conn.W if W is None else W, replace(LIFParams(), dt=dt),
                            batch=self.B, device=device, seed=seed)
        dev = self.sim.device
        self.enc = Encoder(conn, torch, dev)
        self.dec = Decoder(conn, torch, dev)
        self.steps = int(round(window_ms / dt))
        self.set_genomes(genomes)
        self.record_col = record_col
        self.spike_log: list[np.ndarray] = []      # per decision: neurons that spiked in record_col
        self.last_counts = None

    def set_genomes(self, genomes: np.ndarray):
        g = self.t.as_tensor(np.asarray(genomes, np.float32), device=self.sim.device)
        k = N_FEAT * N_POP
        self.gains = g[:, :k].reshape(self.B, N_FEAT, N_POP).contiguous()
        self.weights = g[:, k:k + N_OUT] * 2 - 1
        self.theta = g[:, -1]

    def reset(self, mask):
        self.sim.reset_columns(mask)

    def act(self, feats: np.ndarray) -> np.ndarray:
        sim = self.sim
        sim.clear_inputs()
        sim.set_poisson_input(self.enc.indices, self.enc.rates(feats, self.gains))
        counts = self.dec.new_counts(self.B)
        rec = []
        for _ in range(self.steps):
            pre, b = sim.step()
            if len(pre):
                self.dec.accumulate(counts, pre, b, self.B)
                if self.record_col is not None:
                    rec.append(pre[b == self.record_col])
        if self.record_col is not None:
            self.spike_log.append(self.t.unique(self.t.cat(rec)).cpu().numpy().astype(np.int32)
                                  if rec else np.zeros(0, np.int32))
        self.last_counts = counts
        return self.dec.actions(counts, self.weights, self.theta, self.B)


# ----------------------------------------------------------------- ANN control
ANN_HIDDEN = 8
ANN_IN = 2 * N_FEAT
ANN_GENOME = ANN_IN * ANN_HIDDEN + ANN_HIDDEN + ANN_HIDDEN * 3 + 3
ANN_INIT_MEAN = np.full(ANN_GENOME, 0.5)


class AnnPolicy:
    """features (12) -> tanh(8) -> 3 action scores -> argmax. Weights = 4 * (gene - 0.5)."""
    name = "ann"

    def __init__(self, genomes: np.ndarray):
        w = (np.asarray(genomes, np.float32) - 0.5) * 4.0
        self.B = len(w)
        i = 0
        self.W1 = w[:, i:i + ANN_IN * ANN_HIDDEN].reshape(self.B, ANN_IN, ANN_HIDDEN); i += ANN_IN * ANN_HIDDEN
        self.b1 = w[:, i:i + ANN_HIDDEN]; i += ANN_HIDDEN
        self.W2 = w[:, i:i + ANN_HIDDEN * 3].reshape(self.B, ANN_HIDDEN, 3); i += ANN_HIDDEN * 3
        self.b2 = w[:, i:i + 3]

    def reset(self, mask):
        pass

    def act(self, feats: np.ndarray) -> np.ndarray:
        x = feats.reshape(self.B, ANN_IN)
        h = np.tanh(np.einsum("bi,bih->bh", x, self.W1) + self.b1)
        return (np.einsum("bh,bho->bo", h, self.W2) + self.b2).argmax(1)


# ----------------------------------------------------------------- simple controls
class RandomPolicy:
    name = "random"

    def __init__(self, B: int, seed: int = 0):
        self.B, self.rng = B, np.random.default_rng(seed)

    def reset(self, mask):
        pass

    def act(self, feats):
        return self.rng.integers(0, 3, self.B)


class ReactivePolicy:
    """Hand-written reflex on the same features. A CONTROL: it shows what a few
    explicit rules achieve. Rules like these are forbidden inside the fly agent.

    1. Something directly ahead: turn to the side whose view is more open.
    2. Otherwise, food more to one side than ahead and that side is open: turn to it.
    3. Otherwise go straight.
    """
    name = "reactive"

    def __init__(self, B: int):
        self.B = B

    def reset(self, mask):
        pass

    def act(self, feats):
        act = np.ones(self.B, np.int64)
        for b in range(self.B):
            L, R = feats[b]
            if L[2] >= 1.0:                                   # obstacle in the next cell
                act[b] = 0 if L[0] < R[0] else 2
            elif L[4] > L[5] and L[0] < 1.0:
                act[b] = 0
            elif R[4] > R[5] and R[0] < 1.0:
                act[b] = 2
        return act
