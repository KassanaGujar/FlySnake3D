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
from brain.visual_encoder import N_FEAT, N_FEAT_V1, N_POP, Encoder

# ----------------------------------------------------------------- fly brain
FLY_GENOME = N_FEAT * N_POP + N_OUT + 1
FLY_GENOME_V1 = N_FEAT_V1 * N_POP + N_OUT + 1      # stage 3-5 genomes (54 genes)


def upgrade_fly_genome(g: np.ndarray) -> np.ndarray:
    """Stage 3-5 genome -> current layout, with gain 0 for the added features.

    Gains are stored feature-major and the new features were appended, so the
    old gains keep their meaning and the brain behaves exactly as before.
    """
    g = np.asarray(g, np.float32)
    if g.shape[-1] == FLY_GENOME:
        return g
    assert g.shape[-1] == FLY_GENOME_V1, g.shape
    k = N_FEAT_V1 * N_POP
    pad = np.zeros(g.shape[:-1] + ((N_FEAT - N_FEAT_V1) * N_POP,), np.float32)
    return np.concatenate([g[..., :k], pad, g[..., k:]], -1)
FLY_INIT_MEAN = np.concatenate([np.full(N_FEAT * N_POP, 0.25), np.full(N_OUT, 0.5), [0.25]])

# Stage 7 learning fly: the base genome above, then
#   gains feature -> mushroom-body visual type   N_FEAT * N_MBV
#   gains odour -> ORN type                      N_ODOUR
#   learning genes (see brain/mushroom_body.py)  N_LEARN
from brain.mushroom_body import N_LEARN, N_MBV, N_ODOUR  # noqa: E402
FLY_GENOME_LEARN = FLY_GENOME + N_FEAT * N_MBV + N_ODOUR + N_LEARN
LEARN_DEFAULTS = np.concatenate([np.full(N_FEAT * N_MBV, 0.1), np.full(N_ODOUR, 0.25),
                                 np.full(N_LEARN, 0.5)]).astype(np.float32)
FLY_INIT_MEAN_LEARN = np.concatenate([FLY_INIT_MEAN, LEARN_DEFAULTS])


def to_learning_genome(g: np.ndarray) -> np.ndarray:
    """Any earlier fly genome -> stage 7 layout (new genes at LEARN_DEFAULTS)."""
    g = np.asarray(g, np.float32)
    if g.shape[-1] == FLY_GENOME_LEARN:
        return g
    g = upgrade_fly_genome(g)
    return np.concatenate([g, np.broadcast_to(LEARN_DEFAULTS, g.shape[:-1] + LEARN_DEFAULTS.shape)], -1)


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
                 device: str | None = None, seed: int = 0, W=None, record_col: int | None = None,
                 readout_from_ms: float = 0.0, regular_input: bool = False,
                 learning: bool = False, theta_scale: float | None = None):
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
        # Descending-neuron spikes before this time are not counted: early in the
        # window they still reflect the previous view (signals need several
        # synaptic delays to cross the brain). The brain itself runs throughout.
        self.skip = int(round(readout_from_ms / dt))
        self.regular_input = regular_input
        assert 0 <= self.skip < self.steps
        from brain.motor_decoder import THETA_SCALE
        self.theta_scale = THETA_SCALE if theta_scale is None else theta_scale
        # Stage 7: smell, vision into the mushroom body, dopamine learning.
        self.learning = learning
        self.uses_odour = learning
        if learning:
            from brain.mushroom_body import MB_VISUAL, ODOUR_GLOMERULI, SideEncoder, mushroom_body_wiring
            self.mbv = SideEncoder(conn, torch, dev, MB_VISUAL)
            self.orn = SideEncoder(conn, torch, dev, ODOUR_GLOMERULI, prefix="ORN_")
            self.mb = mushroom_body_wiring(conn)
            self.sim.enable_plasticity(self.mb["kc"], self.mb["mbon"], self.mb["dan"], self.mb["M"])
            self.ate = np.zeros(self.B, bool)
            self.died = np.zeros(self.B, bool)
        self.set_genomes(genomes)
        self.record_col = record_col
        self.spike_log: list[np.ndarray] = []      # per decision: neurons that spiked in record_col
        self.last_counts = None

    def set_genomes(self, genomes: np.ndarray):
        genomes = to_learning_genome(genomes) if self.learning else upgrade_fly_genome(genomes)
        g = self.t.as_tensor(genomes, device=self.sim.device)
        k = N_FEAT * N_POP
        self.gains = g[:, :k].reshape(self.B, N_FEAT, N_POP).contiguous()
        self.weights = g[:, k:k + N_OUT] * 2 - 1
        self.theta = g[:, k + N_OUT]
        if self.learning:
            from brain.mushroom_body import learning_params
            i = FLY_GENOME
            self.mbv_gains = g[:, i:i + N_FEAT * N_MBV].reshape(self.B, N_FEAT, N_MBV).contiguous()
            i += N_FEAT * N_MBV
            self.orn_gains = g[:, i:i + N_ODOUR].reshape(self.B, 1, N_ODOUR).contiguous()
            i += N_ODOUR
            lp = learning_params(genomes[:, i:i + N_LEARN])
            self.sim.set_learning(lp["rate"], lp["recovery"], lp["tau_pre_ms"], lp["tau_mod_ms"])
            self.reward_hz = self.t.as_tensor(lp["reward_hz"], device=self.sim.device, dtype=self.t.float32)
            self.punish_hz = self.t.as_tensor(lp["punish_hz"], device=self.sim.device, dtype=self.t.float32)
            self.new_evaluation()

    def new_evaluation(self):
        """Forget everything learned: synapses back to their published strengths."""
        if self.learning:
            self.sim.reset_plasticity(np.ones(self.B, bool))
            self.ate[:] = False
            self.died[:] = False

    def feedback(self, ate: np.ndarray, died: np.ndarray):
        """What happened after the last move. Drives dopamine during the next move."""
        if self.learning:
            self.ate = np.asarray(ate, bool)
            self.died = np.asarray(died, bool)

    def reset(self, mask):
        self.sim.reset_columns(mask)

    def act(self, feats: np.ndarray, odour: np.ndarray | None = None) -> np.ndarray:
        sim = self.sim
        sim.clear_inputs()
        sim.set_poisson_input(self.enc.indices, self.enc.rates(feats, self.gains),
                              regular=self.regular_input)
        if self.learning:
            t = self.t
            sim.set_poisson_input(self.mbv.indices, self.mbv.rates(feats, self.mbv_gains),
                                  regular=self.regular_input)
            od = np.zeros((self.B, 2, 1), np.float32) if odour is None else np.asarray(odour, np.float32)[:, :, None]
            sim.set_poisson_input(self.orn.indices, self.orn.rates(od, self.orn_gains),
                                  regular=self.regular_input)
            dev = sim.device
            ate = t.as_tensor(self.ate, device=dev, dtype=t.float32)
            died = t.as_tensor(self.died, device=dev, dtype=t.float32)
            pam, ppl1 = self.mb["pam"], self.mb["ppl1"]
            sim.set_poisson_input(pam, (self.reward_hz * ate)[None].expand(len(pam), self.B).contiguous())
            sim.set_poisson_input(ppl1, (self.punish_hz * died)[None].expand(len(ppl1), self.B).contiguous())
            self.ate[:] = False
            self.died[:] = False
        counts = self.dec.new_counts(self.B)
        rec = []
        for i in range(self.steps):
            pre, b = sim.step()
            if len(pre):
                if i >= self.skip:
                    self.dec.accumulate(counts, pre, b, self.B)
                if self.record_col is not None:
                    rec.append(pre[b == self.record_col])
        if self.record_col is not None:
            self.spike_log.append(self.t.unique(self.t.cat(rec)).cpu().numpy().astype(np.int32)
                                  if rec else np.zeros(0, np.int32))
        self.last_counts = counts
        return self.dec.actions(counts, self.weights, self.theta, self.B, self.theta_scale)


# ----------------------------------------------------------------- ANN control
ANN_HIDDEN = 8
ANN_IN = 2 * N_FEAT
ANN_GENOME = ANN_IN * ANN_HIDDEN + ANN_HIDDEN + ANN_HIDDEN * 3 + 3
ANN_INIT_MEAN = np.full(ANN_GENOME, 0.5)
ANN_GENOME_V1 = 2 * N_FEAT_V1 * ANN_HIDDEN + ANN_HIDDEN + ANN_HIDDEN * 3 + 3   # stage 3-5 (131)
ANN_GENOME_ODOUR = 2 * (N_FEAT + 1) * ANN_HIDDEN + ANN_HIDDEN + ANN_HIDDEN * 3 + 3   # stage 7: + smell
ANN_INIT_MEAN_ODOUR = np.full(ANN_GENOME_ODOUR, 0.5)


class AnnPolicy:
    """features (12) -> tanh(8) -> 3 action scores -> argmax. Weights = 4 * (gene - 0.5)."""
    name = "ann"

    def __init__(self, genomes: np.ndarray):
        w = (np.asarray(genomes, np.float32) - 0.5) * 4.0
        self.B = len(w)
        # Stage 3-5 genomes only see the first six features of each eye.
        self.n_feat = N_FEAT_V1 if w.shape[1] == ANN_GENOME_V1 else N_FEAT
        self.uses_odour = w.shape[1] == ANN_GENOME_ODOUR
        n_in = 2 * (self.n_feat + self.uses_odour)
        i = 0
        self.W1 = w[:, i:i + n_in * ANN_HIDDEN].reshape(self.B, n_in, ANN_HIDDEN); i += n_in * ANN_HIDDEN
        self.b1 = w[:, i:i + ANN_HIDDEN]; i += ANN_HIDDEN
        self.W2 = w[:, i:i + ANN_HIDDEN * 3].reshape(self.B, ANN_HIDDEN, 3); i += ANN_HIDDEN * 3
        self.b2 = w[:, i:i + 3]

    def reset(self, mask):
        pass

    def act(self, feats: np.ndarray, odour: np.ndarray | None = None) -> np.ndarray:
        x = feats[:, :, :self.n_feat]
        if self.uses_odour:
            x = np.concatenate([x, np.asarray(odour, np.float32)[:, :, None]], -1)
        x = x.reshape(self.B, -1)
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
