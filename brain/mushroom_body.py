"""Stage 7: smell, visual input to the mushroom body, and dopamine learning.

Everything here is chosen from the FlyWire v783 wiring and annotations, not
invented. Where a choice rests on the literature, the citation is given and
(like the other citations in this project) still needs checking.

1. SMELL  ``odour_features(game)`` gives the food-odour concentration at the
   left and right antenna: exp(-distance / ODOUR_LENGTH_CELLS), with the
   antennae a quarter cell ahead of and a quarter cell to each side of the
   centre of the head cell. It is a smell, not a direction finder: the two
   values differ only slightly, as for a real fly. ``OdourEncoder`` turns it
   into firing of olfactory receptor neurons (ORNs) of a few food-odour
   glomeruli on the same side, with one evolved gain per ORN type.
     Food-odour glomeruli used (vinegar / fermenting fruit; Semmelhack & Wang
     2009, Becher et al. 2010; from memory, needs checking):
     DM1, DM2, DM4, DP1m, VA2, VM2.
   FlyWire `side` for ORNs is taken as the antenna side (assumption).

2. VISION INTO THE MUSHROOM BODY  The stage 3-6 visual types (LC/LPLC) have
   essentially no synapses onto Kenyon cells (1 in total; see
   results/stage7/learning_pathway_probe.json). The visual neuron types that
   do feed Kenyon cells were ranked by synapse count; the top 8 are driven by
   the same visual features through their own evolved gains (``MB_VISUAL``).

3. DOPAMINE LEARNING  Kenyon cell -> MBON synapses become plastic (see
   EventLIF.enable_plasticity). Eating drives the PAM dopamine neurons and
   crashing drives the PPL1 dopamine neurons (PAM ~ reward, PPL1 ~
   punishment: Waddell 2013; Aso et al. 2014; from memory, needs checking).
   How strongly each dopamine neuron acts on each MBON is its synapse count
   onto that MBON in FlyWire, scaled so the largest is 1. The rule weakens a
   Kenyon cell's synapse when that cell was active shortly before dopamine
   reached the MBON (Hige et al. 2015; Handler et al. 2019; from memory), and
   lets it recover slowly. Evolution sets learning rate, recovery, the two
   trace time constants, and how hard eating / crashing drive the dopamine
   neurons.
"""
from __future__ import annotations

import math

import numpy as np

from game.environment import HEADINGS, SnakeGame

ODOUR_LENGTH_CELLS = 6.0
ODOUR_GLOMERULI = ["DM1", "DM2", "DM4", "DP1m", "VA2", "VM2"]
N_ODOUR = len(ODOUR_GLOMERULI)
# Top 8 visual types by synapses onto Kenyon cells in FlyWire v783
# (hemibrain names; ranked in experiments/learning_pathway_probe.py's companion survey).
MB_VISUAL = ["aMe12", "MTe32", "MTe30", "LTe25", "MTe40", "aMe20", "aMe26", "LTe16"]
N_MBV = len(MB_VISUAL)
# Stage 7 calibration of the LIF model (NOT in Shiu et al. 2024; chosen in
# experiments/kc_sparseness_calibration.py, see docs/DEVLOG.md 17 Nov 2026):
#   every neuron: +2 mV threshold per spike, decaying over 1 s (spike-frequency
#   adaptation), which stops the olfactory lock-up; Kenyon cells: +8 mV
#   threshold, which gives 5-6% of Kenyon cells per odour. APL unchanged.
ADAPT_MV, ADAPT_TAU_MS = 2.0, 1000.0
KC_THRESHOLD_MV = 8.0
N_LEARN = 6               # rate, recovery, tau_kc, tau_da, reward drive, punishment drive
MAX_RATE_HZ = 200.0
SIDES = ("left", "right")


def odour_features(game: SnakeGame) -> np.ndarray:
    """Food-odour concentration at (left antenna, right antenna), each in (0, 1]."""
    out = np.zeros(2, np.float32)
    if game.food is None:
        return out
    i = HEADINGS.index(game.heading)
    f = game.heading
    right = HEADINGS[(i + 1) % 4]
    hx, hy = game.head
    for e, s in enumerate((-1, 1)):                      # left antenna, right antenna
        ax = hx + 0.25 * f[0] + 0.25 * s * right[0]
        ay = hy + 0.25 * f[1] + 0.25 * s * right[1]
        d = math.hypot(game.food[0] - ax, game.food[1] - ay)
        out[e] = math.exp(-d / ODOUR_LENGTH_CELLS)
    return out


def _type_index(conn, name: str, side: str | None = None) -> np.ndarray:
    a = conn.annotations
    t = a.cell_type.fillna(a.hemibrain_type).fillna("")
    m = (t == name).to_numpy()
    if side is not None:
        m &= (a.side == side).to_numpy()
    return np.flatnonzero(m)


class SideEncoder:
    """Per-side populations driven by per-side inputs with evolved gains.

    inputs (B, 2, F), gains (B, F, P) -> rates (n_neurons, B). Population p on
    side e only receives side e's inputs.
    """

    def __init__(self, conn, torch, device, types: list[str], prefix: str = ""):
        self.t = torch
        idx, group = [], []
        P = len(types)
        for e, side in enumerate(SIDES):
            for p, ctype in enumerate(types):
                ii = _type_index(conn, prefix + ctype, side)
                assert len(ii), f"no neurons of type {prefix + ctype} on the {side}"
                idx.append(ii)
                group.append(np.full(len(ii), e * P + p))
        self.indices = np.concatenate(idx)
        self.group = torch.as_tensor(np.concatenate(group), device=device, dtype=torch.long)
        self.device = device

    def rates(self, inputs: np.ndarray, gains) -> "torch.Tensor":
        t = self.t
        x = t.as_tensor(inputs, device=self.device, dtype=t.float32)
        drive = t.einsum("bef,bfp->bep", x, gains).clamp_(0, 1) * MAX_RATE_HZ
        return drive.reshape(drive.shape[0], -1).T[self.group]


def apply_calibration(sim, conn):
    """Adaptation for every neuron + raised Kenyon cell threshold (stage 7)."""
    import numpy as _np
    sim.enable_adaptation(ADAPT_MV, ADAPT_TAU_MS)
    kc = _np.flatnonzero((conn.annotations.cell_class == "Kenyon_Cell").to_numpy())
    sim.set_threshold_offset(kc, KC_THRESHOLD_MV)


def mushroom_body_wiring(conn):
    """Indices and dopamine->MBON weights for EventLIF.enable_plasticity."""
    a = conn.annotations
    cc = a.cell_class.fillna("")
    ct = a.cell_type.fillna("")
    kc = np.flatnonzero((cc == "Kenyon_Cell").to_numpy())
    mbon = np.flatnonzero((cc == "MBON").to_numpy())
    pam = np.flatnonzero(ct.str.startswith("PAM").to_numpy())
    ppl1 = np.flatnonzero(ct.str.startswith("PPL1").to_numpy())
    dan = np.sort(np.concatenate([pam, ppl1]))
    M = np.abs(conn.W.tocsr()[mbon][:, dan].toarray()).astype(np.float32)   # W[post, pre]
    M /= max(M.max(), 1.0)
    return {"kc": kc, "mbon": mbon, "pam": pam, "ppl1": ppl1, "dan": dan, "M": M}


def learning_params(genes: np.ndarray) -> dict:
    """Six genes in [0, 1] -> learning parameters (arrays over the batch)."""
    g = np.clip(np.asarray(genes, np.float64), 0, 1)
    return {"rate": 10.0 ** (-5 + 4 * g[:, 0]),            # 1e-5 .. 1e-1 per ms
            "recovery": 10.0 ** (-3 + 3 * g[:, 1]),        # 0.001 .. 1 per second
            "tau_pre_ms": 5.0 * 10.0 ** (2 * g[:, 2]),     # 5 .. 500 ms
            "tau_mod_ms": 5.0 * 10.0 ** (2 * g[:, 3]),     # 5 .. 500 ms
            "reward_hz": MAX_RATE_HZ * g[:, 4],
            "punish_hz": MAX_RATE_HZ * g[:, 5]}
