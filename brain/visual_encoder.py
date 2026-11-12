"""Visual encoder: what the snake's head "sees" -> firing rates of visual neurons.

THIS IS AN ENGINEERED INTERFACE, not a model of the fly's retina. Milestone 0
showed that photoreceptor input does not propagate in this LIF model, so
vision enters at the optic lobe's output neurons (visual projection neurons,
"LC"/"LPLC" types), which are known to encode visual features.

Two stages:

1. ``view_features(game)``: egocentric visual features for each eye. They are
   purely sensory quantities about what is visible from the head: how large
   things appear (1 / distance) in a few directions, how fast the view ahead
   is expanding, and where the food appears. There is no path, no plan, no
   look-ahead and no knowledge of which move is safe.

2. ``Encoder.rates(features, gains)``: a non-negative linear map from each
   eye's features to the firing rate of each visual population IN THE SAME
   EYE. The gains are evolved. Evolution decides which feature drives which
   cell type; nothing here says what the fly should do about it.

Which side: each eye only sees its own half of the view, plus the shared
frontal field. FlyWire's ``side`` annotation picks the neurons of each eye.
Within an eye there is NO retinotopy (every neuron of a type gets the same
rate). That is a real limitation: the fly cannot tell "30 degrees left" from
"80 degrees left" except through the separate features below.

Cell types and the published feature each is associated with. The association
is the reason for including the type; it is NOT assumed to hold in Snake:
  LPLC2  looming                       Klapoetke et al. 2017; Ache et al. 2019
  LC4    looming, escape timing        von Reyn et al. 2017
  LPLC1  looming / small object        Tanaka & Clark 2022
  LC6    looming                       Wu et al. 2016
  LC16   looming, backward walking     Wu et al. 2016
  LC9    visual projection, drove DNa01 contralaterally in our Milestone 0 probe
  LC10a  moving-object tracking        Ribeiro et al. 2018
  LC11   small moving object           Keles & Frye 2017
"""
from __future__ import annotations

import math

import numpy as np

from game.environment import HEADINGS, SnakeGame

POPULATIONS = ["LPLC2", "LC4", "LPLC1", "LC6", "LC16", "LC9", "LC10a", "LC11"]
FEATURES = ["obstacle_side", "obstacle_diagonal", "obstacle_front", "looming_front",
            "food_side", "food_front"]
N_FEAT = len(FEATURES)
N_POP = len(POPULATIONS)
SIDES = ("left", "right")
MAX_RATE_HZ = 200.0


def _ray(game: SnakeGame, d: tuple[int, int]) -> int:
    """Cells from the head to the first wall/body/obstacle cell along ``d`` (>= 1)."""
    x, y = game.head
    n = 1
    while True:
        c = (x + d[0] * n, y + d[1] * n)
        if not game.in_bounds(c) or c in game.obstacles or c in game.occupied:
            return n
        n += 1


def view_features(game: SnakeGame, prev_front: float | None = None) -> tuple[np.ndarray, float]:
    """Features for (left eye, right eye): array (2, N_FEAT) in [0, 1], and the
    current frontal apparent size (pass it back as ``prev_front`` next tick)."""
    i = HEADINGS.index(game.heading)
    f = game.heading
    left, right = HEADINGS[(i - 1) % 4], HEADINGS[(i + 1) % 4]
    size = lambda d: 1.0 / _ray(game, d)                      # apparent size ~ 1 / distance
    front = size(f)
    loom = 0.0 if prev_front is None else min(1.0, 2.0 * max(0.0, front - prev_front))
    out = np.zeros((2, N_FEAT), np.float32)
    for e, side in enumerate((left, right)):
        out[e, 0] = size(side)
        out[e, 1] = size((f[0] + side[0], f[1] + side[1]))
        out[e, 2] = front
        out[e, 3] = loom
    if game.food is not None:
        dx, dy = game.food[0] - game.head[0], game.food[1] - game.head[1]
        ahead = dx * f[0] + dy * f[1]
        lateral = dx * right[0] + dy * right[1]                # > 0: food is to the right
        dist = math.hypot(dx, dy)
        s = 0.3 + 0.7 / max(dist, 1.0)                         # apparent size, never invisible
        if ahead > 0:
            out[:, 5] = s * ahead / dist
        if lateral != 0 or ahead < 0:
            e = 1 if lateral > 0 else 0                        # directly behind counts as left
            out[e, 4] = s * (abs(lateral) / dist if ahead >= 0 else 1.0)
    return out, front


class Encoder:
    """Maps features to per-neuron Poisson rates on the simulator's device."""

    def __init__(self, conn, torch, device):
        self.t = torch
        idx, group = [], []
        for e, side in enumerate(SIDES):
            for p, ctype in enumerate(POPULATIONS):
                ii = conn.indices_of_type(ctype, side=side)
                idx.append(ii)
                group.append(np.full(len(ii), e * N_POP + p))
        self.indices = np.concatenate(idx)                     # all stimulated neurons
        self.group = torch.as_tensor(np.concatenate(group), device=device, dtype=torch.long)
        self.device = device

    def rates(self, feats: np.ndarray, gains) -> "torch.Tensor":
        """feats (B, 2, N_FEAT), gains tensor (B, N_FEAT, N_POP) in [0, 1] -> (n_neurons, B) Hz."""
        t = self.t
        f = t.as_tensor(feats, device=self.device)             # (B, 2, F)
        drive = t.einsum("bef,bfp->bep", f, gains).clamp_(0, 1) * MAX_RATE_HZ   # (B, 2, P)
        return drive.reshape(drive.shape[0], -1).T[self.group]  # (n_neurons, B)
