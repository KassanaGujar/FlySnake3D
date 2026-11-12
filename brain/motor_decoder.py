"""Motor decoder: spikes of descending neurons -> LEFT / STRAIGHT / RIGHT.

The decoder is deliberately tiny and contains no game knowledge:

    turn = sum_k  w_k * (spikes of type k on the right - spikes on the left)
    RIGHT if turn > theta,  LEFT if turn < -theta,  else STRAIGHT

It never sees the board, the food, distances or any plan. The weights w_k
(one signed number per cell type) and theta are evolved, so evolution decides
whether a neuron type steers towards or away from its own side.

Readout cell types. All are descending neurons (brain -> ventral nerve cord)
reported to be involved in walking / steering. That is why they are read;
whether they carry useful signals here is an empirical question:
  DNa01, DNa02   steering during walking   Rayshubskiy et al. 2020; Namiki et al. 2018
  DNa03, DNg13   steering circuit          Yang et al. 2024
  DNp09          walking initiation/turns  Bidaye et al. 2020
(Citations are from memory of the literature and should be checked before
publication; the cell types themselves are verified to exist in FlyWire v783.)
"""
from __future__ import annotations

import numpy as np

READOUT_TYPES = ["DNa01", "DNa02", "DNa03", "DNg13", "DNp09"]
N_OUT = len(READOUT_TYPES)
THETA_SCALE = 4.0     # theta gene in [0, 1] -> threshold in weighted spikes per decision


class Decoder:
    def __init__(self, conn, torch, device):
        self.t = torch
        slot = np.full(conn.W.shape[0], -1, np.int64)
        self.indices = {}
        for k, ctype in enumerate(READOUT_TYPES):
            for s, side in enumerate(("left", "right")):
                ii = conn.indices_of_type(ctype, side=side)
                slot[ii] = 2 * k + s
                self.indices[(ctype, side)] = ii
        self.slot = torch.as_tensor(slot, device=device)
        self.device = device

    def new_counts(self, B: int):
        return self.t.zeros(2 * N_OUT * B, device=self.device)

    def accumulate(self, counts, pre, b, B: int):
        """Add this step's spikes (pre, b) of readout neurons to ``counts``."""
        s = self.slot[pre]
        m = s >= 0
        if bool(m.any()):
            counts.index_add_(0, s[m] * B + b[m], self.t.ones(int(m.sum()), device=self.device))

    def actions(self, counts, weights, theta, B: int) -> np.ndarray:
        """counts (2*N_OUT*B,), weights (B, N_OUT) in [-1, 1], theta (B,) in [0, 1] -> actions (B,)."""
        c = counts.view(N_OUT, 2, B)
        turn = ((c[:, 1] - c[:, 0]).T * weights).sum(1)          # right minus left
        th = theta * THETA_SCALE
        act = self.t.ones(B, dtype=self.t.long, device=self.device)      # STRAIGHT
        act[turn > th] = 2
        act[turn < -th] = 0
        return act.cpu().numpy()
