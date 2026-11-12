"""Event-driven version of the batched LIF simulator.

Same model and same update order as ``TorchLIF`` (Shiu et al. 2024, see
neural_simulator.py). The only difference is how synaptic input is delivered:

  TorchLIF   every step: g += W @ delayed_spikes     (touches all 15 M synapses)
  EventLIF   every step: for each neuron that spiked, add its outgoing
             synapses to g                            (touches only active ones)

In this model the brain is silent without input and only a few hundred
neurons are active under visual stimulation, so almost all of the matrix
product is multiplication by zero. Delivering spikes as events is the same
arithmetic with the zeros skipped. It is validated against TorchLIF in
tests/test_event_simulator.py (same seed -> same spike counts).

Extra over TorchLIF, needed to run many games in parallel:
  reset_columns(mask)   reset the state of selected brains only
  step() returns the (neuron, brain) index pairs that spiked, not a dense mask
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from brain.neural_simulator import LIFParams


class EventLIF:
    def __init__(self, W: sp.spmatrix, params: LIFParams = LIFParams(), batch: int = 1,
                 device: str | None = None, seed: int = 0):
        import torch

        self.torch = t = torch
        self.p = p = params
        self.N = W.shape[0]
        self.B = batch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = t.float32
        self.gen = t.Generator(device=self.device).manual_seed(seed)

        # CSC: column j lists the postsynaptic targets of presynaptic neuron j.
        Wc = W.tocsc().astype(np.float32)
        Wc.sort_indices()
        self.indptr = t.from_numpy(Wc.indptr.astype(np.int64)).to(self.device)
        self.rows = t.from_numpy(Wc.indices.astype(np.int32)).to(self.device)
        self.vals = t.from_numpy(Wc.data * p.w_syn).to(self.device)

        self.decay_g = float(np.exp(-p.dt / p.tau))
        self.decay_v = float(np.exp(-p.dt / p.t_mbr))
        self.A_coef = p.tau / (p.tau - p.t_mbr)
        self.delay_steps = int(round(p.t_dly / p.dt))
        self.rfc_steps = int(round(p.t_rfc / p.dt))
        self.rfc_per_neuron = t.full((self.N,), self.rfc_steps, device=self.device, dtype=t.int16)
        self._inputs: list[tuple] = []
        self.reset()

    # -- state -------------------------------------------------------------
    # The membrane is stored as u = v - v_0 so the sub-threshold update is two
    # in-place passes. Refractory neurons are tracked as a short list of recent
    # spike events instead of a dense counter.
    def reset(self):
        t = self.torch
        self.u = t.zeros((self.N, self.B), device=self.device, dtype=self.dtype)
        self.g = t.zeros((self.N, self.B), device=self.device, dtype=self.dtype)
        self.ring: list = [None] * self.delay_steps       # spikes waiting out the synaptic delay
        self.refq: list = [None] * max(self.rfc_steps, 1)  # flat indices still refractory
        self.t_step = 0

    @property
    def v(self):
        return self.u + self.p.v_0

    def reset_columns(self, mask):
        """Reset brains where ``mask`` (B,) is True: membrane, synapses, pending spikes."""
        t = self.torch
        m = t.as_tensor(np.asarray(mask), device=self.device, dtype=t.bool)
        if not bool(m.any()):
            return
        self.u[:, m] = 0
        self.g[:, m] = 0
        for k, ev in enumerate(self.ring):
            if ev is not None:
                keep = ~m[ev[1]]
                self.ring[k] = (ev[0][keep], ev[1][keep]) if bool(keep.any()) else None
        for k, flat in enumerate(self.refq):
            if flat is not None:
                keep = ~m[flat % self.B]
                self.refq[k] = flat[keep] if bool(keep.any()) else None

    # -- stimulation -----------------------------------------------------
    def clear_inputs(self):
        self._inputs = []
        self.rfc_per_neuron.fill_(self.rfc_steps)

    def set_poisson_input(self, indices, rates_hz):
        """Same contract as TorchLIF.set_poisson_input (stimulated neurons have no refractory period)."""
        t = self.torch
        idx = t.as_tensor(np.asarray(indices), device=self.device, dtype=t.long)
        r = t.as_tensor(rates_hz, device=self.device, dtype=self.dtype) if t.is_tensor(rates_hz) \
            else t.as_tensor(np.asarray(rates_hz, dtype=np.float32), device=self.device)
        if r.ndim == 0:
            r = r.expand(len(idx))
        if r.ndim == 1:
            r = r[:, None].expand(len(idx), self.B)
        p_spike = (r * self.p.dt * 1e-3).clamp(0, 1).contiguous()
        self._inputs.append((idx, p_spike))
        self.rfc_per_neuron[idx] = 0

    # -- integration -------------------------------------------------------
    def _deliver(self, pre, b):
        """g[post, b] += w for every synapse of every (pre, b) spike event."""
        t = self.torch
        starts = self.indptr[pre]
        lens = self.indptr[pre + 1] - starts
        total = int(lens.sum())
        if total == 0:
            return
        ev = t.repeat_interleave(t.arange(len(pre), device=self.device), lens)
        first = t.cumsum(lens, 0) - lens
        syn = starts[ev] + (t.arange(total, device=self.device) - first[ev])
        flat = self.rows[syn].long() * self.B + b[ev]
        self.g.view(-1).index_add_(0, flat, self.vals[syn])

    def step(self):
        t = self.torch
        p = self.p
        # 1) exact sub-threshold update, frozen while refractory.
        #    u' = u*dv + g*A_coef*(dg - dv),  g' = g*dg   (same closed form as TorchLIF)
        held = [f for f in self.refq if f is not None] if self.rfc_steps > 0 else []
        if held:
            hidx = t.cat(held)
            hu, hg = self.u.view(-1)[hidx], self.g.view(-1)[hidx]
        self.u.mul_(self.decay_v).add_(self.g, alpha=self.A_coef * (self.decay_g - self.decay_v))
        self.g.mul_(self.decay_g)
        if held:
            self.u.view(-1)[hidx] = hu
            self.g.view(-1)[hidx] = hg

        # 2) threshold
        nz = (self.u > (p.v_th - p.v_0)).nonzero()
        pre, b = nz[:, 0], nz[:, 1]

        # 3) synaptic input from spikes emitted delay_steps ago, then Poisson input
        slot = self.t_step % self.delay_steps
        ev = self.ring[slot]
        if ev is not None:
            self._deliver(*ev)
        self.ring[slot] = (pre, b) if len(pre) else None
        for idx, p_spike in self._inputs:
            hits = t.rand(p_spike.shape, generator=self.gen, device=self.device) < p_spike
            self.u[idx] += hits.to(self.dtype) * (p.w_syn * p.f_poi)

        # 4) reset the neurons that spiked and start their refractory period
        rslot = self.t_step % max(self.rfc_steps, 1)
        self.refq[rslot] = None
        if len(pre):
            flat = pre * self.B + b
            self.u.view(-1)[flat] = p.v_rst - p.v_0
            self.g.view(-1)[flat] = 0
            if self.rfc_steps > 0:
                r = flat[self.rfc_per_neuron[pre] > 0]
                self.refq[rslot] = r if len(r) else None
        self.t_step += 1
        return pre, b

    def run(self, n_steps: int, record=None):
        """Advance ``n_steps``; return spike counts (N, B), or (len(record), B)."""
        t = self.torch
        counts = t.zeros(self.N * self.B, device=self.device, dtype=t.int32)
        one = t.ones(1, device=self.device, dtype=t.int32)
        for _ in range(n_steps):
            pre, b = self.step()
            if len(pre):
                counts.index_add_(0, pre * self.B + b, one.expand(len(pre)))
        counts = counts.view(self.N, self.B)
        if record is not None:
            counts = counts[t.as_tensor(np.asarray(record), device=self.device, dtype=t.long)]
        return counts.cpu().numpy()
