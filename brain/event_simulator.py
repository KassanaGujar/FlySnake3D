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
        # Phase of each neuron's regular input train (used only by regular inputs).
        # Random and different per brain, so trains are not synchronised.
        # Drawn from its own generator so the Poisson stream is unchanged.
        pgen = t.Generator(device=self.device).manual_seed(seed + 7919)
        self.phase = t.rand((self.N, self.B), generator=pgen, device=self.device)
        self.pscale = None                                 # set by enable_plasticity
        self.adapt_mv = 0.0                                # set by enable_adaptation
        self.reset()

    def enable_adaptation(self, step_mv: float, tau_ms: float):
        """Spike-frequency adaptation (NOT in Shiu et al. 2024; off by default).

        Each spike raises that neuron's threshold by ``step_mv``; the extra
        threshold decays with time constant ``tau_ms``. Real neurons slow down
        under sustained drive; the published LIF model does not, and in the
        olfactory system that lets activity lock at the maximum rate forever
        (experiments/olfactory_lockup_probe.py).
        """
        t = self.torch
        self.adapt_mv = float(step_mv)
        self.adapt_decay = float(np.exp(-self.p.dt / tau_ms))
        self.a = t.zeros((self.N, self.B), device=self.device, dtype=self.dtype)

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
        if self.adapt_mv:
            self.a[:, m] = 0
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

    def set_poisson_input(self, indices, rates_hz, regular: bool = False):
        """Same contract as TorchLIF.set_poisson_input (stimulated neurons have no refractory period).

        regular=True replaces the Poisson train by an evenly spaced train of the
        same rate (a phase accumulator per neuron). The mean drive is identical;
        only the spike-count randomness of the input is removed. This is a
        deviation from Shiu et al., who use Poisson input.
        """
        t = self.torch
        idx = t.as_tensor(np.asarray(indices), device=self.device, dtype=t.long)
        r = t.as_tensor(rates_hz, device=self.device, dtype=self.dtype) if t.is_tensor(rates_hz) \
            else t.as_tensor(np.asarray(rates_hz, dtype=np.float32), device=self.device)
        if r.ndim == 0:
            r = r.expand(len(idx))
        if r.ndim == 1:
            r = r[:, None].expand(len(idx), self.B)
        p_spike = (r * self.p.dt * 1e-3).clamp(0, 1).contiguous()
        self._inputs.append((idx, p_spike, regular))
        self.rfc_per_neuron[idx] = 0

    # -- integration -------------------------------------------------------
    # -- dopamine-gated plasticity (stage 7) -------------------------------
    def enable_plasticity(self, pre_idx, post_idx, mod_idx, mod_to_post):
        """Make every existing synapse from ``pre_idx`` onto ``post_idx`` plastic.

        Used for Kenyon cell -> MBON synapses. Each brain (batch column) has its
        own multiplicative scale per plastic synapse, starting at 1 (the
        published strength). Which synapses exist is never changed.

        The rule (applied by ``learn``) is the standard mushroom-body rule:
        a synapse is weakened when its presynaptic Kenyon cell was active
        shortly before a dopamine neuron that innervates that MBON fired, and
        it slowly recovers towards its published strength otherwise.

        mod_idx      dopamine neuron indices
        mod_to_post  (len(post_idx), len(mod_idx)) non-negative weights: how
                     much each dopamine neuron acts on each MBON (from the wiring)
        """
        t = self.torch
        pre_idx = np.asarray(pre_idx, np.int64)
        post_idx = np.asarray(post_idx, np.int64)
        pre_pos = np.full(self.N, -1, np.int64); pre_pos[pre_idx] = np.arange(len(pre_idx))
        post_pos = np.full(self.N, -1, np.int64); post_pos[post_idx] = np.arange(len(post_idx))
        indptr = self.indptr.cpu().numpy()
        rows = self.rows.cpu().numpy().astype(np.int64)
        # presynaptic neuron of every stored synapse
        pre_of = np.repeat(np.arange(self.N), np.diff(indptr))
        m = (pre_pos[pre_of] >= 0) & (post_pos[rows] >= 0)
        syn = np.flatnonzero(m)
        pidx = np.full(len(rows), -1, np.int32)
        pidx[syn] = np.arange(len(syn), dtype=np.int32)
        self.pidx = t.from_numpy(pidx).to(self.device)
        self.p_pre = t.from_numpy(pre_pos[pre_of[syn]]).to(self.device)     # into pre_idx
        self.p_post = t.from_numpy(post_pos[rows[syn]]).to(self.device)     # into post_idx
        self.n_plastic = len(syn)
        self.pscale = t.ones((self.n_plastic, self.B), device=self.device)
        mod_idx = np.asarray(mod_idx, np.int64)
        mod_pos = np.full(self.N, -1, np.int64); mod_pos[mod_idx] = np.arange(len(mod_idx))
        self.pl_M = t.as_tensor(np.asarray(mod_to_post, np.float32), device=self.device)
        assert self.pl_M.shape == (len(post_idx), len(mod_idx))
        self.pl_pre_pos = t.from_numpy(pre_pos).to(self.device)
        self.pl_mod_pos = t.from_numpy(mod_pos).to(self.device)
        self.e_pre = t.zeros((len(pre_idx), self.B), device=self.device)    # Kenyon cell eligibility
        self.e_mod = t.zeros((len(mod_idx), self.B), device=self.device)    # dopamine trace
        self.set_learning(np.zeros(self.B), np.zeros(self.B), np.full(self.B, 20.0), np.full(self.B, 20.0))

    def set_learning(self, rate, recovery, tau_pre_ms, tau_mod_ms):
        """Per-brain learning parameters (arrays of length B).

        rate        weakening per unit (eligibility x dopamine) per ms
        recovery    fraction of the gap to 1 recovered per second
        tau_*_ms    decay times of the Kenyon cell and dopamine traces
        """
        t = self.torch
        f = lambda x: t.as_tensor(np.asarray(x, np.float32), device=self.device)
        dt = self.p.dt
        self.l_rate = f(rate) * dt
        self.l_rec = f(recovery) * dt * 1e-3
        self.l_dpre = t.exp(-dt / f(tau_pre_ms))
        self.l_dmod = t.exp(-dt / f(tau_mod_ms))

    def reset_plasticity(self, mask):
        """Back to the published synapse strengths for brains where ``mask`` is True."""
        if getattr(self, "pscale", None) is None:
            return
        m = self.torch.as_tensor(np.asarray(mask), device=self.device, dtype=self.torch.bool)
        self.pscale[:, m] = 1.0
        self.e_pre[:, m] = 0
        self.e_mod[:, m] = 0

    def _learn(self, pre, b):
        """Update traces with this step's spikes, then the plastic synapse scales."""
        t = self.torch
        self.e_pre.mul_(self.l_dpre)
        self.e_mod.mul_(self.l_dmod)
        if len(pre):
            kp = self.pl_pre_pos[pre]
            m = kp >= 0
            if bool(m.any()):
                self.e_pre.index_put_((kp[m], b[m]), t.ones(int(m.sum()), device=self.device), accumulate=True)
            dp = self.pl_mod_pos[pre]
            m = dp >= 0
            if bool(m.any()):
                self.e_mod.index_put_((dp[m], b[m]), t.ones(int(m.sum()), device=self.device), accumulate=True)
        da = self.pl_M @ self.e_mod                          # dopamine reaching each MBON (n_post, B)
        dw = self.e_pre[self.p_pre] * da[self.p_post]        # coincidence per plastic synapse
        self.pscale.add_(dw * self.l_rate, alpha=-1.0)
        self.pscale.add_((1.0 - self.pscale) * self.l_rec)
        self.pscale.clamp_(0.0, 1.0)

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
        bb = b[ev]
        flat = self.rows[syn].long() * self.B + bb
        vals = self.vals[syn]
        if getattr(self, "pscale", None) is not None:
            pi = self.pidx[syn]
            m = pi >= 0
            if bool(m.any()):
                vals = vals.clone()
                vals[m] = vals[m] * self.pscale[pi[m].long(), bb[m]]
        self.g.view(-1).index_add_(0, flat, vals)

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
        if self.adapt_mv:
            self.a.mul_(self.adapt_decay)
            nz = (self.u - self.a > (p.v_th - p.v_0)).nonzero()
        else:
            nz = (self.u > (p.v_th - p.v_0)).nonzero()
        pre, b = nz[:, 0], nz[:, 1]
        if self.adapt_mv and len(pre):
            self.a.view(-1).index_add_(0, pre * self.B + b,
                                       self.torch.full((len(pre),), self.adapt_mv, device=self.device))
        if self.pscale is not None:
            self._learn(pre, b)

        # 3) synaptic input from spikes emitted delay_steps ago, then Poisson input
        slot = self.t_step % self.delay_steps
        ev = self.ring[slot]
        if ev is not None:
            self._deliver(*ev)
        self.ring[slot] = (pre, b) if len(pre) else None
        for idx, p_spike, regular in self._inputs:
            if regular:
                ph = self.phase[idx] + p_spike
                hits = ph >= 1.0
                self.phase[idx] = ph - hits.to(self.dtype)
            else:
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
