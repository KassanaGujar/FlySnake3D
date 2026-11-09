"""Leaky integrate-and-fire dynamics of the whole fly brain.

The model is the one published by Shiu et al. (2024, Nature 634:210) and
implemented in Brian2 in philshiu/Drosophila_brain_model/model.py. We keep it
*unchanged* and provide two interchangeable backends:

* ``Brian2Reference`` - calls Brian2 with the published equations. Slow, but
  it is the ground truth. Used for validation only.
* ``TorchLIF``        - the same equations re-implemented with sparse matrices
  in PyTorch so that (a) it runs on a GPU and (b) many candidate brains can be
  simulated *in parallel as a batch* - which is what evolution needs.

Neuron equations (per neuron, all neurons identical):

    dv/dt = (v_0 - v + g) / t_mbr        membrane potential
    dg/dt = -g / tau                     synaptic conductance-like current
    v > v_th  ->  spike;  v = v_rst, g = 0, refractory for t_rfc

A presynaptic spike from neuron i adds  w_syn * W[j, i]  to g_j after t_dly,
where W[j, i] is the signed synapse count. External stimulation is a Poisson
spike train that adds  w_syn * f_poi  directly to v (Shiu's "optogenetic"
activation).

Because the equations are linear between spikes we integrate them *exactly*
(this is also what Brian2's ``method='linear'`` does):

    g(t+dt) = g * e^{-dt/tau}
    u = v - v_0,  A = g * tau / (tau - t_mbr)
    u(t+dt) = (u - A) * e^{-dt/t_mbr} + A * e^{-dt/tau}
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Parameters - copied verbatim from Shiu et al. model.py (default_params).
# Units: millivolts and milliseconds.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LIFParams:
    v_0: float = -52.0     # resting potential (mV)          Kakaria & de Bivort 2017
    v_rst: float = -52.0   # reset potential (mV)
    v_th: float = -45.0    # spike threshold (mV)
    t_mbr: float = 20.0    # membrane time constant (ms)
    tau: float = 5.0       # synaptic time constant (ms)     Jürgensen et al. 2021
    t_rfc: float = 2.2     # refractory period (ms)          Lazar et al. 2021
    t_dly: float = 1.8     # synaptic delay (ms)             Paul et al. 2015
    w_syn: float = 0.275   # mV per synapse (free parameter, fit by Shiu et al.)
    f_poi: float = 250.0   # Poisson input weight multiplier (w_syn * f_poi per input spike)
    dt: float = 0.1        # integration step (ms) - Brian2 default

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# PyTorch backend
# ---------------------------------------------------------------------------


class TorchLIF:
    """Batched whole-brain LIF simulator.

    Shapes: N neurons, B independent brains ("batch"). All B brains share the
    same wiring W, but each can receive different stimuli and (later) carry
    different evolved per-neuron parameters such as thresholds.

    Typical use::

        sim = TorchLIF(conn.W, batch=4)
        sim.set_poisson_input(indices, rates_hz)   # rates: (len(indices),) or (len(indices), B)
        counts = sim.run(n_steps=1000, record=readout_idx)   # 100 ms
    """

    def __init__(self, W: sp.csr_matrix, params: LIFParams = LIFParams(), batch: int = 1,
                 device: str | None = None, seed: int = 0, dtype=None):
        import torch

        self.torch = torch
        self.p = params
        self.N = W.shape[0]
        self.B = batch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        # Synaptic matrix in mV: W_mv[post, pre] = w_syn * signed synapse count.
        Wc = W.tocsr().astype(np.float32)
        self.W = torch.sparse_csr_tensor(
            torch.from_numpy(Wc.indptr.astype(np.int64)),
            torch.from_numpy(Wc.indices.astype(np.int64)),
            torch.from_numpy(Wc.data * params.w_syn),
            size=Wc.shape, device=self.device, dtype=self.dtype,
        )

        p = params
        self.decay_g = float(np.exp(-p.dt / p.tau))
        self.decay_v = float(np.exp(-p.dt / p.t_mbr))
        self.A_coef = p.tau / (p.tau - p.t_mbr)
        self.delay_steps = int(round(p.t_dly / p.dt))
        self.rfc_steps = int(round(p.t_rfc / p.dt))
        # Per-neuron threshold (an evolvable parameter later). Shape (N, B).
        self.v_th = torch.full((self.N, self.B), p.v_th, device=self.device, dtype=self.dtype)
        self.rfc_steps_per_neuron = torch.full((self.N, 1), self.rfc_steps, device=self.device,
                                               dtype=torch.int16)
        self._inputs: list[tuple] = []
        self.reset()

    # -- state -------------------------------------------------------------
    def reset(self):
        t = self.torch
        self.v = t.full((self.N, self.B), self.p.v_0, device=self.device, dtype=self.dtype)
        self.g = t.zeros((self.N, self.B), device=self.device, dtype=self.dtype)
        self.ref = t.zeros((self.N, self.B), device=self.device, dtype=t.int16)
        # Ring buffer of past spikes for the synaptic delay.
        self.buf = t.zeros((self.delay_steps, self.N, self.B), device=self.device, dtype=self.dtype)
        self.t_step = 0

    # -- stimulation -----------------------------------------------------
    def clear_inputs(self):
        self._inputs = []
        self.rfc_steps_per_neuron.fill_(self.rfc_steps)

    def set_poisson_input(self, indices, rates_hz):
        """Drive neurons ``indices`` with Poisson spikes at ``rates_hz``.

        ``rates_hz`` is a scalar, a (k,) array (same for every brain) or a
        (k, B) array (one rate per brain). Like Shiu et al., stimulated
        neurons get no refractory period. Replaces any previous input on the
        same call list; call ``clear_inputs`` to remove all.
        """
        t = self.torch
        idx = t.as_tensor(np.asarray(indices), device=self.device, dtype=t.long)
        r = t.as_tensor(np.asarray(rates_hz, dtype=np.float32), device=self.device)
        if r.ndim == 0:
            r = r.expand(len(idx))
        if r.ndim == 1:
            r = r[:, None].expand(len(idx), self.B)
        p_spike = (r * self.p.dt * 1e-3).clamp(0, 1).contiguous()
        self._inputs.append((idx, p_spike))
        self.rfc_steps_per_neuron[idx] = 0

    # -- integration -------------------------------------------------------
    def step(self):
        t = self.torch
        p = self.p
        # 1) exact sub-threshold update, frozen while refractory
        free = self.ref <= 0
        A = self.g * self.A_coef
        u = self.v - p.v_0
        v_new = p.v_0 + (u - A) * self.decay_v + A * self.decay_g
        g_new = self.g * self.decay_g
        self.v = t.where(free, v_new, self.v)
        self.g = t.where(free, g_new, self.g)
        self.ref = (self.ref - 1).clamp_(min=0)

        # 2) threshold
        spikes = self.v > self.v_th

        # 3) synaptic input from spikes emitted delay_steps ago
        slot = self.t_step % self.delay_steps
        delayed = self.buf[slot]
        if bool(delayed.any()):
            self.g = self.g + t.sparse.mm(self.W, delayed)
        self.buf[slot] = spikes.to(self.dtype)
        #    external Poisson input onto v
        for idx, p_spike in self._inputs:
            hits = t.rand(p_spike.shape, generator=self.gen, device=self.device) < p_spike
            self.v[idx] += hits.to(self.dtype) * (p.w_syn * p.f_poi)

        # 4) reset
        self.v = t.where(spikes, t.full_like(self.v, p.v_rst), self.v)
        self.g = t.where(spikes, t.zeros_like(self.g), self.g)
        self.ref = t.where(spikes, self.rfc_steps_per_neuron.expand_as(self.ref), self.ref)
        self.t_step += 1
        return spikes

    def run(self, n_steps: int, record=None):
        """Advance ``n_steps`` and return spike counts.

        ``record``: optional index array; if given, returns counts only for
        those neurons as an (len(record), B) numpy array, else (N, B).
        """
        t = self.torch
        counts = t.zeros((self.N, self.B), device=self.device, dtype=t.int32)
        for _ in range(n_steps):
            counts += self.step()
        if record is not None:
            counts = counts[t.as_tensor(np.asarray(record), device=self.device, dtype=t.long)]
        return counts.cpu().numpy()


# ---------------------------------------------------------------------------
# Brian2 reference backend (ground truth)
# ---------------------------------------------------------------------------


def brian2_reference_counts(W: sp.csr_matrix, stim_idx, rate_hz: float, t_ms: float,
                            params: LIFParams = LIFParams(), seed: int = 0) -> np.ndarray:
    """Run the published Brian2 model once; return per-neuron spike counts (N,).

    Equations and objects are the same as Shiu et al. ``create_model`` +
    ``poi``; only the data-loading code differs (W is passed in directly).
    """
    from brian2 import (NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network,
                        mV, ms, Hz, seed as b2seed, defaultclock, prefs)

    prefs.codegen.target = "numpy"  # no C++ compiler required
    b2seed(seed)
    defaultclock.dt = params.dt * ms
    ns = dict(v_0=params.v_0 * mV, v_rst=params.v_rst * mV, v_th=params.v_th * mV,
              t_mbr=params.t_mbr * ms, tau=params.tau * ms)
    eqs = """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    rfc                            : second
    """
    n = W.shape[0]
    neu = NeuronGroup(n, eqs, method="linear", threshold="v > v_th",
                      reset="v = v_rst; g = 0 * mV", refractory="rfc", namespace=ns)
    neu.v = params.v_0 * mV
    neu.g = 0 * mV
    neu.rfc = params.t_rfc * ms
    coo = W.tocoo()
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=params.t_dly * ms)
    syn.connect(i=coo.col.astype(np.int64), j=coo.row.astype(np.int64))  # pre=col, post=row
    syn.w = coo.data.astype(np.float64) * params.w_syn * mV
    pois = []
    for i in np.asarray(stim_idx):
        pois.append(PoissonInput(target=neu[int(i):int(i) + 1], target_var="v", N=1,
                                 rate=rate_hz * Hz, weight=params.w_syn * params.f_poi * mV))
        neu.rfc[int(i)] = 0 * ms
    mon = SpikeMonitor(neu)
    net = Network(neu, syn, mon, *pois)
    net.run(t_ms * ms)
    return np.asarray(mon.count[:], dtype=np.int64)
