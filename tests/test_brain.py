"""Tests for the connectome loader and LIF simulator.

Small synthetic networks test the *dynamics* exactly; one data test (skipped
if data/ is missing) checks the real connectome loads with the known counts.
"""
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from brain.neural_simulator import LIFParams, TorchLIF

DATA = Path(__file__).resolve().parents[1] / "data"
P = LIFParams()


def tiny(n=2, pre=0, post=1, syn=10.0):
    W = sp.lil_matrix((n, n), dtype=np.float32)
    W[post, pre] = syn
    return W.tocsr()


def test_rest_is_stable():
    sim = TorchLIF(tiny(), P, batch=1, device="cpu")
    counts = sim.run(1000)
    assert counts.sum() == 0
    assert np.allclose(sim.v.numpy(), P.v_0)


def test_exact_integration_matches_closed_form():
    """With g0 injected at rest and no spikes, v follows the analytic solution."""
    sim = TorchLIF(tiny(), P, batch=1, device="cpu")
    g0 = 5.0  # mV, sub-threshold
    sim.g[1, 0] = g0
    n = 37
    for _ in range(n):
        sim.step()
    t = n * P.dt
    A = g0 * P.tau / (P.tau - P.t_mbr)
    v_expected = P.v_0 + (0 - A) * np.exp(-t / P.t_mbr) + A * np.exp(-t / P.tau)
    assert sim.v[1, 0].item() == pytest.approx(v_expected, abs=1e-4)
    assert sim.g[1, 0].item() == pytest.approx(g0 * np.exp(-t / P.tau), abs=1e-5)


def test_synaptic_delay_and_weight():
    """A spike in neuron 0 reaches neuron 1 exactly t_dly later with w_syn*count."""
    syn = 10.0
    sim = TorchLIF(tiny(syn=syn), P, batch=1, device="cpu")
    sim.v[0, 0] = -40.0  # above threshold -> spikes on the first step
    first = sim.step()
    assert bool(first[0, 0]) and not bool(first[1, 0])
    delay = int(round(P.t_dly / P.dt))
    for k in range(1, delay):
        sim.step()
        assert sim.g[1, 0].item() == 0.0, f"input arrived early at step {k}"
    sim.step()
    assert sim.g[1, 0].item() == pytest.approx(syn * P.w_syn, rel=1e-5)


def test_inhibitory_sign_lowers_voltage():
    sim = TorchLIF(tiny(syn=-50.0), P, batch=1, device="cpu")
    sim.v[0, 0] = -40.0
    for _ in range(40):
        sim.step()
    assert sim.v[1, 0].item() < P.v_0


def test_poisson_drive_rate_and_reproducibility():
    """Shiu-style Poisson input (w_syn*f_poi per event) makes the target fire ~at the input rate."""
    def run(seed):
        sim = TorchLIF(tiny(), P, batch=4, device="cpu", seed=seed)
        sim.set_poisson_input([0], 100.0)
        return sim.run(10_000)  # 1 s
    a, b, c = run(1), run(1), run(2)
    assert np.array_equal(a, b), "same seed must give identical spikes"
    assert not np.array_equal(a, c)
    rate = a[0].mean()
    assert 80 < rate < 120, rate  # each Poisson event (68.75 mV) forces a spike


def test_batch_columns_are_independent():
    sim = TorchLIF(tiny(), P, batch=2, device="cpu")
    sim.set_poisson_input([0], np.array([[200.0, 0.0]]))
    counts = sim.run(2000)
    assert counts[0, 0] > 0 and counts[0, 1] == 0


@pytest.mark.skipif(not (DATA / "Connectivity_783.parquet").exists(), reason="run scripts/fetch_data.py")
def test_real_connectome_loads():
    from brain.connectome_loader import load_connectome
    c = load_connectome()
    assert c.n_neurons == 138_639
    assert c.n_edges == 15_091_983
    assert len(c.indices_of_type("DNp01")) == 2          # the giant fibers, one per side
    assert c.synapses_between("LPLC2", "DNp01") > 0      # looming -> escape pathway exists
    with pytest.raises(KeyError):
        c.indices_of_type("NotARealNeuronType")
