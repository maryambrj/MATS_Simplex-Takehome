"""Unit tests for Mess3Process."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "part1"))

import numpy as np
import pytest

from src.mess3 import Mess3Process


# ── Test 1: transition matrices have correct shape ───────────────────
def test_transition_matrices_shape():
    p = Mess3Process(alpha=0.6, x=0.15)
    assert p.T0.shape == (3, 3)
    assert p.T1.shape == (3, 3)
    assert p.T2.shape == (3, 3)
    assert p.T.shape == (3, 3)


# ── Test 2: all entries are nonnegative ──────────────────────────────
def test_matrices_nonnegative():
    p = Mess3Process(alpha=0.6, x=0.15)
    assert (p.T0 >= 0).all()
    assert (p.T1 >= 0).all()
    assert (p.T2 >= 0).all()


# ── Test 3: total transition rows sum to 1 ───────────────────────────
def test_total_transition_rows_sum_to_one():
    p = Mess3Process(alpha=0.6, x=0.15)
    row_sums = p.T.sum(axis=1)
    np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0], atol=1e-9)


# ── Test 4: initial distribution is uniform ──────────────────────────
def test_initial_distribution():
    p = Mess3Process(alpha=0.6, x=0.15)
    np.testing.assert_allclose(p.initial_distribution, [1/3, 1/3, 1/3], atol=1e-9)


# ── Test 5: sample_step returns valid outputs ────────────────────────
def test_sample_step_valid_outputs():
    p = Mess3Process(alpha=0.6, x=0.15)
    rng = np.random.default_rng(42)

    for state in [0, 1, 2]:
        for _ in range(50):
            token, next_state = p.sample_step(state, rng)
            assert token in {0, 1, 2}
            assert next_state in {0, 1, 2}


# ── Test 6: sample_sequence shapes ───────────────────────────────────
def test_sample_sequence_shapes():
    p = Mess3Process(alpha=0.6, x=0.15)
    rng = np.random.default_rng(42)
    seq = p.sample_sequence(16, rng)

    assert seq["tokens"].shape == (16,)
    assert seq["hidden_states"].shape == (17,)
    
    # Values check
    assert np.all(np.isin(seq["tokens"], [0, 1, 2]))
    assert np.all(np.isin(seq["hidden_states"], [0, 1, 2]))


# ── Test 7: predictive vector update preserves normalization ─────────
def test_update_predictive_vector():
    p = Mess3Process(alpha=0.6, x=0.15)
    eta = np.array([0.5, 0.3, 0.2])

    for token in [0, 1, 2]:
        eta_prime = p.update_predictive_vector(eta, token)
        assert eta_prime.shape == (3,)
        assert (eta_prime >= 0).all()
        assert abs(eta_prime.sum() - 1.0) < 1e-9


# ── Test 8: compute_predictive_vectors shape ─────────────────────────
def test_compute_predictive_vectors_shape():
    p = Mess3Process(alpha=0.6, x=0.15)
    rng = np.random.default_rng(42)
    seq = p.sample_sequence(16, rng)
    
    etas = p.compute_predictive_vectors(seq["tokens"])
    
    assert etas.shape == (17, 3)
    np.testing.assert_allclose(etas[0], [1/3, 1/3, 1/3], atol=1e-9)
    for row in etas:
        assert abs(row.sum() - 1.0) < 1e-9


# ── Test 9: invalid alpha fails ──────────────────────────────────────
# Should raise ValueError for boundary values or out-of-bounds
@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        Mess3Process(alpha=alpha, x=0.15)


# ── Test 10: invalid x fails ─────────────────────────────────────────
@pytest.mark.parametrize("x", [0.0, 0.5, -0.1, 0.6])
def test_invalid_x(x):
    with pytest.raises(ValueError, match="x"):
        Mess3Process(alpha=0.6, x=x)
