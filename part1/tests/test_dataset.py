"""Unit tests for Mess3MixtureDatasetBuilder."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from configs.experiment import EXPERIMENT_SPEC
from src.dataset import Mess3MixtureDatasetBuilder


@pytest.fixture
def builder():
    return Mess3MixtureDatasetBuilder(EXPERIMENT_SPEC.dataset)


# ── Test 1: sample_sequence_record returns required keys ─────────────
def test_sample_sequence_record_keys(builder):
    rng = np.random.default_rng(42)
    record = builder.sample_sequence_record(rng)

    for key in [
        "component_index",
        "component_name",
        "tokens",
        "tokens_with_bos",
        "hidden_states",
        "predictive_vectors"
    ]:
        assert key in record


# ── Test 2: shapes are correct ───────────────────────────────────────
def test_shapes_are_correct(builder):
    rng = np.random.default_rng(42)
    rec = builder.sample_sequence_record(rng)

    assert rec["tokens"].shape == (16,)
    assert rec["tokens_with_bos"].shape == (17,)
    assert rec["hidden_states"].shape == (17,)
    assert rec["predictive_vectors"].shape == (17, 3)


# ── Test 3: BOS added correctly ──────────────────────────────────────
def test_bos_added_correctly(builder):
    rng = np.random.default_rng(42)
    rec = builder.sample_sequence_record(rng)

    tokens = rec["tokens"]
    twb = rec["tokens_with_bos"]

    assert twb[0] == 3  # EXPERIMENT_SPEC defines bos_token as 3
    np.testing.assert_array_equal(twb[1:], tokens)


# ── Test 4: component metadata valid ─────────────────────────────────
def test_component_metadata_valid(builder):
    rng = np.random.default_rng(42)
    for _ in range(10):
        rec = builder.sample_sequence_record(rng)
        idx = rec["component_index"]
        name = rec["component_name"]

        assert idx in {0, 1, 2}
        assert name in {"C0", "C1", "C2"}
        # Ensures index matches the name since they're ordered
        assert name == f"C{idx}"


# ── Test 5: build_split is reproducible ──────────────────────────────
def test_build_split_reproducible(builder):
    split1 = builder.build_split(num_sequences=5, seed=123)
    split2 = builder.build_split(num_sequences=5, seed=123)

    for r1, r2 in zip(split1, split2):
        assert r1["component_index"] == r2["component_index"]
        np.testing.assert_array_equal(r1["tokens"], r2["tokens"])
        np.testing.assert_array_equal(r1["hidden_states"], r2["hidden_states"])
        np.testing.assert_allclose(r1["predictive_vectors"], r2["predictive_vectors"])


# ── Test 6: no within-sequence component switching ───────────────────
def test_no_within_sequence_component_switching(builder):
    # Structural test: verify there's only one component index per record
    rng = np.random.default_rng(42)
    rec = builder.sample_sequence_record(rng)
    
    assert isinstance(rec["component_index"], int)
    assert np.isscalar(rec["component_index"])
    # If the tokens vector size matched index size, it would be an array.
    # We enforce just one single int id.
