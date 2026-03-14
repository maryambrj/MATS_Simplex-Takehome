"""Unit tests for the frozen experiment specification (Step 1)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy
import dataclasses
import pytest

from configs.experiment import build_experiment_spec
from src.types import ExperimentSpec, Mess3ComponentSpec
from src.validate_config import validate_experiment_spec


# ── helpers ──────────────────────────────────────────────────────────

def _spec():
    """Shortcut: return a fresh default spec."""
    return build_experiment_spec()


def _replace_component(spec, index, **overrides):
    """Return a new ExperimentSpec with one component field changed."""
    components = list(spec.dataset.components)
    components[index] = dataclasses.replace(components[index], **overrides)
    new_dataset = dataclasses.replace(spec.dataset, components=components)
    return dataclasses.replace(spec, dataset=new_dataset)


# ── Test 1: default config builds ────────────────────────────────────

def test_default_config_builds():
    spec = _spec()
    assert isinstance(spec, ExperimentSpec)


# ── Test 2: default config validates ─────────────────────────────────

def test_default_config_validates():
    spec = _spec()
    validate_experiment_spec(spec)  # should not raise


# ── Test 3: mixture weights sum to 1 ────────────────────────────────

def test_mixture_weights_sum_to_one():
    spec = _spec()
    assert abs(sum(spec.dataset.mixture_weights) - 1.0) < 1e-9


# ── Test 4: exactly 3 components ────────────────────────────────────

def test_exactly_three_components():
    spec = _spec()
    assert len(spec.dataset.components) == 3


# ── Test 5: expected component values ────────────────────────────────

def test_expected_component_values():
    spec = _spec()
    c0, c1, c2 = spec.dataset.components

    assert c0.name == "C0" and c0.alpha == 0.60 and c0.x == 0.15
    assert c1.name == "C1" and c1.alpha == 0.75 and c1.x == 0.10
    assert c2.name == "C2" and c2.alpha == 0.60 and c2.x == 0.35


# ── Test 6: invalid alpha fails ─────────────────────────────────────

def test_invalid_alpha_raises():
    spec = _replace_component(_spec(), 0, alpha=1.0)
    with pytest.raises(ValueError, match="invalid alpha"):
        validate_experiment_spec(spec)


# ── Test 7: invalid x fails ─────────────────────────────────────────

def test_invalid_x_raises():
    spec = _replace_component(_spec(), 0, x=0.5)
    with pytest.raises(ValueError, match="invalid x"):
        validate_experiment_spec(spec)


# ── Test 8: invalid weight sum fails ────────────────────────────────

def test_invalid_weight_sum_raises():
    spec = _spec()
    new_dataset = dataclasses.replace(
        spec.dataset, mixture_weights=[0.5, 0.5, 0.5]
    )
    bad_spec = dataclasses.replace(spec, dataset=new_dataset)
    with pytest.raises(ValueError, match="sum to 1"):
        validate_experiment_spec(bad_spec)


# ── Test 9: invalid bos token fails ─────────────────────────────────

def test_invalid_bos_token_raises():
    spec = _spec()
    new_dataset = dataclasses.replace(spec.dataset, bos_token=99)
    bad_spec = dataclasses.replace(spec, dataset=new_dataset)
    with pytest.raises(ValueError, match="bos_token must be 3"):
        validate_experiment_spec(bad_spec)


# ── Test 10: invalid head divisibility fails ─────────────────────────

def test_invalid_head_divisibility_raises():
    spec = _spec()
    new_model = dataclasses.replace(spec.model, d_model=65)
    bad_spec = dataclasses.replace(spec, model=new_model)
    with pytest.raises(ValueError, match="divisible by n_heads"):
        validate_experiment_spec(bad_spec)
