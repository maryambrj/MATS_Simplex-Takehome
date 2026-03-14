"""Validation logic for ExperimentSpec.

Every check raises ``ValueError`` with a clear, human-readable message
when a constraint is violated.
"""

from src.types import ExperimentSpec


def validate_experiment_spec(spec: ExperimentSpec) -> None:
    """Validate *spec* against all fixed experiment constraints.

    Raises
    ------
    ValueError
        If any constraint is violated.
    """

    # ------------------------------------------------------------------
    # General checks
    # ------------------------------------------------------------------
    if not spec.name:
        raise ValueError("Experiment name must be non-empty")

    # ------------------------------------------------------------------
    # Dataset checks
    # ------------------------------------------------------------------
    ds = spec.dataset

    if ds.generator_type != "mess3_mixture":
        raise ValueError(
            f"generator_type must be 'mess3_mixture', got '{ds.generator_type}'"
        )

    if len(ds.components) != 3:
        raise ValueError(
            f"Expected exactly 3 Mess3 components, got {len(ds.components)}"
        )

    component_names = [c.name for c in ds.components]
    if len(component_names) != len(set(component_names)):
        raise ValueError(
            f"Component names must be unique, got {component_names}"
        )

    if ds.token_values != [0, 1, 2]:
        raise ValueError(
            f"token_values must be [0, 1, 2], got {ds.token_values}"
        )

    if ds.bos_token != 3:
        raise ValueError(f"bos_token must be 3, got {ds.bos_token}")

    if ds.vocab_size != 4:
        raise ValueError(f"vocab_size must be 4, got {ds.vocab_size}")

    if ds.sequence_length != 16:
        raise ValueError(
            f"sequence_length must be 16, got {ds.sequence_length}"
        )

    if ds.model_input_length != 17:
        raise ValueError(
            f"model_input_length must be 17, got {ds.model_input_length}"
        )

    # Mixture-weight checks
    if len(ds.mixture_weights) != 3:
        raise ValueError(
            f"Expected exactly 3 mixture weights, got {len(ds.mixture_weights)}"
        )

    for w in ds.mixture_weights:
        if w < 0:
            raise ValueError(
                f"Mixture weight must be non-negative, got {w}"
            )

    weight_sum = sum(ds.mixture_weights)
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(
            f"Mixture weights must sum to 1, got {weight_sum}"
        )

    # ------------------------------------------------------------------
    # Component parameter checks
    # ------------------------------------------------------------------
    for comp in ds.components:
        if not (0 < comp.alpha < 1):
            raise ValueError(
                f"Component {comp.name} has invalid alpha={comp.alpha}; "
                f"expected 0 < alpha < 1"
            )
        if not (0 < comp.x < 0.5):
            raise ValueError(
                f"Component {comp.name} has invalid x={comp.x}; "
                f"expected 0 < x < 0.5"
            )

    # ------------------------------------------------------------------
    # Model checks
    # ------------------------------------------------------------------
    m = spec.model

    if m.architecture != "decoder_only_transformer":
        raise ValueError(
            f"architecture must be 'decoder_only_transformer', "
            f"got '{m.architecture}'"
        )

    # Structural constraint first (independent of exact fixed values)
    if m.d_model % m.n_heads != 0:
        raise ValueError(
            f"d_model ({m.d_model}) must be divisible by n_heads ({m.n_heads})"
        )

    if m.n_layers != 2:
        raise ValueError(f"n_layers must be 2, got {m.n_layers}")

    if m.d_model != 64:
        raise ValueError(f"d_model must be 64, got {m.d_model}")

    if m.n_heads != 4:
        raise ValueError(f"n_heads must be 4, got {m.n_heads}")

    if m.d_mlp != 256:
        raise ValueError(f"d_mlp must be 256, got {m.d_mlp}")

    # ------------------------------------------------------------------
    # Training checks
    # ------------------------------------------------------------------
    t = spec.training

    if t.objective != "next_token_prediction":
        raise ValueError(
            f"objective must be 'next_token_prediction', got '{t.objective}'"
        )

    if t.optimizer.lower() != "adam":
        raise ValueError(
            f"optimizer (lowercased) must be 'adam', got '{t.optimizer.lower()}'"
        )

    if t.learning_rate <= 0:
        raise ValueError(
            f"learning_rate must be > 0, got {t.learning_rate}"
        )


def validate_default_experiment() -> None:
    """Build the default experiment spec and validate it end-to-end."""
    from configs.experiment import build_experiment_spec

    spec = build_experiment_spec()
    validate_experiment_spec(spec)
