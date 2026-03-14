"""Build the one fixed experiment specification for the Mess3 non-ergodic mixture baseline."""

from src.types import (
    DatasetSpec,
    ExperimentSpec,
    Mess3ComponentSpec,
    ModelSpec,
    TrainingSpec,
)


def build_experiment_spec() -> ExperimentSpec:
    """Return the complete, frozen specification for the baseline experiment.

    All values are hard-coded.  This is intentionally not configurable.
    """

    dataset = DatasetSpec(
        generator_type="mess3_mixture",
        components=[
            Mess3ComponentSpec(name="C0", alpha=0.60, x=0.15),
            Mess3ComponentSpec(name="C1", alpha=0.75, x=0.10),
            Mess3ComponentSpec(name="C2", alpha=0.60, x=0.35),
        ],
        mixture_weights=[1 / 3, 1 / 3, 1 / 3],
        token_values=[0, 1, 2],
        bos_token=3,
        vocab_size=4,
        sequence_length=16,
        model_input_length=17,
    )

    model = ModelSpec(
        architecture="decoder_only_transformer",
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_mlp=256,
    )

    training = TrainingSpec(
        objective="next_token_prediction",
        optimizer="adam",
        learning_rate=5e-4,
    )

    return ExperimentSpec(
        name="mess3_nonergodic_mixture_baseline",
        dataset=dataset,
        model=model,
        training=training,
    )


# Convenience: importable singleton so callers can do
#   from configs.experiment import EXPERIMENT_SPEC
EXPERIMENT_SPEC = build_experiment_spec()
