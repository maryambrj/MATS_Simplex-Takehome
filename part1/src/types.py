"""Typed dataclasses for the Mess3 non-ergodic mixture experiment specification."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Mess3ComponentSpec:
    """Specification for a single Mess3 process component.

    A Mess3 process is a future generator type parameterised by alpha and x.
    At this stage (Step 1) it is only a named config object.
    """

    name: str
    alpha: float
    x: float


@dataclass(frozen=True)
class DatasetSpec:
    """Specification for the dataset generated from a mixture of Mess3 components.

    Each sequence is drawn entirely from one component (chosen at the start).
    """

    generator_type: str
    components: List[Mess3ComponentSpec]
    mixture_weights: List[float]
    token_values: List[int]
    bos_token: int
    vocab_size: int
    sequence_length: int
    model_input_length: int


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a small decoder-only transformer (config only)."""

    architecture: str
    n_layers: int
    d_model: int
    n_heads: int
    d_mlp: int


@dataclass(frozen=True)
class TrainingSpec:
    """Specification for the training procedure (config only)."""

    objective: str
    optimizer: str
    learning_rate: float


@dataclass(frozen=True)
class ExperimentSpec:
    """Top-level specification that fully describes one fixed experiment."""

    name: str
    dataset: DatasetSpec
    model: ModelSpec
    training: TrainingSpec
