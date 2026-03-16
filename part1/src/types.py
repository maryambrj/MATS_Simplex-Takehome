from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Mess3ComponentSpec:

    name: str
    alpha: float
    x: float


@dataclass(frozen=True)
class DatasetSpec:

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

    architecture: str
    n_layers: int
    d_model: int
    n_heads: int
    d_mlp: int


@dataclass(frozen=True)
class TrainingSpec:

    objective: str
    optimizer: str
    learning_rate: float


@dataclass(frozen=True)
class ExperimentSpec:

    name: str
    dataset: DatasetSpec
    model: ModelSpec
    training: TrainingSpec
