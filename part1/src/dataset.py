"""Dataset builder for creating sequences from a mixture of Mess3 processes."""

import numpy as np


class Mess3MixtureDatasetBuilder:

    def __init__(self, dataset_spec):

        self.spec = dataset_spec
        
        self.generator_type = dataset_spec.generator_type
        if self.generator_type != "mess3_mixture":
            raise ValueError(f"Unsupported generator_type: {self.generator_type}")

        if len(dataset_spec.components) != 3:
            raise ValueError(f"Expected exactly 3 components, got {len(dataset_spec.components)}")

        self.mixture_weights = np.array(dataset_spec.mixture_weights, dtype=np.float64)
        if len(self.mixture_weights) != 3:
            raise ValueError(f"Expected 3 mixture weights, got {len(self.mixture_weights)}")

        weight_sum = np.sum(self.mixture_weights)
        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError(f"Mixture weights must sum to 1.0, got {weight_sum}")

        self.sequence_length = dataset_spec.sequence_length
        if self.sequence_length != 16:
            raise ValueError(f"Sequence length must be 16, got {self.sequence_length}")

        self.bos_token = dataset_spec.bos_token
        if self.bos_token != 3:
            raise ValueError(f"BOS token must be 3, got {self.bos_token}")

        from src.mess3 import Mess3Process
        self.components = [
            Mess3Process(alpha=comp.alpha, x=comp.x)
            for comp in dataset_spec.components
        ]

    def sample_component_index(self, rng: np.random.Generator) -> int:
        """Sample which component generates the next sequence."""
        return int(rng.choice(len(self.components), p=self.mixture_weights))

    def sample_sequence_record(self, rng: np.random.Generator) -> dict:
        """Generate exactly one sequence record and its predictive vectors."""
        component_index = self.sample_component_index(rng)
        component_spec = self.spec.components[component_index]
        process = self.components[component_index]

        seq = process.sample_sequence(self.sequence_length, rng)
        tokens = seq["tokens"]                     # shape (16,)
        hidden_states = seq["hidden_states"]       # shape (17,)
        predictive_vectors = process.compute_predictive_vectors(tokens)  # shape (17, 3)

        tokens_with_bos = np.concatenate([
            np.array([self.bos_token], dtype=int),
            tokens
        ])

        assert tokens.shape == (16,)
        assert tokens_with_bos.shape == (17,)
        assert hidden_states.shape == (17,)
        assert predictive_vectors.shape == (17, 3)

        return {
            "component_index": component_index,
            "component_name": component_spec.name,
            "tokens": tokens,
            "tokens_with_bos": tokens_with_bos,
            "hidden_states": hidden_states,
            "predictive_vectors": predictive_vectors,
        }

    def build_split(self, num_sequences: int, seed: int) -> list[dict]:
        rng = np.random.default_rng(seed)
        return [self.sample_sequence_record(rng) for _ in range(num_sequences)]

    def build_train_val_splits(
        self,
        train_size: int,
        val_size: int,
        train_seed: int,
        val_seed: int,
    ) -> dict:
        return {
            "train": self.build_split(train_size, train_seed),
            "val": self.build_split(val_size, val_seed),
        }
