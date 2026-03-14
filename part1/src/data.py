"""PyTorch Dataset wrapper for Mess3 mixture data."""

import torch
from torch.utils.data import Dataset


class Mess3Dataset(Dataset):
    """PyTorch Dataset wrapping the sequence records from Mess3MixtureDatasetBuilder.
    
    Returns standard tensors for training a causal language model.
    """

    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        
        # The full sequence is tokens_with_bos, length 17
        seq = torch.tensor(record["tokens_with_bos"], dtype=torch.long)
        
        # For next-token prediction:
        # Input to the model is the first 16 tokens (BOS + 15 generated)
        # Target for the model is the last 16 tokens (16 generated)
        x = seq[:-1]
        y = seq[1:]
        
        # Optional: return other metadata if analysis scripts need it later
        # For basic training, just x and y are strictly needed.
        return {
            "x": x,
            "y": y,
            "component_index": torch.tensor(record["component_index"], dtype=torch.long),
            "predictive_vectors": torch.tensor(record["predictive_vectors"], dtype=torch.float32),
        }
