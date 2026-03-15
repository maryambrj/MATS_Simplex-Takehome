from torch.utils.data import Dataset
import torch

class NextTokenDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        twb = rec["tokens_with_bos"]   # length 17

        x = torch.tensor(twb[:-1], dtype=torch.long)   # length 16
        y = torch.tensor(twb[1:], dtype=torch.long)    # length 16

        return {
            "input_ids": x,
            "target_ids": y,
            "component_index": rec["component_index"],
        }
