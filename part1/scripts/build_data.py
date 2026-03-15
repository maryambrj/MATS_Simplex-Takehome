"""Script to pre-generate training and validation dataset splits."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# Also add part1 explicitly just like train.py does
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from configs.experiment import EXPERIMENT_SPEC
from part1.src.dataset import Mess3MixtureDatasetBuilder

def main():
    print("Initializing dataset builder...")
    builder = Mess3MixtureDatasetBuilder(EXPERIMENT_SPEC.dataset)

    print("Generating train split (20,000 sequences)...")
    train_records = builder.build_split(num_sequences=20000, seed=42)

    print("Generating val split (2,000 sequences)...")
    val_records = builder.build_split(num_sequences=2000, seed=123)

    # Confirm counts of components in the training split
    counts = {0: 0, 1: 0, 2: 0}
    for rec in train_records:
        counts[rec["component_index"]] += 1

    print("\nComponent distribution in TRAIN split:")
    print(f"  C0: {counts[0]} ({counts[0]/20000:.1%})")
    print(f"  C1: {counts[1]} ({counts[1]/20000:.1%})")
    print(f"  C2: {counts[2]} ({counts[2]/20000:.1%})")

    # Save to disk
    # We guarantee we are in part1/scripts, so we resolve relative to here
    artifacts_dir = Path(__file__).resolve().parent.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_path = artifacts_dir / "train_records.pt"
    val_path = artifacts_dir / "val_records.pt"

    print(f"\nSaving to {train_path}...")
    torch.save(train_records, train_path)

    print(f"Saving to {val_path}...")
    torch.save(val_records, val_path)
    print("Done!")

if __name__ == "__main__":
    main()
