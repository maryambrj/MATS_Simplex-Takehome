"""Training script for learning the Mess3 processes with a tiny Transformer."""

import sys
from pathlib import Path

# Add project root to sys.path so we can import configs and src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.experiment import EXPERIMENT_SPEC
from part1.src.dataset import Mess3MixtureDatasetBuilder
from part1.src.data import Mess3Dataset
from part1.src.model import TinyTransformer, TransformerConfig


def main():
    # 1. Configuration constants from EXPERIMENT_SPEC
    TRAIN_SIZE = 10000
    VAL_SIZE = 1000
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = EXPERIMENT_SPEC.training.learning_rate  # 5e-4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Build Dataset
    print("Generating dataset splits...")
    builder = Mess3MixtureDatasetBuilder(EXPERIMENT_SPEC.dataset)
    splits = builder.build_train_val_splits(
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        train_seed=42,
        val_seed=123
    )
    
    train_dataset = Mess3Dataset(splits["train"])
    val_dataset = Mess3Dataset(splits["val"])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Initialize Model
    print("Initializing tiny transformer...")
    config = TransformerConfig()
    # verify config matches spec exactly
    assert config.vocab_size == EXPERIMENT_SPEC.dataset.vocab_size
    assert config.n_layers == EXPERIMENT_SPEC.model.n_layers
    assert config.d_model == EXPERIMENT_SPEC.model.d_model
    assert config.n_heads == EXPERIMENT_SPEC.model.n_heads
    assert config.d_mlp == EXPERIMENT_SPEC.model.d_mlp
    assert config.max_seq_len == EXPERIMENT_SPEC.dataset.model_input_length
    
    model = TinyTransformer(config).to(device)
    
    # 4. Initialize Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 5. Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        # Train pass
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        
        # Eval pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                logits, loss = model(x, targets=y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()
