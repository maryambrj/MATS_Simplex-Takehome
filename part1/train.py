"""Training script for learning the Mess3 processes with a tiny Transformer."""

import sys
from pathlib import Path

# Add project root to sys.path so we can import configs and src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.experiment import EXPERIMENT_SPEC
from part1.src.torch_dataset import NextTokenDataset
from part1.src.model import DecoderOnlyTransformer


import json

def evaluate(model, dataloader, device):
    """Tiny evaluation function returning average cross entropy loss."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            x = batch["input_ids"].to(device)
            y = batch["target_ids"].to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            val_loss += loss.item()
    return val_loss / len(dataloader)


def main():
    # 1. Configuration constants from EXPERIMENT_SPEC
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = EXPERIMENT_SPEC.training.learning_rate  # 5e-4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Load Dataset splits
    print("Loading pre-generated dataset splits...")
    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    
    train_records = torch.load(artifacts_dir / "train_records.pt", weights_only=False)
    val_records = torch.load(artifacts_dir / "val_records.pt", weights_only=False)
    
    train_dataset = NextTokenDataset(train_records)
    val_dataset = NextTokenDataset(val_records)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Initialize Model
    print("Initializing DecoderOnlyTransformer...")
    model = DecoderOnlyTransformer(
        vocab_size=EXPERIMENT_SPEC.dataset.vocab_size,            # 4
        max_seq_len=EXPERIMENT_SPEC.dataset.model_input_length,   # 17
        d_model=EXPERIMENT_SPEC.model.d_model,                    # 64
        n_heads=EXPERIMENT_SPEC.model.n_heads,                    # 4
        d_mlp=EXPERIMENT_SPEC.model.d_mlp,                        # 256
        n_layers=EXPERIMENT_SPEC.model.n_layers                   # 2
    ).to(device)
    
    # 4. Initialize Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 5. Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    train_log = []
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        # Train pass
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch in pbar:
            x = batch["input_ids"].to(device)
            y = batch["target_ids"].to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        
        # Eval pass
        val_loss = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        train_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

    print("Training complete. Saving artifacts...")
    
    # 6. Save outputs
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = artifacts_dir / "final_model.pt"
    torch.save(model.state_dict(), model_path)
    
    log_path = artifacts_dir / "train_log.json"
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)
        
    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {log_path}")

if __name__ == "__main__":
    main()
