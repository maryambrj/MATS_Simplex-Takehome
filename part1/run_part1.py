"""End-to-end runner for Part 1."""

import os
import subprocess
import sys
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt

def main():
    root_dir = Path(__file__).resolve().parent

    # 1. Build Data
    artifacts_dir = root_dir / "artifacts"
    train_path = artifacts_dir / "train_records.pt"
    val_path = artifacts_dir / "val_records.pt"
    
    if not train_path.exists() or not val_path.exists():
        print(f"Dataset splits not found. Building data...")
        build_script = root_dir / "scripts" / "build_data.py"
        result = subprocess.run([sys.executable, str(build_script)])
        if result.returncode != 0:
            print("Failed to build data. Exiting.")
            sys.exit(result.returncode)
    else:
        print(f"Dataset splits already exist in {artifacts_dir}. Skipping generation.")

    # 2. Train Model
    print("\nStarting training...")
    train_script = root_dir / "train.py"
    result = subprocess.run([sys.executable, str(train_script)])
    if result.returncode != 0:
        print("Training failed. Exiting.")
        sys.exit(result.returncode)
        
    print("\nPart 1 End-to-End finished successfully!")

    # 3. Generate Evaluation Plots & Print Examples
    print("\nGenerating evaluation plots and examples...")
    
    # Plot 1: Training and validation loss vs epoch
    log_path = artifacts_dir / "train_log.json"
    if log_path.exists():
        with open(log_path, "r") as f:
            log_data = json.load(f)
            
        epochs = [entry["epoch"] for entry in log_data]
        train_loss = [entry["train_loss"] for entry in log_data]
        val_loss = [entry["val_loss"] for entry in log_data]
        
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label="Train Loss", marker='o')
        plt.plot(epochs, val_loss, label="Validation Loss", marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Cross Entropy)")
        plt.title("Training and Validation Loss vs Epoch")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_loss_path = artifacts_dir / "plot_loss.png"
        plt.savefig(plot_loss_path)
        plt.close()
        print(f"Saved: {plot_loss_path}")

    # Plot 2: Bar chart of component counts & Example sequences
    if train_path.exists():
        train_records = torch.load(train_path, weights_only=False)
        
        counts = {0: 0, 1: 0, 2: 0}
        for rec in train_records:
            counts[rec["component_index"]] += 1
            
        components = ["C0", "C1", "C2"]
        count_values = [counts[0], counts[1], counts[2]]
        
        plt.figure(figsize=(6, 5))
        bars = plt.bar(components, count_values, color=['skyblue', 'lightgreen', 'salmon'])
        plt.xlabel("Component")
        plt.ylabel("Number of Sequences")
        plt.title("Component Counts in Training Split")
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 50, int(yval), ha='center', va='bottom')
            
        plt.tight_layout()
        plot_comp_path = artifacts_dir / "plot_components.png"
        plt.savefig(plot_comp_path)
        plt.close()
        print(f"Saved: {plot_comp_path}")
        
        print("\n--- Example Sequences ---")
        examples_found = set()
        for rec in train_records:
            comp_idx = rec["component_index"]
            if comp_idx not in examples_found:
                examples_found.add(comp_idx)
                tokens = rec["tokens"]
                tokens_str = " ".join(str(t) for t in tokens)
                print(f"C{comp_idx}\t{tokens_str}")
            
            if len(examples_found) == 3:
                break

if __name__ == "__main__":
    main()
