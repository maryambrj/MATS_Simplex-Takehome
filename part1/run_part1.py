"""End-to-end runner for Part 1."""

import os
import subprocess
import sys
from pathlib import Path

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

if __name__ == "__main__":
    main()
