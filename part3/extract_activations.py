import sys
import torch
import numpy as np
from pathlib import Path

# Add root and part1 so we can load part1 modules natively
part1_dir = Path(__file__).resolve().parent.parent / "part1"
sys.path.insert(0, str(part1_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import DecoderOnlyTransformer
from configs.experiment import EXPERIMENT_SPEC


def extract_activations():
    """
    Loads the flattened prefix analysis dataset, runs all prefixes to the model,
    and extracts ONLY the final residual-stream activation vector per explicitly requested.
    """
    # Force CPU to avoid massive MPS kernel launch overhead for 34k size-1 forwards
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # 1. Load the Model
    part1_dir = Path(__file__).resolve().parent.parent / "part1"
    artifacts_dir = part1_dir / "artifacts"
    
    spec = EXPERIMENT_SPEC
    model = DecoderOnlyTransformer(
        vocab_size=spec.dataset.vocab_size,
        max_seq_len=spec.dataset.model_input_length,
        d_model=spec.model.d_model,
        n_heads=spec.model.n_heads,
        d_mlp=spec.model.d_mlp,
        n_layers=spec.model.n_layers
    ).to(device)
    
    model_path = artifacts_dir / "final_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Hook to grab the final residual stream (input to LayerNorm)
    final_acts = []
    def hook(module, input, output):
        final_acts.append(input[0].detach().cpu().numpy())
    handle = model.ln_f.register_forward_hook(hook)
    
    # 2. Load the tracking Dataset
    part3_artifacts = Path(__file__).resolve().parent / "artifacts"
    dataset = torch.load(part3_artifacts / "analysis_dataset.pt", map_location='cpu', weights_only=False)
    
    X_final = []
    
    # `dataset["tokens"]` is a list of lists format [[BOS], [BOS, t1], [BOS, t1, t2], ...]
    print("Beginning extraction loop on CPU...")
    with torch.no_grad():
        for i, prefix in enumerate(dataset["tokens"]):
            if i % 5000 == 0:
                print(f"Processed {i}/34000 prefixes...")
            model_input = torch.tensor([prefix], dtype=torch.long).to(device)
            final_acts.clear()
            _ = model(model_input)
            
            # Extract only the activation from the final token position in the prefix
            # act is shape (1, T, 64) -> grab final T-1
            last_token_act = final_acts[0][0, -1, :] 
            X_final.append(last_token_act)
            
    handle.remove()
    
    # Re-save the merged dataset replacing "tokens" with the true X_final matrix
    X_final = np.array(X_final)
    print(f"Extracted X_final activations shape: {X_final.shape}")
    
    # Clean up memory by removing the string paths
    del dataset["tokens"]
    dataset["X_final"] = X_final
    
    torch.save(dataset, part3_artifacts / "dataset_with_activations.pt")
    print(f"Saved dataset with activations to {part3_artifacts / 'dataset_with_activations.pt'}")

if __name__ == "__main__":
    extract_activations()
