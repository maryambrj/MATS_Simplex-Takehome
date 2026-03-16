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
    
    # Hooks to grab residual streams at different layers
    # Layer 0: Input to the first block
    # Layer 1: Input to the second block
    # Layer 2: Input to the final LayerNorm (ln_f)
    activations = {0: [], 1: [], 2: []}
    
    def get_hook(layer_idx):
        def hook(module, input, output):
            activations[layer_idx].append(input[0].detach().cpu().numpy())
        return hook

    handles = [
        model.h[0].register_forward_hook(get_hook(0)),
        model.h[1].register_forward_hook(get_hook(1)),
        model.ln_f.register_forward_hook(get_hook(2))
    ]
    
    # 2. Load the tracking Dataset
    part3_artifacts = Path(__file__).resolve().parent / "artifacts"
    dataset = torch.load(part3_artifacts / "analysis_dataset.pt", map_location='cpu', weights_only=False)
    
    extracted_layers = {0: [], 1: [], 2: []}
    
    # `dataset["tokens"]` is a list of lists format [[BOS], [BOS, t1], [BOS, t1, t2], ...]
    print("Beginning extraction loop on CPU...")
    with torch.no_grad():
        for i, prefix in enumerate(dataset["tokens"]):
            if i % 5000 == 0:
                print(f"Processed {i}/34000 prefixes...")
            model_input = torch.tensor([prefix], dtype=torch.long).to(device)
            
            # Clear previously captured activations
            for k in activations: activations[k].clear()
            
            _ = model(model_input)
            
            # For each layer, extract only the activation from the final token position in the prefix
            for k in activations:
                # act is shape (1, T, 64) -> grab final token position
                last_token_act = activations[k][0][0, -1, :] 
                extracted_layers[k].append(last_token_act)
            
    for h in handles:
        h.remove()
    
    # Re-save the merged dataset
    for k in extracted_layers:
        extracted_layers[k] = np.array(extracted_layers[k])
        print(f"Extracted Layer {k} activations shape: {extracted_layers[k].shape}")
    
    # Clean up memory by removing the string paths
    del dataset["tokens"]
    dataset["X_layer0"] = extracted_layers[0]
    dataset["X_layer1"] = extracted_layers[1]
    dataset["X_layer2"] = extracted_layers[2]
    
    torch.save(dataset, part3_artifacts / "dataset_with_activations.pt")
    print(f"Saved dataset with all layers to {part3_artifacts / 'dataset_with_activations.pt'}")

if __name__ == "__main__":
    extract_activations()
