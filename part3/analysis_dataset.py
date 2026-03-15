import sys
import torch
import numpy as np
from pathlib import Path

# Add root so we can load part3 and part1 modules
part1_dir = Path(__file__).resolve().parent.parent / "part1"
sys.path.insert(0, str(part1_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mess3 import compute_component_likelihood_and_belief

def build_analysis_dataset(num_sequences=2000):
    """
    Loads val_records.pt, extracts 2000 sequences, and computes all the exact prefix targeting logic 
    listed in Part 5, 6, and 7. Result is a flat dataset where N = 2000 * 17 prefixes.
    """
    part1_dir = Path(__file__).resolve().parent.parent / "part1"
    artifacts_dir = part1_dir / "artifacts"
    
    val_records = torch.load(artifacts_dir / "val_records.pt", map_location='cpu', weights_only=False)
    records = val_records[:num_sequences]
    
    # The exact 3 components from the prompt
    components_params = [
        {"alpha": 0.60, "x": 0.15},  # C0
        {"alpha": 0.79, "x": 0.11},  # C1
        {"alpha": 0.60, "x": 0.49},  # C2
    ]
    
    metadata = {
        "positions": [],
        "y_component_id": [],
        "Y_comp_post": [],
        "Y_belief_c0": [],
        "Y_belief_c1": [],
        "Y_belief_c2": [],
        "Y_belief_all": [],
        "tokens": [] # need to save tokens so we can map to model activations in next script
    }
    
    print(f"Generating tracking dataset for {num_sequences} sequences...")
    
    for seq_idx, record in enumerate(records):
        true_c = record["component_index"]
        seq_tokens = record["tokens"].tolist()  # length 16
        
        # for each prefix length 0 to 16
        for t in range(17):
            prefix = seq_tokens[:t]
            
            p_s_given_c = np.zeros((3, 3), dtype=np.float64)
            likelihoods = np.zeros(3, dtype=np.float64)
            
            for c in range(3):
                params = components_params[c]
                lh, belief = compute_component_likelihood_and_belief(prefix, params["alpha"], params["x"])
                likelihoods[c] = lh
                p_s_given_c[c] = belief
                
            # Part 6: Posterior over components score_c = (1/3) * likelihood_c
            scores = (1/3) * likelihoods
            if scores.sum() > 0:
                p_c = scores / scores.sum()
            else:
                p_c = np.array([1/3, 1/3, 1/3])
                
            y_all = np.concatenate([p_s_given_c[0], p_s_given_c[1], p_s_given_c[2]])
            
            metadata["positions"].append(t)
            metadata["y_component_id"].append(true_c)
            metadata["Y_comp_post"].append(p_c)
            metadata["Y_belief_c0"].append(p_s_given_c[0])
            metadata["Y_belief_c1"].append(p_s_given_c[1])
            metadata["Y_belief_c2"].append(p_s_given_c[2])
            metadata["Y_belief_all"].append(y_all)
            metadata["tokens"].append([3] + prefix)  # Include BOS token 3
            
    # Process into numpy arrays
    for k in metadata:
        metadata[k] = np.array(metadata[k], dtype=object if k == "tokens" else None)
        
    print(f"Constructed analysis dataset yielding {len(metadata['positions'])} prefix records.")
    
    # Save the prepared analysis dataset to disk
    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(metadata, out_dir / "analysis_dataset.pt")
    print(f"Saved dataset to {out_dir / 'analysis_dataset.pt'}")

if __name__ == "__main__":
    build_analysis_dataset()
