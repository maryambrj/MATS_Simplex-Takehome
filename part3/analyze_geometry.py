import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def run_geometry_analysis():
    # Setup paths
    part3_dir = Path(__file__).resolve().parent
    data_path = part3_dir / "artifacts" / "dataset_with_activations.pt"
    out_dir = part3_dir / "results"
    
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        
    print(f"Loading dataset from {data_path}...")
    ds = torch.load(data_path, map_location='cpu', weights_only=False)
    X = ds["X_final"]  # (N, 64)
    N = X.shape[0]
    
    sum_txt = []
    
    # -----------------------------------------------------------------
    # Analysis A: PCA Effective Dimensionality
    # -----------------------------------------------------------------
    X_centered = X - X.mean(axis=0)
    pca = PCA().fit(X_centered)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    
    n_90 = np.argmax(cum_var >= 0.90) + 1
    n_95 = np.argmax(cum_var >= 0.95) + 1
    
    sum_txt.append(f"Final-layer PCA dims for 90% variance: {n_90}")
    sum_txt.append(f"Final-layer PCA dims for 95% variance: {n_95}")
    
    # 1. Output file: pca_cumulative_variance.png
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(cum_var) + 1), cum_var, color="black", linewidth=2)
    plt.axhline(0.90, color="gray", linestyle="--", alpha=0.5, label="90%")
    plt.axhline(0.95, color="gray", linestyle="-.", alpha=0.5, label="95%")
    plt.title("Cumulative Explained Variance (Final Residual Stream)")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance")
    plt.xlim(1, 40)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / "pca_cumulative_variance.png", dpi=150)
    plt.close()
    
    # -----------------------------------------------------------------
    # Analysis B: Linear Regression
    # -----------------------------------------------------------------
    targets = {
        "Posterior_P(C)": ds["Y_comp_post"],
        "Belief_C0": ds["Y_belief_c0"],
        "Belief_C1": ds["Y_belief_c1"],
        "Belief_C2": ds["Y_belief_c2"],
    }
    
    r2_scores = {}
    sum_txt.append("\nR^2 global test scores:")
    
    for name, Y in targets.items():
        X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=42)
        reg = LinearRegression().fit(X_tr, Y_tr)
        score = reg.score(X_te, Y_te)
        r2_scores[name] = score
        
        sum_txt.append(f"R^2 for {name} probe: {score:.4f}")
        
    # 3. Output file: probe_r2_table.csv
    df = pd.DataFrame(list(r2_scores.items()), columns=["Target", "Test_R2"])
    df.to_csv(out_dir / "probe_r2_table.csv", index=False)
    
    # 4. Output file: probe_r2_barplot.png
    plt.figure(figsize=(8, 5))
    colors = ['skyblue'] + ['lightgreen'] * 3
    plt.bar(list(r2_scores.keys()), list(r2_scores.values()), color=colors)
    plt.title("Linear Decodability (Test R^2) of Belief States")
    plt.ylabel("R^2 Score")
    plt.ylim(0, 1.05)
    for i, v in enumerate(r2_scores.values()):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
    plt.savefig(out_dir / "probe_r2_barplot.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------
    # Analysis C: PCA Scatter Plot Colored by True Component
    # -----------------------------------------------------------------
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X)
    
    # 2. Output file: activation_scatter_by_component.png
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=ds["y_component_id"], cmap='Set1', s=5, alpha=0.5)
    
    from matplotlib.lines import Line2D
    colors_cmap = plt.cm.Set1(np.linspace(0, 1, 9))
    h = [Line2D([0], [0], marker='o', color='w', label=f'C{i}', markerfacecolor=colors_cmap[i], markersize=8) for i in range(3)]
    plt.legend(handles=h)
    plt.title("PCA Scatter: Final Layer Activations Colored by True Component")
    plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
    plt.savefig(out_dir / "activation_scatter_by_component.png", dpi=150)
    plt.close()

    # 2b. Output file: activation_scatter_c0_by_belief.png (Within-component subset)
    mask_c0 = (ds["y_component_id"] == 0)
    X_c0 = X[mask_c0]
    pca_c0 = PCA(n_components=2)
    X_c0_2d = pca_c0.fit_transform(X_c0)
    
    belief_c0_s0 = ds["Y_belief_c0"][mask_c0][:, 0]  # P(S_t=0 | C=0)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_c0_2d[:, 0], X_c0_2d[:, 1], c=belief_c0_s0, cmap='viridis', s=5, alpha=0.6)
    plt.colorbar(scatter, label="P(S_t=0 | C=0, prefix)")
    plt.title("PCA of C0-only Activations (Colored by State Belief)")
    plt.xlabel(f"PC1 ({pca_c0.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca_c0.explained_variance_ratio_[1]*100:.1f}%)")
    plt.savefig(out_dir / "activation_scatter_c0_by_belief.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------
    # Analysis D: Context-Position Dependence R^2 Plot
    # -----------------------------------------------------------------
    pos = ds["positions"]  # (N,)
    unique_pos = np.sort(np.unique(pos))
    
    pos_r2s = {name: [] for name in targets.keys()}
    
    for t in unique_pos:
        mask = (pos == t)
        X_t = X[mask]
        
        for name, Y in targets.items():
            Y_t = Y[mask]
            
            # small subset, run train/test to prevent overfitting, 
            # or just test on same since n_samples=2000, n_feats=64.
            # Using 80/20 train/test split.
            X_tr, X_te, Y_tr, Y_te = train_test_split(X_t, Y_t, test_size=0.2, random_state=42)
            reg = LinearRegression().fit(X_tr, Y_tr)
            score = reg.score(X_te, Y_te)
            pos_r2s[name].append(max(0, score)) # clamp negatives for plotting
            
    # 5. Output file: position_probe_r2.png
    plt.figure(figsize=(8, 5))
    plt.plot(unique_pos, pos_r2s["Posterior_P(C)"], label="Posterior P(C|prefix)", color="blue", marker="o")
    
    # We can plot the average belief score for brevity, or all 3.
    avg_belief = np.mean([pos_r2s["Belief_C0"], pos_r2s["Belief_C1"], pos_r2s["Belief_C2"]], axis=0)
    plt.plot(unique_pos, avg_belief, label="Avg Belief P(S_t|C,prefix)", color="green", marker="s")
    
    plt.title("Context-Position Dependence (Linear Probe R^2 vs Prefix Length)")
    plt.xlabel("Prefix Length (t)")
    plt.ylabel("Test R^2 Score")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out_dir / "position_probe_r2.png", dpi=150)
    plt.close()

    # 6. Output file: summary_metrics.txt
    txt_content = "\n".join(sum_txt)
    with open(out_dir / "summary_metrics.txt", "w") as f:
        f.write(txt_content)
        
    print(txt_content)
    print("\nAll deliverables generated in", out_dir)

if __name__ == "__main__":
    run_geometry_analysis()
