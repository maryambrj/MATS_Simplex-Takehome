import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
    
    layers = [0, 1, 2]
    targets = {
        "Posterior_P(C)": ds["Y_comp_post"],
        "Belief_C0": ds["Y_belief_c0"],
        "Belief_C1": ds["Y_belief_c1"],
        "Belief_C2": ds["Y_belief_c2"],
    }
    
    pos = ds["positions"]
    unique_pos = np.sort(np.unique(pos))
    
    sum_txt = []
    
    # -----------------------------------------------------------------
    # Comparative Plots Storage
    # -----------------------------------------------------------------
    layer_r2_scores = {l: {} for l in layers}
    layer_pos_r2s = {l: {name: [] for name in targets.keys()} for l in layers}
    layer_cum_var = {}
    
    for l in layers:
        X = ds[f"X_layer{l}"]
        print(f"Analyzing Layer {l}...")
        
        # PCA
        X_centered = X - X.mean(axis=0)
        pca = PCA().fit(X_centered)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        layer_cum_var[l] = cum_var
        
        n_90 = np.argmax(cum_var >= 0.90) + 1
        n_95 = np.argmax(cum_var >= 0.95) + 1
        sum_txt.append(f"Layer {l} PCA dims for 90% variance: {n_90}")
        sum_txt.append(f"Layer {l} PCA dims for 95% variance: {n_95}")
        
        # Regression Global
        sum_txt.append(f"Layer {l} R^2 global test scores:")
        for name, Y in targets.items():
            X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=42)
            reg = LinearRegression().fit(X_tr, Y_tr)
            score = reg.score(X_te, Y_te)
            layer_r2_scores[l][name] = score
            sum_txt.append(f"  R^2 for {name} probe: {score:.4f}")
            
        # Regression Position-Dependent
        for t in unique_pos:
            mask = (pos == t)
            X_t = X[mask]
            for name, Y in targets.items():
                Y_t = Y[mask]
                X_tr, X_te, Y_tr, Y_te = train_test_split(X_t, Y_t, test_size=0.2, random_state=42)
                reg = LinearRegression().fit(X_tr, Y_tr)
                score = reg.score(X_te, Y_te)
                layer_pos_r2s[l][name].append(max(0, score))
        
    # -----------------------------------------------------------------
    # Multi-Layer Comparative Plots
    # -----------------------------------------------------------------
    
    # NEW: Multi-layer PCA Scatter (3 panels) - 3D
    fig = plt.figure(figsize=(18, 6))
    comp_colors = ['#e41a1c', '#377eb8', '#4daf4a'] # Red, Blue, Green
    cmap_comp = ListedColormap(comp_colors)
    
    for i, l in enumerate(layers):
        X = ds[f"X_layer{l}"]
        pca_3d = PCA(n_components=3)
        X_3d = pca_3d.fit_transform(X)
        
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=ds["y_component_id"], cmap=cmap_comp, s=2, alpha=0.4)
        ax.set_title(f"Layer {l} 3D PCA\n(EV: {pca_3d.explained_variance_ratio_.sum()*100:.1f}%)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
    
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], marker='o', color='w', label=f'C{i}', markerfacecolor=comp_colors[i], markersize=8) for i in range(3)]
    fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.05))
    plt.tight_layout()
    plt.savefig(out_dir / "activation_scatter_by_component.png", dpi=150, bbox_inches='tight')
    plt.close()

    # NEW: Multi-layer Within-Component Scatter (3 panels) - 3D
    fig = plt.figure(figsize=(18, 6))
    mask_c0 = (ds["y_component_id"] == 0)
    belief_c0_s0 = ds["Y_belief_c0"][mask_c0][:, 0]  # P(S_t=0 | C=0)
    
    for i, l in enumerate(layers):
        X = ds[f"X_layer{l}"]
        X_c0 = X[mask_c0]
        pca_c0 = PCA(n_components=3)
        X_c0_3d = pca_c0.fit_transform(X_c0)
        
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        scatter = ax.scatter(X_c0_3d[:, 0], X_c0_3d[:, 1], X_c0_3d[:, 2], c=belief_c0_s0, cmap='viridis', s=2, alpha=0.5)
        ax.set_title(f"Layer {l} C0 Belief 3D PCA")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
    
    fig.colorbar(scatter, ax=fig.axes, label="P(S_t=0 | C=0, prefix)", shrink=0.6)
    plt.suptitle("Within-Component C0 Belief 3D Manifold Across Layers")
    plt.savefig(out_dir / "activation_scatter_c0_by_belief.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 1. PCA Cumulative Variance Comparison
    plt.figure(figsize=(8, 5))
    for l in layers:
        plt.plot(np.arange(1, len(layer_cum_var[l]) + 1), layer_cum_var[l], label=f"Layer {l}")
    plt.axhline(0.90, color="gray", linestyle="--", alpha=0.5)
    plt.axhline(0.95, color="gray", linestyle="-.", alpha=0.5)
    plt.title("PCA Cumulative Explained Variance Across Layers")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance")
    plt.xlim(1, 40)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / "pca_cumulative_variance.png", dpi=150)
    plt.close()
    
    # 4. Multi-Layer Probe R^2 Comparison (Bar Plot)
    plt.figure(figsize=(10, 6))
    x = np.arange(len(targets))
    width = 0.25
    for i, l in enumerate(layers):
        vals = [layer_r2_scores[l][name] for name in targets.keys()]
        plt.bar(x + i*width, vals, width, label=f"Layer {l}")
    plt.xticks(x + width, list(targets.keys()))
    plt.title("Linear Decodability (R^2) Comparison Across Layers")
    plt.ylabel("Test R^2 Score")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.savefig(out_dir / "probe_r2_barplot.png", dpi=150)
    plt.close()
    
    # 5. Position Dependence Comparison (Final Layer only for clarity or all?)
    # User original plot had one color for posterior and one for belief.
    # Let's show both for Layer 2.
    plt.figure(figsize=(8, 5))
    plt.plot(unique_pos, layer_pos_r2s[2]["Posterior_P(C)"], label="L2 Posterior P(C)", color="blue", marker="o")
    avg_belief_l2 = np.mean([layer_pos_r2s[2]["Belief_C0"], layer_pos_r2s[2]["Belief_C1"], layer_pos_r2s[2]["Belief_C2"]], axis=0)
    plt.plot(unique_pos, avg_belief_l2, label="L2 Avg Belief P(S|C)", color="green", marker="s")
    plt.title("Context-Position Dependence (L2 Linear Probe R^2)")
    plt.xlabel("Prefix Length (t)")
    plt.ylabel("Test R^2 Score")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out_dir / "position_probe_r2.png", dpi=150)
    plt.close()

    # 6. Save Table
    rows = []
    for l in layers:
        for name, score in layer_r2_scores[l].items():
            rows.append({"Layer": l, "Target": name, "R2": score})
    pd.DataFrame(rows).to_csv(out_dir / "probe_r2_table.csv", index=False)

    # 7. Summary Metrics
    txt_content = "\n".join(sum_txt)
    with open(out_dir / "summary_metrics.txt", "w") as f:
        f.write(txt_content)
    print(txt_content)
    print("\nAll multi-layer deliverables generated in", out_dir)

if __name__ == "__main__":
    run_geometry_analysis()
