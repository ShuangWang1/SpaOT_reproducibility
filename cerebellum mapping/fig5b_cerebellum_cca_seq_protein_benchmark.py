"""
Benchmark: Moscot and PASTE2 on Seq <-> Protein alignment
Same 3-step pipeline as cerebellum_cca_seq_protein.py:
  Step 1: GW-only (feature structure as spatial in moscot/paste2, spatial coords as X)
  Step 2: CCA on GW-reconstructed features
  Step 3: FGW (spatial coords as structure, CCA features as cross-modal cost)

Data:
  Protein: cerebellum/Cerebellum-PLATO.h5ad    (1677 x 1399)
  Seq:     cerebellum/Cerebellum-MAGIC-seq.h5ad (1677 x 16116)
Both share identical spatial coordinates (co-registered Visium spots).
PCA (100 components) applied before distance computation.
"""

import os, sys, pickle
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from moscot.problems.space import AlignmentProblem
from paste2 import PASTE2

OUT_DIR = 'cerebellum_cca_seq_protein_res'
os.makedirs(OUT_DIR, exist_ok=True)

N_PCA = 100
N_CCA = 10

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
adata_prot = sc.read_h5ad('cerebellum/Cerebellum-PLATO.h5ad')
adata_seq  = sc.read_h5ad('cerebellum/Cerebellum-MAGIC-seq.h5ad')

X_prot_raw = adata_prot.X.astype(np.float64)           # (1677, 1399)
X_seq_raw  = adata_seq.X.toarray().astype(np.float64)  # (1677, 16116)
sp_prot    = adata_prot.obsm['spatial'].astype(np.float32)
sp_seq     = adata_seq.obsm['spatial'].astype(np.float32)

prot_labels = np.array([s.replace(' ', '_') for s in adata_prot.obs['cluster'].values])
seq_labels  = np.array([s.replace(' ', '_') for s in adata_seq.obs['cluster'].values])
cats        = sorted(set(seq_labels))

print(f"  Protein: {X_prot_raw.shape}   Seq: {X_seq_raw.shape}")

m, n = len(X_prot_raw), len(X_seq_raw)  # both 1677

# ── Accuracy metrics (same as cerebellum_cca_seq_protein.py) ──────────────────
def cluster_acc(G, src_labels, tgt_labels):
    """For each tgt spot pick argmax src, compare labels."""
    idx  = np.argmax(G, axis=0)
    pred = src_labels[idx]
    return float(np.mean(pred == tgt_labels))

def per_cluster_acc(G, src_labels, tgt_labels):
    idx  = np.argmax(G, axis=0)
    pred = src_labels[idx]
    return {cl: float(np.mean(pred[tgt_labels == cl] == cl)) for cl in cats}

# ── CCA step: mirrors cerebellum_cca_seq_protein.py Step 2 ────────────────────
def run_cca_step(G_step1, feat_prot, feat_seq, label=''):
    X_prot_recon = G_step1.T @ feat_prot.astype(np.float64)  # (n, N_PCA)
    cca = CCA(n_components=N_CCA)
    Zx, Zy = cca.fit_transform(X_prot_recon, feat_seq.astype(np.float64))
    X_prot_cca = cca.transform(feat_prot.astype(np.float64),
                                np.zeros((m, feat_seq.shape[1])))[0]  # (m, N_CCA)
    corrs = [float(np.corrcoef(Zx[:, i], Zy[:, i])[0, 1]) for i in range(N_CCA)]
    print(f"  [{label}] CCA correlations: {[f'{c:.3f}' for c in corrs]}")
    return X_prot_cca.astype(np.float32), Zy.astype(np.float32), corrs

# ══════════════════════════════════════════════════════════════════════════════
# MOSCOT
# Convention mirrors cerebellum_image2seq_from_MAGIC_benchmark_moscot.py:
#   Step 1: X = spatial coords (cross-cost), obsm['spatial'] = PCA features (GW structure)
#           -> cross-cost is trivially near-zero (co-registered coords), effective pure GW
#   Step 3: X = CCA features (cross-modal cost), obsm['spatial'] = spatial coords (GW structure)
# ══════════════════════════════════════════════════════════════════════════════
# print("\n" + "=" * 60)
# print("MOSCOT")
# print("=" * 60)

# def run_moscot(X_prot, coords_prot, X_seq, coords_seq, alpha=0.5, label=''):
#     obs = pd.DataFrame(
#         {'dataset': ['protein'] * len(X_prot) + ['seq'] * len(X_seq)},
#         index=[f'prot_{i}' for i in range(len(X_prot))] +
#               [f'seq_{i}'  for i in range(len(X_seq))]
#     )
#     adata_combined = ad.AnnData(
#         X=np.vstack([X_prot, X_seq]).astype(np.float32), obs=obs)
#     adata_combined.obsm['spatial'] = np.vstack([coords_prot, coords_seq]).astype(np.float32)
#     ap = AlignmentProblem(adata=adata_combined)
#     ap = ap.prepare(batch_key='dataset', policy='sequential')
#     ap = ap.solve(alpha=alpha, scale_cost='mean', max_iterations=1000)
#     G = np.array(ap[('protein', 'seq')].solution.transport_matrix)
#     del ap, adata_combined
#     acc = cluster_acc(G, prot_labels, seq_labels)
#     print(f"  [{label}] shape: {G.shape}, acc: {acc:.3f}")
#     return G

# # pca = PCA(n_components=1000)

# # X_prot_PCA = pca.fit_transform(X_prot_raw)
# # X_seq_PCA = pca.fit_transform(X_seq_raw)

# n_rows, n_cols = X_prot_raw.shape
# target_cols = 16116

# pad_width = target_cols - n_cols

# X_prot_raw_reshape = np.pad(
#     X_prot_raw,
#     pad_width=((0, 0), (0, pad_width)),
#     mode='constant',
#     constant_values=0
# )

# print("Step 1: Moscot GW — PCA features as spatial structure, coords as X...")
# G_mos1 = run_moscot(sp_prot, X_prot_raw_reshape, sp_seq, X_seq_raw, alpha=0.999, label='GW')
# with open(f'{OUT_DIR}/benchmark_moscot_step1_plan.pickle', 'wb') as f:
#     pickle.dump(G_mos1, f)

# print("Step 2: CCA (Moscot)...")
# X_prot_cca_mos, Zy_mos, corrs_mos = run_cca_step(G_mos1, X_prot_raw, X_seq_raw, label='Moscot CCA')

# print("Step 3: Moscot FGW — spatial coords as structure, CCA features as X...")
# G_mos3 = run_moscot(X_prot_cca_mos, sp_prot, Zy_mos, sp_seq, alpha=0.5, label='FGW')
# with open(f'{OUT_DIR}/benchmark_moscot_step3_plan.pickle', 'wb') as f:
#     pickle.dump(G_mos3, f)

# acc_mos1    = cluster_acc(G_mos1, prot_labels, seq_labels)
# acc_mos3    = cluster_acc(G_mos3, prot_labels, seq_labels)
# per_cl_mos1 = per_cluster_acc(G_mos1, prot_labels, seq_labels)
# per_cl_mos3 = per_cluster_acc(G_mos3, prot_labels, seq_labels)
# print(f"Moscot GW  overall accuracy: {acc_mos1:.3f}")
# print(f"Moscot FGW overall accuracy: {acc_mos3:.3f}")
# with open(f'{OUT_DIR}/benchmark_moscot_results.pickle', 'wb') as f:
#     pickle.dump({'step1_acc': acc_mos1, 'step3_acc': acc_mos3,
#                  'step1_per_cl': per_cl_mos1, 'step3_per_cl': per_cl_mos3,
#                  'corrs': corrs_mos}, f)

# ══════════════════════════════════════════════════════════════════════════════
# PASTE2
# Convention mirrors cerebellum_image2seq_from_MAGIC_benchmark_paste2.py:
#   Step 1: data.X = spatial coords, obsm['spatial'] = PCA features (GW structure)
#   Step 3: data.X = CCA features,   obsm['spatial'] = spatial coords (GW structure)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PASTE2")
print("=" * 60)

def run_paste2(X_prot, coords_prot, X_seq, coords_seq, alpha=0.5, s=0.95, label=''):
    data_prot = ad.AnnData(
        X=X_prot.astype(np.float32),
        obs=pd.DataFrame(index=[f'prot_{i}' for i in range(len(X_prot))]),
        var=pd.DataFrame(index=[f'f{i}'    for i in range(X_prot.shape[1])]))
    data_prot.obsm['spatial'] = coords_prot.astype(np.float32)
    data_seq = ad.AnnData(
        X=X_seq.astype(np.float32),
        obs=pd.DataFrame(index=[f'seq_{i}' for i in range(len(X_seq))]),
        var=pd.DataFrame(index=[f'f{i}'   for i in range(X_seq.shape[1])]))
    data_seq.obsm['spatial'] = coords_seq.astype(np.float32)
    G, cost = PASTE2.partial_pairwise_align(
        data_prot, data_seq, s=s, alpha=alpha,
        dissimilarity='euclidean', return_obj=True, verbose=False)
    G = np.array(G)
    acc = cluster_acc(G, prot_labels, seq_labels)
    print(f"  [{label}] shape: {G.shape}, acc: {acc:.3f}")
    return G

print("Step 1: PASTE2 GW — PCA features as spatial structure, coords as X...")

G_p21 = run_paste2(sp_prot, X_prot_raw, sp_seq, X_seq_raw, alpha=1.0, s=0.99, label='GW')
with open(f'{OUT_DIR}/benchmark_paste2_step1_plan.pickle', 'wb') as f:
    pickle.dump(G_p21, f)

print("Step 2: CCA (PASTE2)...")
X_prot_cca_p2, Zy_p2, corrs_p2 = run_cca_step(G_p21, X_prot_raw, X_seq_raw, label='PASTE2 CCA')

print("Step 3: PASTE2 FGW — spatial coords as structure, CCA features as X...")
G_p23 = run_paste2(X_prot_cca_p2, sp_prot, Zy_p2, sp_seq, alpha=0.5, s=0.99, label='FGW')
with open(f'{OUT_DIR}/benchmark_paste2_step3_plan.pickle', 'wb') as f:
    pickle.dump(G_p23, f)

acc_p21    = cluster_acc(G_p21, prot_labels, seq_labels)
acc_p23    = cluster_acc(G_p23, prot_labels, seq_labels)
per_cl_p21 = per_cluster_acc(G_p21, prot_labels, seq_labels)
per_cl_p23 = per_cluster_acc(G_p23, prot_labels, seq_labels)
print(f"PASTE2 GW  overall accuracy: {acc_p21:.3f}")
print(f"PASTE2 FGW overall accuracy: {acc_p23:.3f}")
with open(f'{OUT_DIR}/benchmark_paste2_results.pickle', 'wb') as f:
    pickle.dump({'step1_acc': acc_p21, 'step3_acc': acc_p23,
                 'step1_per_cl': per_cl_p21, 'step3_per_cl': per_cl_p23,
                 'corrs': corrs_p2}, f)

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print(f"{'Method':<25} {'Overall':>8}  " + "  ".join(f"{cl[:14]:>14}" for cl in cats))
print("-" * 90)
for name, acc, per_cl in [
    ("Moscot GW (Step1)",   acc_mos1, per_cl_mos1),
    ("Moscot FGW (Step3)",  acc_mos3, per_cl_mos3),
    ("PASTE2 GW (Step1)",   acc_p21,  per_cl_p21),
    ("PASTE2 FGW (Step3)",  acc_p23,  per_cl_p23),
]:
    print(f"{name:<25} {acc:>8.3f}  " +
          "  ".join(f"{per_cl.get(cl, float('nan')):>14.3f}" for cl in cats))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
colors  = adata_seq.uns["cluster_colors"]
palette = dict(zip(cats, colors))

# Fig 1 — per-cluster accuracy bar chart (4 methods)
fig, ax = plt.subplots(figsize=(11, 5))
x       = np.arange(len(cats))
width   = 0.18
methods = [
    ("Moscot GW",  per_cl_mos1, '#607D8B'),
    ("Moscot FGW", per_cl_mos3, '#2196F3'),
    ("PASTE2 GW",  per_cl_p21,  '#FF9800'),
    ("PASTE2 FGW", per_cl_p23,  '#E91E63'),
]
offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

for (label, per_cl, col), off in zip(methods, offsets):
    vals = [per_cl.get(cl, 0.0) for cl in cats]
    bars = ax.bar(x + off, vals, width=width, label=label, color=col,
                  alpha=0.88, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                f'{v:.2f}', ha='center', va='bottom', fontsize=7, rotation=40)

ax.set_xticks(x)
ax.set_xticklabels([cl.replace('_', '\n') for cl in cats], fontsize=10)
ax.set_ylabel('Label-transfer accuracy', fontsize=11)
ax.set_ylim(0, 1.35)
ax.axhline(1.0, ls='--', color='gray', lw=0.8, alpha=0.5)
ax.legend(fontsize=9, frameon=True)
ax.set_title('Seq <-> Protein: Benchmark (Moscot vs PASTE2)', fontsize=12, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/benchmark_per_cluster_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved {OUT_DIR}/benchmark_per_cluster_accuracy.png")

# Fig 2 — spatial scatter for FGW results (best step of each method)
fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
panels = [
    ('Ground truth (RNA-Seq clusters)',           seq_labels,                       None),
    (f'SpaOT (acc={acc_fpgw3:.3f})',      prot_labels[np.argmax(G_fpgw3, axis=0)], None),
    (f'Moscot (acc={acc_mos3:.3f})',      prot_labels[np.argmax(G_mos3, axis=0)], None),
    (f'PASTE2 (acc={acc_p23:.3f})',       prot_labels[np.argmax(G_p23,  axis=0)], None),
]
for ax, (title, labels, _) in zip(axes, panels):
    sp = adata_seq.obsm['spatial']
    for cl, col in palette.items():
        mask = labels == cl
        if np.any(mask):
            ax.scatter(sp[mask, 0], sp[mask, 1], c=col, s=50,
                       linewidths=0, rasterized=True, label=cl.replace('_', ' '))
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold')

axes[0].legend(fontsize=7, loc='lower left', frameon=True, framealpha=0.85)
plt.suptitle('Seq <-> Protein: spatial label transfer (benchmark)',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/benchmark_spatial_transfer.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {OUT_DIR}/benchmark_spatial_transfer.png")

import numpy as np
import matplotlib.pyplot as plt

# ====== DATA ======
methods = {
    "spaOT": [0.776, 0.782, 0.717],
    "Moscot": [0.555, 0.747, 0.437],
    "Paste2": [0.794, 0.510, 0.427],
}

labels = ["Protein", "Metabolite", "Image"]

# ====== RADAR CHART SETUP ======
num_vars = len(labels)

# Angles for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close the loop

# ====== FIGURE ======
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

# Start from the top and go clockwise
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Axis labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)

# Radial limits: 0 to 1.0
ax.set_ylim(0, 0.8)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=10)

# Grid style
ax.grid(True, linestyle="--", alpha=0.5)

# ====== PLOT EACH METHOD ======
colors = {
    "spaOT": "red",
    "Moscot": "blue",
    "Paste2": "green",
}

for name, values in methods.items():
    vals = values + values[:1]  # close the loop
    ax.plot(angles, vals, linewidth=2, label=name, color=colors[name])
    ax.fill(angles, vals, alpha=0.15, color=colors[name])

# Legend
plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1.10))

plt.tight_layout()
plt.savefig("demo_radar.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nDone. All results in:", OUT_DIR)
