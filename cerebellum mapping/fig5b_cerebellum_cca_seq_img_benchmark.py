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
from scipy.spatial import cKDTree
import ot

OUT_DIR = 'cerebellum_cca_seq_img_res'
os.makedirs(OUT_DIR, exist_ok=True)

N_PCA = 100
N_CCA = 10

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")

adata_seq  = sc.read_h5ad('cerebellum/Cerebellum-MAGIC-seq.h5ad')


X_seq_raw  = adata_seq.X.toarray().astype(np.float64)  # (1677, 16116)

sp_seq     = adata_seq.obsm['spatial'].astype(np.float32)


seq_labels  = np.array([s.replace(' ', '_') for s in adata_seq.obs['cluster'].values])
cats        = sorted(set(seq_labels))


with open('cerebellum_image2seq_from_MAGIC/tile_centers_no_bg.pickle', 'rb') as f:
    tile_centers = pickle.load(f)
with open('cerebellum_image2seq_from_MAGIC/tile_embeddings_dino.pickle', 'rb') as f:
    feat_tile_pca = pickle.load(f)

tile_centers = np.array(tile_centers).astype(np.float32) 
seq_coords = adata_seq.obsm['spatial'].astype(np.float32)
tree = cKDTree(seq_coords)
_, nn_idx = tree.query(tile_centers)
img_labels = seq_labels[nn_idx]

print(f"  img: {feat_tile_pca.shape}   Seq: {X_seq_raw.shape}")

m, n = len(feat_tile_pca), len(X_seq_raw)  # both 1677

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
    X_prot_recon = G_step1.T @ feat_prot.astype(np.float64) * feat_seq.shape[0]  # (n, N_PCA)
    cca = CCA(n_components=N_CCA)
    Zx, Zy = cca.fit_transform(X_prot_recon, feat_seq.astype(np.float64))
    X_prot_cca = cca.transform(feat_prot.astype(np.float64),
                                np.zeros((m, feat_seq.shape[1])))[0]  # (m, N_CCA)
    corrs = [float(np.corrcoef(Zx[:, i], Zy[:, i])[0, 1]) for i in range(N_CCA)]
    print(f"  [{label}] CCA correlations: {[f'{c:.3f}' for c in corrs]}")
    return X_prot_cca.astype(np.float32), Zy.astype(np.float32), corrs


resolved_path = os.path.realpath('..')
sys.path.append(resolved_path + '/moscot-framework_reproducibility')
from lib.fused_pgw import fused_partial_gromov_wasserstein

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — GW-only alignment (PCA feature structures, M = 0)
# Mirrors cerebellum_data_process_step1.py
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 1: GW-only (feature structure, M=0)")
print("=" * 60)

M_zero = np.zeros((m, n), dtype=np.float64)
p = np.ones(m) / m
q = np.ones(n) / n

 
structure_seq = ot.dist(X_seq_raw, X_seq_raw, metric='euclidean')
    
structure_seq = (structure_seq - structure_seq.min()) / (structure_seq.max() - structure_seq.min())
    
structure_meta = ot.dist(feat_tile_pca, feat_tile_pca, metric='euclidean')
    
structure_meta = (structure_meta - structure_meta.min()) / (structure_meta.max() - structure_meta.min())


G1 = fused_partial_gromov_wasserstein(
    M_zero, structure_meta.astype(np.float64), structure_seq.astype(np.float64),
    p=p.astype(np.float64), q=q.astype(np.float64),
    omega2=1.0, Lambda=1.0)

G1 = np.array(G1)

with open(f'{OUT_DIR}/step1_gw_plan.pickle', 'wb') as f:
    pickle.dump(G1, f)

acc1     = cluster_acc(G1, img_labels, seq_labels)
per_cl1  = per_cluster_acc(G1, img_labels, seq_labels)
print(f"  Overall accuracy : {acc1:.3f}")
print(f"  Per-cluster      : {per_cl1}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CCA on GW-reconstructed protein features
# Mirrors cerebellum_cca_step2.py
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 2: CCA (GW-reconstructed protein → seq)")
print("=" * 60)

# Transport protein PCA features to seq spots via the step-1 plan
X_prot_recon = G1.T @ feat_tile_pca * 1677  # (n, 100) — protein features in seq space

N_CCA = 10
cca = CCA(n_components=N_CCA)

Y = adata_seq.X.toarray()
Zx, Zy = cca.fit_transform(X_prot_recon, Y)   # both (1677, N_CCA)

with open(f'{OUT_DIR}/step2_cca_output.pickle', 'wb') as f:
    pickle.dump({'protein': Zx, 'seq': Zy}, f)

# Project raw protein PCA features into CCA space (used as cross-modal cost in step 3)
X_prot_cca = cca.transform(feat_tile_pca, np.zeros((m, Y.shape[1])))[0]  # (m, N_CCA)
with open(f'{OUT_DIR}/step2_cca_protein_proj.pickle', 'wb') as f:
    pickle.dump(X_prot_cca, f)

corrs = [float(np.corrcoef(Zx[:, i], Zy[:, i])[0, 1]) for i in range(N_CCA)]
print(f"  CCA canonical correlations: {[f'{c:.3f}' for c in corrs]}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FPGW: spatial structure + CCA cross-modal cost
# Mirrors cerebellum_fpgw_step3.py
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 3: FPGW (spatial structure + CCA cross-modal cost)")
print("=" * 60)

def spatial_structure(adata):
    sp = adata.obsm['spatial'].astype(np.float64)
    C = ot.dist(sp, sp, metric='euclidean')
    C = (C - C.min()) / (C.max() - C.min())
    return C

C_img_sp = ot.dist(np.array(tile_centers), np.array(tile_centers), metric='euclidean')
C_img_sp = (C_img_sp - C_img_sp.min()) / (C_img_sp.max() - C_img_sp.min())
C_seq_sp  = spatial_structure(adata_seq)


def make_cross_dist(A, B):
    M = ot.dist(A.astype(np.float64), B.astype(np.float64), metric='euclidean')
    M = (M - M.min()) / (M.max() - M.min())
    return M

M_cca = make_cross_dist(X_prot_cca, Zy)   # (m, n) — CCA-space cross-modal cost

G3 = fused_partial_gromov_wasserstein(
    M_cca.astype(np.float64), C_img_sp.astype(np.float64), C_seq_sp.astype(np.float64),
    p=p.astype(np.float64), q=q.astype(np.float64),
    omega2=0.5, Lambda=1.0)
G3 = np.array(G3)

with open(f'{OUT_DIR}/step3_fpgw_plan.pickle', 'wb') as f:
    pickle.dump(G3, f)

acc3    = cluster_acc(G3, img_labels, seq_labels)
per_cl3 = per_cluster_acc(G3, img_labels, seq_labels)
print(f"  Overall accuracy : {acc3:.3f}")
print(f"  Per-cluster      : {per_cl3}")





# ══════════════════════════════════════════════════════════════════════════════
# MOSCOT
# Convention mirrors cerebellum_image2seq_from_MAGIC_benchmark_moscot.py:
#   Step 1: X = spatial coords (cross-cost), obsm['spatial'] = PCA features (GW structure)
#           -> cross-cost is trivially near-zero (co-registered coords), effective pure GW
#   Step 3: X = CCA features (cross-modal cost), obsm['spatial'] = spatial coords (GW structure)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MOSCOT")
print("=" * 60)

def run_moscot(X_prot, coords_prot, X_seq, coords_seq, alpha=0.5, label=''):
    obs = pd.DataFrame(
        {'dataset': ['protein'] * len(X_prot) + ['seq'] * len(X_seq)},
        index=[f'prot_{i}' for i in range(len(X_prot))] +
              [f'seq_{i}'  for i in range(len(X_seq))]
    )
    adata_combined = ad.AnnData(
        X=np.vstack([X_prot, X_seq]).astype(np.float32), obs=obs)
    adata_combined.obsm['spatial'] = np.vstack([coords_prot, coords_seq]).astype(np.float32)
    ap = AlignmentProblem(adata=adata_combined)
    ap = ap.prepare(batch_key='dataset', policy='sequential')
    ap = ap.solve(alpha=alpha, scale_cost='mean', max_iterations=1000)
    G = np.array(ap[('protein', 'seq')].solution.transport_matrix)
    del ap, adata_combined
    acc = cluster_acc(G, img_labels, seq_labels)
    print(f"  [{label}] shape: {G.shape}, acc: {acc:.3f}")
    return G

# pca = PCA(n_components=1000)

# X_prot_PCA = pca.fit_transform(feat_tile_pca)
# X_seq_PCA = pca.fit_transform(X_seq_raw)

n_rows, n_cols = feat_tile_pca.shape
target_cols = 16116

pad_width = target_cols - n_cols

feat_tile_pca_reshape = np.pad(
    feat_tile_pca,
    pad_width=((0, 0), (0, pad_width)),
    mode='constant',
    constant_values=0
)

print("Step 1: Moscot GW — PCA features as spatial structure, coords as X...")
G_mos1 = run_moscot(tile_centers, feat_tile_pca_reshape, sp_seq, X_seq_raw, alpha=0.999, label='GW')
with open(f'{OUT_DIR}/benchmark_moscot_step1_plan.pickle', 'wb') as f:
    pickle.dump(G_mos1, f)

print("Step 2: CCA (Moscot)...")
X_prot_cca_mos, Zy_mos, corrs_mos = run_cca_step(G_mos1, feat_tile_pca, X_seq_raw, label='Moscot CCA')

print("Step 3: Moscot FGW — spatial coords as structure, CCA features as X...")
G_mos3 = run_moscot(X_prot_cca_mos, tile_centers, Zy_mos, sp_seq, alpha=0.5, label='FGW')
with open(f'{OUT_DIR}/benchmark_moscot_step3_plan.pickle', 'wb') as f:
    pickle.dump(G_mos3, f)

acc_mos1    = cluster_acc(G_mos1, img_labels, seq_labels)
acc_mos3    = cluster_acc(G_mos3, img_labels, seq_labels)
per_cl_mos1 = per_cluster_acc(G_mos1, img_labels, seq_labels)
per_cl_mos3 = per_cluster_acc(G_mos3, img_labels, seq_labels)
print(f"Moscot GW  overall accuracy: {acc_mos1:.3f}")
print(f"Moscot FGW overall accuracy: {acc_mos3:.3f}")




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
    acc = cluster_acc(G, img_labels, seq_labels)
    print(f"  [{label}] shape: {G.shape}, acc: {acc:.3f}")
    return G

print("Step 1: PASTE2 GW — PCA features as spatial structure, coords as X...")

G_p21 = run_paste2(tile_centers, feat_tile_pca, sp_seq, X_seq_raw, alpha=1.0, s=0.99, label='GW')
with open(f'{OUT_DIR}/benchmark_paste2_step1_plan.pickle', 'wb') as f:
    pickle.dump(G_p21, f)

print("Step 2: CCA (PASTE2)...")
X_prot_cca_p2, Zy_p2, corrs_p2 = run_cca_step(G_p21, feat_tile_pca, X_seq_raw, label='PASTE2 CCA')

print("Step 3: PASTE2 FGW — spatial coords as structure, CCA features as X...")
G_p23 = run_paste2(X_prot_cca_p2, tile_centers, Zy_p2, sp_seq, alpha=0.5, s=0.99, label='FGW')
with open(f'{OUT_DIR}/benchmark_paste2_step3_plan.pickle', 'wb') as f:
    pickle.dump(G_p23, f)

acc_p21    = cluster_acc(G_p21, img_labels, seq_labels)
acc_p23    = cluster_acc(G_p23, img_labels, seq_labels)
per_cl_p21 = per_cluster_acc(G_p21, img_labels, seq_labels)
per_cl_p23 = per_cluster_acc(G_p23, img_labels, seq_labels)
print(f"PASTE2 GW  overall accuracy: {acc_p21:.3f}")
print(f"PASTE2 FGW overall accuracy: {acc_p23:.3f}")
with open(f'{OUT_DIR}/benchmark_paste2_results.pickle', 'wb') as f:
    pickle.dump({'step1_acc': acc_p21, 'step3_acc': acc_p23,
                 'step1_per_cl': per_cl_p21, 'step3_per_cl': per_cl_p23,
                 'corrs': corrs_p2}, f)
    

# reading pickle
with open('cerebellum_image2meta/gw_plan_moscot.pickle', 'rb') as f:
    G_moscot = pickle.load(f)
with open('cerebellum_image2meta/gw_plan_paste2.pickle', 'rb') as f:
    G_paste2 = pickle.load(f)
with open('cerebellum_image2meta/gw_plan_ours.pickle', 'rb') as f:
    G_ours = pickle.load(f)    
    

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
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
panels = [
    ('Ground truth (Seq clusters)',           seq_labels,                       None),
    (f'Moscot FGW (acc={acc_mos3:.3f})',      prot_labels[np.argmax(G_mos3, axis=0)], None),
    (f'PASTE2 FGW (acc={acc_p23:.3f})',       prot_labels[np.argmax(G_p23,  axis=0)], None),
]
for ax, (title, labels, _) in zip(axes, panels):
    sp = adata_seq.obsm['spatial']
    for cl, col in palette.items():
        mask = labels == cl
        if np.any(mask):
            ax.scatter(sp[mask, 0], sp[mask, 1], c=col, s=16,
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

print("\nDone. All results in:", OUT_DIR)




#plot
# ── Cluster colour palette ────────────────────────────────────────────────────
palette = {
    'Fiber tracts':   '#E74C3C',
    'Granular layer': '#3498DB',
    'Lateral recess': '#2ECC71',
    'Molecular layer':'#9B59B6',
}
cats = sorted(set(clusters))

# ── Best-match seq spot per tile ──────────────────────────────────────────────
matched_idx = np.argmax(G1, axis=1)   # (n_tiles,) → index into seq_coords

# ── Compute crop region (zoom to tile coverage) ───────────────────────────────
pad    = PATCH_SIZE
tile_centers = np.array(tile_centers)
x_min  = max(0,  tile_centers[:, 0].min() - pad)
x_max  = min(W,  tile_centers[:, 0].max() + pad)
y_min  = max(0,  tile_centers[:, 1].min() - pad)
y_max  = min(H,  tile_centers[:, 1].max() + pad)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10),
                                gridspec_kw={'wspace': 0.05})

# ── Left: image mosaic ────────────────────────────────────────────────────────
# Set extent so data coords match original (unscaled) pixel coordinates.
ax1.imshow(canvas, extent=[0, W, H, 0], origin='upper', interpolation='nearest')
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_max, y_min)   # y inverted: larger y at bottom
ax1.set_title("Image tiles (H&E patch mosaic)", fontsize=14, fontweight='bold', pad=8)
ax1.axis('off')

# Overlay small coloured dots at the sampled tile centres
for i in range(len(tile_centers)):
    tx, ty = tile_centers[i]
    cl = clusters[matched_idx[i]]
    col = palette.get(cl, 'gray')
    ax1.plot(tx, ty, 'o', ms=10, color=col, markeredgecolor='white',
             markeredgewidth=0.4, zorder=6, alpha=0.85)

# ── Right: seq scatter ────────────────────────────────────────────────────────
for cl in cats:
    m = clusters == cl
    ax2.scatter(seq_coords[m, 0], seq_coords[m, 1],
                c=palette.get(cl, 'gray'), s=50, linewidths=0,
                rasterized=True, label=cl, alpha=0.9, zorder=4)

ax2.invert_yaxis()
ax2.set_aspect('equal')
ax2.set_xlim(seq_coords[:, 0].min() - 50, seq_coords[:, 0].max() + 50)
ax2.set_ylim(seq_coords[:, 1].max() + 50, seq_coords[:, 1].min() - 50)
ax2.set_title("Sequence spots (spatial)", fontsize=14, fontweight='bold', pad=8)
ax2.axis('off')
ax2.legend(fontsize=10, loc='lower left', frameon=True, framealpha=0.9,
           markerscale=1.5, title='Cluster', title_fontsize=10)

plt.suptitle(
    "Image → Sequence Mapping  (GW transport plan: gw_plan_ours_noM)",
    fontsize=15, fontweight='bold', y=1.01)

out_path = "cerebellum_image2seq_from_MAGIC/tmp_new.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()