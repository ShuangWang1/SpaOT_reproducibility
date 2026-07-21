import os
import gc
import ot
import pickle
import anndata
import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import scipy.sparse as sp
import anndata as ad
import tifffile
from PIL import Image
import h5py
import commot as ct

# ── Arrow color ──────────────────────────────────────────────────────────────
ARROW_COLOR = "#091306"   # very dark green, almost black, to contrast with the colorful cell types
OUTPUT_DIR  = "ccc_direction_colored"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Cell type color map ───────────────────────────────────────────────────────
CELL_TYPE_CMAP = {
    "B cell":                                                       "#AEC6CF",  # pastel blue
    "Conventional DC 1":                                            "#C3B1E1",  # soft purple
    "Conventional DC 2":                                            "#B39EB5",  # mauve
    "MAST":                                                         "#FF6961",  # soft red
    "Mucosal-Associated Invariant T-Cell":                          "#77DD77",  # soft green
    "Natural Killer Cell":                                          "#03A9F4",  # sky blue
    "T Regulatory Cell":                                            "#A30059",  # dark rose
    "T cell":                                                       "#FFB347",  # pastel orange
    "Th17 cell":                                                    "#7a4900",  # brown
    "classical monocyte":                                           "#0000a6",  # navy
    "effector memory CD8-positive, alpha-beta T cell":              "#63ffac",  # mint
    "endothelial cell":                                             "#FF0000",  # bright red  ← distinct
    "epithelial cell of proximal tubule":                           "#004d43",  # dark teal
    "kidney collecting duct intercalated cell":                     "#8fb0ff",  # periwinkle
    "kidney collecting duct principal cell":                        "#997d87",  # dusty rose
    "kidney connecting tubule epithelial cell":                     "#5a0007",  # dark burgundy
    "kidney distal convoluted tubule epithelial cell":              "#809693",  # slate
    "kidney interstitial cell":                                     "#00E5FF",  # bright cyan  ← distinct
    "kidney loop of Henle thick ascending limb epithelial cell":    "#1b4400",  # forest green
    "kidney loop of Henle thin ascending limb epithelial cell":     "#4fc601",  # lime green
    "kidney loop of Henle thin descending limb epithelial cell":    "#3b5dff",  # blue
    "kidney resident macrophage":                                   "#9C27B0",  # purple
    "macrophage":                                                   "#FF2F80",  # hot pink
    "memory T cell":                                                "#61615a",  # warm grey
    "migratory dendritic cell":                                     "#ba0900",  # crimson
    "naive T cell":                                                 "#C8E6C9",  # light mint
    "non-classical monocyte":                                       "#00c2a0",  # teal
    "parietal epithelial cell":                                     "#FFAA92",  # peach
    "plasma Cell":                                                  "#FF90C9",  # pink
    "plasmacytoid Dendritic cell":                                  "#E040FB",  # magenta
    "podocyte":                                                     "#FF8C00",  # dark orange
    "schwann cell":                                                 "#DDEFFF",  # light blue-white
}
# ─────────────────────────────────────────────────────────────────────────────

adata_xen_all = sc.read_h5ad('../xen_iu04.h5ad')

# Read cell_type from the h5ad file that has compatibility issues, via h5py
with h5py.File('../Xenium_adata/IU04_with_cell_type.h5ad', 'r') as f:
    ct_cats = np.array(f['obs/cell_type/categories']).astype(str)
    ct_codes = np.array(f['obs/cell_type/codes'])
    cell_type_arr = ct_cats[ct_codes]
adata_xen_all.obs['cell_type'] = cell_type_arr
adata_xen_all.obs['cell_type'] = adata_xen_all.obs['cell_type'].astype('category')

img_xen_full = np.array(tifffile.imread("../Xenium/HE_images/0015383__IU04.TIF"))

img_h, img_w = img_xen_full.shape[:2]

x = adata_xen_all.obsm["spatial"][:, 0]
y = adata_xen_all.obsm["spatial"][:, 1]

x = x - x.min()
y = y - y.min()

scale_x = img_w / max(x)
scale_y = img_h / max(y)

x = x * scale_x
y = y * scale_y

x = img_w - x

# for sample 0
x -= 250
x -= 10
y -= 10
y -= 100
y -= 50

xen_x = x
xen_y = y

adata_xen_all.obsm['spatial'][:, 0] = xen_x
adata_xen_all.obsm['spatial'][:, 1] = xen_y


adata_vis_all = sc.read_h5ad('../vis_iu04.h5ad')


def impute_xenium(adata_vis, adata_xen, G0):
    X_vis = adata_vis.X
    if sp.issparse(X_vis):
        X_vis = X_vis.tocsr()

    # Normalize transport plan so weights per spot sum to 1
    G0 = G0 / G0.sum(axis=1, keepdims=True)

    # Impute expression
    X_xen_imputed = G0 @ X_vis

    # Create new AnnData for imputed full transcriptome
    adata_xen_imputed = ad.AnnData(
        X=X_xen_imputed,
        obs=adata_xen.obs.copy(),
        var=adata_vis.var.copy(),
        obsm=adata_xen.obsm.copy()
    )
    return adata_xen_imputed


df_cellchat = ct.pp.ligand_receptor_database(species='human', signaling_type='Secreted Signaling', database='CellChat')
print(df_cellchat.shape)

for i in range(1, 41):
    if not os.path.exists(f'../xen_hd_map/glom{i}_spaot_seg.pickle'):
        continue

    G0 = np.load(f'../xen_hd_map/glom{i}_spaot_seg.npy')

    adata_xen = adata_xen_all[adata_xen_all.obs['glomerulus'] == f'Selection {i}']
    adata_vis = adata_vis_all[adata_vis_all.obs['glomerulus'] == f'G{i}']

    if adata_xen.obs['spatial_region'][0] == 1:
        adata_xen.obsm['spatial'][:, 0] -= 50
    if adata_xen.obs['spatial_region'][0] == 2:
        adata_xen.obsm['spatial'][:, 0] -= 150
    if adata_xen.obs['spatial_region'][0] == 3:
        adata_xen.obsm['spatial'][:, 0] -= 250

    adata_vis_imputed1 = impute_xenium(adata_vis, adata_xen, G0)

    adata_vis_imputed1.var_names_make_unique()
    adata_vis_imputed1.obs_names_make_unique()

    x_sel = adata_vis_imputed1.obsm['spatial'][:, 0]
    y_sel = adata_vis_imputed1.obsm['spatial'][:, 1]

    pad = 50

    xmin = int(max(x_sel.min() - pad, 0))
    ymin = int(max(y_sel.min() - pad, 0))

    xen_x_crop = x_sel - xmin
    xen_y_crop = y_sel - ymin

    adata_vis_imputed1.obsm['spatial'][:, 0] = xen_x_crop
    adata_vis_imputed1.obsm['spatial'][:, 1] = xen_y_crop

    # run COMMOT
    df_cellchat_filtered = ct.pp.filter_lr_database(df_cellchat, adata_vis_imputed1, min_cell_pct=0.05)

    if len(df_cellchat_filtered) == 0:
        print(f'empty {i}')
        continue

    ct.tl.spatial_communication(
        adata_vis_imputed1,
        database_name='cellchat',
        df_ligrec=df_cellchat_filtered,
        dis_thr=500,
        heteromeric=True,
        pathway_sum=True,
    )

    for pathway_name in df_cellchat_filtered[2].unique():
        ct.tl.communication_direction(
            adata_vis_imputed1,
            database_name='cellchat',
            pathway_name=pathway_name,
            k=5,
        )
        out_pdf = os.path.join(OUTPUT_DIR, f'commot_res/ccc_direction_{i}_{pathway_name}_ct.pdf')
        ct.pl.plot_cell_communication(
            adata_vis_imputed1,
            database_name='cellchat',
            pathway_name=pathway_name,
            plot_method='grid',
            background_legend=True,
            scale=0.0004,
            ndsize=20,
            grid_density=0.4,
            summary='sender',
            background='cluster',
            clustering='cell_type',
            cluster_cmap=CELL_TYPE_CMAP,
            cmap='Alphabet',
            normalize_v=True,
            normalize_v_quantile=0.995,
            arrow_color=ARROW_COLOR,
            filename=out_pdf,
            grid_knn=5
        )
        print(f"Saved: {out_pdf}")
