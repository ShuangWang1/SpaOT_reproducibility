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
import commot as ct

# ── Arrow color ──────────────────────────────────────────────────────────────
# H&E background is dark purple; use a bright contrasting color.
# Options: "#FFFF00" (yellow), "#00FFFF" (cyan), "#FFFFFF" (white), "#FF6600" (orange)
ARROW_COLOR = "#FFFF00"   # bright yellow — change this to your preference
OUTPUT_DIR  = "ccc_direction_colored"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

adata_xen_all = sc.read_h5ad('../xen_iu04.h5ad')
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

    # put image to adata_vis_imputed1
    library_id = "custom"

    img = np.array(Image.open(f"../xen_img/glom_{i}.png"))

    x_sel = adata_vis_imputed1.obsm['spatial'][:, 0]
    y_sel = adata_vis_imputed1.obsm['spatial'][:, 1]

    pad = 50

    xmin = int(max(x_sel.min() - pad, 0))
    ymin = int(max(y_sel.min() - pad, 0))

    xen_x_crop = x_sel - xmin
    xen_y_crop = y_sel - ymin

    adata_vis_imputed1.obsm['spatial'][:, 0] = xen_x_crop
    adata_vis_imputed1.obsm['spatial'][:, 1] = xen_y_crop

    adata_vis_imputed1.uns['spatial'] = {
        library_id: {
            "images": {
                "hires": img
            },
            "scalefactors": {
                "tissue_hires_scalef": 1.0
            }
        }
    }

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
        out_pdf = os.path.join(OUTPUT_DIR, f'commot_res/ccc_direction_{i}_{pathway_name}.pdf')
        ct.pl.plot_cell_communication(
            adata_vis_imputed1,
            database_name='cellchat',
            pathway_name=pathway_name,
            plot_method='grid',
            background_legend=True,
            scale=0.0004,
            ndsize=1,
            grid_density=0.4,
            summary='sender',
            background='image',
            clustering='leiden',
            cmap='Alphabet',
            normalize_v=True,
            normalize_v_quantile=0.995,
            arrow_color=ARROW_COLOR,   # <── bright color for dark H&E background
            filename=out_pdf,
            grid_knn=5
        )
        print(f"Saved: {out_pdf}")
        
        
#venn diagram for vis, xen, and imputed
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

os.makedirs("commot_venn", exist_ok=True)

def get_lr_set(df):
    """
    Tries to extract a set of ligand-receptor identifiers from a filtered table.
    Adjust this if your filtered dataframe uses a specific column name.
    """
    if df is None or len(df) == 0:
        return set()

    for col in ["interaction_name", "pair", "lr_pair", "lr", "ligand_receptor"]:
        if col in df.columns:
            return set(df[col].astype(str).tolist())

    # fallback: use index
    return set(df.index.astype(str).tolist())

summary_rows = []
for i in range(1, 41):
    if not os.path.exists(f'../xen_hd_map/glom{i}_moscot_init_seg.pickle'):
        continue

    G0 = np.load(f'../xen_hd_map/glom{i}_moscot_init_seg.npy')

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

    # put image to adata_vis_imputed1
    library_id = "custom"

    img = np.array(Image.open(f"../xen_img/glom_{i}.png"))

    x_sel = adata_vis_imputed1.obsm['spatial'][:, 0]
    y_sel = adata_vis_imputed1.obsm['spatial'][:, 1]

    pad = 50

    xmin = int(max(x_sel.min() - pad, 0))
    ymin = int(max(y_sel.min() - pad, 0))

    xen_x_crop = x_sel - xmin
    xen_y_crop = y_sel - ymin

    adata_vis_imputed1.obsm['spatial'][:, 0] = xen_x_crop
    adata_vis_imputed1.obsm['spatial'][:, 1] = xen_y_crop

    adata_vis_imputed1.uns['spatial'] = {
        library_id: {
            "images": {
                "hires": img
            },
            "scalefactors": {
                "tissue_hires_scalef": 1.0
            }
        }
    }

    # run COMMOT
    df_cellchat_filtered = ct.pp.filter_lr_database(df_cellchat, adata_vis_imputed1, min_cell_pct=0.05)
    df_cellchat_filtered_vis = ct.pp.filter_lr_database(df_cellchat, adata_vis, min_cell_pct=0.05)
    df_cellchat_filtered_xen = ct.pp.filter_lr_database(df_cellchat, adata_xen, min_cell_pct=0.05)

    # Convert to sets
    set_imp = get_lr_set(df_cellchat_filtered)
    set_vis = get_lr_set(df_cellchat_filtered_vis)
    set_xen = get_lr_set(df_cellchat_filtered_xen)
    
    print(f"Glomerulus {i}:")
    print(f"  Imputed: {len(set_imp)}")
    print(f"  Vis: {len(set_vis)}")
    print(f"  Xen: {len(set_xen)}")
    
    
    n_imp = len(set_imp)
    n_vis = len(set_vis)
    n_xen = len(set_xen)
    
    summary_rows.append({
        "glomerulus": f"glomerulus {i}",
        "Imputed": n_imp,
        "Vis": n_vis,
        "Xen": n_xen
    })
    

    # Plot Venn diagram
    fig, ax = plt.subplots(figsize=(6, 6))
    venn3(
        subsets=(set_imp, set_vis, set_xen),
        set_labels=("Imputed", "Vis", "Xen"),
        ax=ax
    )
    ax.set_title(f"Glomerulus {i}")

    out_path = f"commot_venn/glom_{i}_venn.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    

#make a barplot of the counts

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.rename(columns={
    "Imputed": "SpaOT-mapped",
    "Vis": "Visium HD",
    "Xen": "Xenium"
})

import seaborn as sns    
    
    
from matplotlib.ticker import MaxNLocator

plot_df = summary_df.melt(
    id_vars="glomerulus",
    value_vars=["SpaOT-mapped", "Visium HD", "Xenium"],
    var_name="dataset",
    value_name="count"
)

palette = {
    "SpaOT-mapped": "orange",
    "Visium HD": "#1f77b4",
    "Xenium": "#008fee",
}

plt.figure(figsize=(14, 6))
ax = sns.barplot(
    data=plot_df,
    x="glomerulus",
    y="count",
    hue="dataset",
    palette=palette
)

ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.xticks(rotation=45, ha="right")
plt.xlabel("")
plt.ylabel("LR count")
plt.title("LR counts per glomerulus")
plt.tight_layout()
plt.savefig("commot_venn/overall_lr_counts_barplot.png", dpi=300, bbox_inches="tight")
plt.close()