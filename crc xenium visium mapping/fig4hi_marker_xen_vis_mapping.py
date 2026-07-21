import scanpy as sc
import numpy as np

import squidpy as sq

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import pickle


# data preparation
adata = sc.read_h5ad("xenium_p2/xenium.h5ad")
adata_8um = sc.read_h5ad("visium_p2/visium.h5ad")

common_genes = list(set(adata.var_names).intersection(adata_8um.var_names))
adata = adata[:, common_genes].copy()

#remove duplicate
adata_8um = adata_8um[:, ~adata_8um.var_names.duplicated()].copy()
adata_8um = adata_8um[:, common_genes].copy()

n_sample = 10000  # adjust based on memory
rng = np.random.default_rng(seed=42) 
idx_xenium = rng.choice(adata.n_obs, n_sample, replace=False)

adata_X = adata[idx_xenium]
adata_Y = adata_8um

sc.pp.log1p(adata_X)
sc.pp.log1p(adata_Y)



#load moscot transport plan and compute morans I for xenium after mapping
with open('moscot_plan.pickle','rb') as f:
    mos = pickle.load(f)
G_mos = np.array(mos.transport_matrix)

row_sums = G_mos.T.sum(axis=1, keepdims=True) 
Xs_proj = (G_mos.T @ adata_Y.X) / row_sums  

adata_X_mos = adata_X.copy()
adata_X_mos.X = Xs_proj


#load fpgw transport plan
with open('fpgw_moscot_init','rb') as f:
    G0 = pickle.load(f)

row_sums = G0.T.sum(axis=1, keepdims=True) 
Xs_proj = (G0.T @ adata_Y.X) / row_sums  

adata_X_fpgw = adata_X.copy()
adata_X_fpgw.X = Xs_proj

row_sums = G_mos.sum(axis=1, keepdims=True) 
Xs_proj = (G_mos @ adata_X.X) / row_sums  

adata_Y_mos = adata_Y.copy()
adata_Y_mos.X = Xs_proj


row_sums = G0.sum(axis=1, keepdims=True) 
Xs_proj = (G0 @ adata_X.X) / row_sums  

adata_Y_fpgw = adata_Y.copy()
adata_Y_fpgw.X = Xs_proj


genes = ['S100A12','FCER2','CLEC9A',  
      'PIGR', 'DES', 'CXCL10', 'APOE', 'SULT1B1', 'DPYSL3', 'FABP2',                       
      'UGT2B17', 'L1TD1', 'CXCL9', 'CXCL11', 'ANXA1', 'CHI3L1', 'CCR7' ]

#xenium to visium mapping, reconstruct visium

datasets = {
    "moscot": adata_Y_mos,
    "fpgw": adata_Y_fpgw,
    "xen": adata_X,
    "vis": adata_Y,
}
for gene in genes:
    all_values = []

    # --- collect values across datasets ---
    for ad in datasets.values():
        vals = ad[:, gene].X
        if hasattr(vals, "toarray"):
            vals = vals.toarray()
        all_values.append(vals.flatten())

    all_values = np.concatenate(all_values)

    # --- shared normalization ---
    vmin = all_values.min()
    vmax = all_values.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    # --- create 1x4 subplot ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, (name, ad) in zip(axes, datasets.items()):
        vals = ad[:, gene].X
        if hasattr(vals, "toarray"):
            vals = vals.toarray()
        vals = vals.flatten()

        coords = ad.obsm['spatial']
        
        if name == "vis":
            coords = np.column_stack([coords[:, 1], -coords[:, 0]])

        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        s=5, c=vals, cmap=cmap, norm=norm)
        mosI = ad.uns["moranI"].loc[gene, "I"] if gene in ad.uns["moranI"].index else np.nan
        ax.set_title(f"{name} (Moran's I: {mosI:.3f})")
        ax.set_xticks([])
        ax.set_yticks([])

    # --- shared colorbar ---
    # cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.04)
    # cbar.set_label(f"{gene} expression")

    plt.suptitle(gene)
    plt.tight_layout()

    plt.savefig(f'moranI_marker_gene_vis_reconstruct/{gene}_combined.png', dpi=150)
    plt.close()
    

#visium to xenium mapping, reconstruct xenium
datasets = {
    "moscot": adata_X_mos,
    "fpgw": adata_X_fpgw,
    "xen": adata_X,
    "vis": adata_Y,
}
for gene in genes:
    all_values = []

    # --- collect values across datasets ---
    for ad in datasets.values():
        vals = ad[:, gene].X
        if hasattr(vals, "toarray"):
            vals = vals.toarray()
        all_values.append(vals.flatten())

    all_values = np.concatenate(all_values)

    # --- shared normalization ---
    vmin = all_values.min()
    vmax = all_values.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    # --- create 1x4 subplot ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, (name, ad) in zip(axes, datasets.items()):
        vals = ad[:, gene].X
        if hasattr(vals, "toarray"):
            vals = vals.toarray()
        vals = vals.flatten()

        coords = ad.obsm['spatial'].copy()

        # 🔄 rotate ONLY vis
        if name == "vis":
            coords = np.column_stack([coords[:, 1], -coords[:, 0]])

        ax.scatter(coords[:, 0], coords[:, 1],
                   s=5, c=vals, cmap=cmap, norm=norm)

        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    plt.suptitle(gene)
    plt.tight_layout()

    plt.savefig(f'moranI_marker_gene/{gene}_combined.png', dpi=150)
    plt.close()
