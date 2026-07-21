import pandas as pd
import scanpy as sc
import numpy as np
    
import anndata as ad   
import scipy.sparse as sp

import pickle


adata = sc.read_h5ad("xenium_p2/xenium.h5ad")
adata_8um = sc.read_h5ad("visium_p2/visium.h5ad")

common_genes = list(set(adata.var_names).intersection(adata_8um.var_names))
adata = adata[:, common_genes].copy()

#remove duplicate
adata_8um = adata_8um[:, ~adata_8um.var_names.duplicated()].copy()
#adata_8um = adata_8um[:, common_genes].copy()

n_sample = 10000  # adjust based on memory
rng = np.random.default_rng(seed=42) 
idx_xenium = rng.choice(adata.n_obs, n_sample, replace=False)

adata_xen = adata[idx_xenium]
adata_vis = adata_8um


#DEG analysis
def impute_xenium(adata_vis,adata_xen,G0):
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
        var=adata_vis.var.copy()
    )
    return adata_xen_imputed

# getting DEG from class 0 and class 1,2,3 glomeruli from Glom_sclerosis_annotation.xlsx





with open('fpgw_moscot_init','rb') as f:
    G0 = pickle.load(f)
    
adata_vis_imputed1 = impute_xenium(adata_vis,adata_xen,G0.T)

selenop = adata_vis_imputed1[:, "SELENOP"].X

if not isinstance(selenop, np.ndarray):
    selenop = selenop.toarray()

selenop = selenop.flatten()

high_cut = np.quantile(selenop, 0.80)
low_cut = np.quantile(selenop, 0.20)

adata_vis_imputed1.obs["SELENOP_enriched"] = "Middle"

adata_vis_imputed1.obs.loc[
    selenop >= high_cut,
    "SELENOP_enriched"
] = "High"

adata_vis_imputed1.obs.loc[
    selenop <= low_cut,
    "SELENOP_enriched"
] = "Low"

adata_deg = adata_vis_imputed1[
    adata_vis_imputed1.obs["SELENOP_enriched"].isin(["High", "Low"])
].copy()

sc.tl.rank_genes_groups(
    adata_deg,
    groupby="SELENOP_enriched",
    groups=["High"],
    reference="Low",
    method="wilcoxon"
)

deg = sc.get.rank_genes_groups_df(adata_deg, group=["High"])
deg_filtered = deg[
    (deg['pvals_adj'] < 0.05) 
    &(abs(deg['logfoldchanges']) > 0.3)
]

deg_filtered.to_csv('DEGs_SELENOP.csv', index=False)
