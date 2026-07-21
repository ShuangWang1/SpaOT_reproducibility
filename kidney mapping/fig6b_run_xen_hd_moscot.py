import sys
sys.path.append("..")

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pickle

import scanpy as sc
from scipy.spatial.distance import cdist

adata_hd = sc.read_h5ad('outs_04/binned_outputs/square_008um/8um.h5ad')
flag_8um = True
hd_flag = 'seg'

adata_hd = sc.read_h5ad('outs_04/segmented_outputs/seg.h5ad') 
adata_hd.var_names_make_unique()

adata_xen = sc.read_h5ad('Xenium_adata/IU04.h5ad')


coords = adata_hd.obsm["spatial"]

from sklearn.cluster import DBSCAN
# Run DBSCAN
# clustering = DBSCAN(
#     eps=200,      # distance threshold (adjust if needed)
#     min_samples=50
# ).fit(coords)

# labels = clustering.labels_
# adata_hd.obs["spatial_region"] = labels


coords = adata_xen.obsm["spatial"]

clustering = DBSCAN(
    eps=200,      # distance threshold (adjust if needed)
    min_samples=50
).fit(coords)

labels = clustering.labels_
adata_xen.obs["spatial_region"] = labels



import os,sys
resolved_path = os.path.realpath('..')
sys.path.append(resolved_path+'/moscot-framework_reproducibility')

from scipy.spatial.distance import cdist

from lib.fused_pgw import fused_partial_gromov_wasserstein, fused_partial_gromov_wasserstein_mass



region_dict = {0:3,1:1,2:0,3:2}
if hd_flag == 'seg':
    region_dict = {0:3,1:2,2:0,3:1}
    
for region in range(4):
    if flag_8um or hd_flag == 'seg':
        region_src = region_dict[region]
    else:
        region_src = region
    adata_src = adata_hd[adata_hd.obs["spatial_region"] == region_src].copy()
    adata_tgt = adata_xen[adata_xen.obs["spatial_region"] == region].copy()
    
    common_gene = list(set(adata_src.var_names).intersection(adata_tgt.var_names))
    adata_src = adata_src[:,common_gene]
    adata_tgt = adata_tgt[:,common_gene]
    
    import scanpy as sc
    sc.pp.normalize_total(adata_src, target_sum=1e4)
    sc.pp.normalize_total(adata_tgt, target_sum=1e4)

    sc.pp.log1p(adata_src)
    sc.pp.log1p(adata_tgt)
    
    
    

    adata_src.obs["dataset"] = "xenium"
    adata_tgt.obs["dataset"] = "visium"

    import scanpy as sc
    adata = sc.concat(
        {"xenium": adata_src, "visium": adata_tgt},
        label="technology",
        join="inner",   # already inner, but safe
        merge="same"
    )

    from moscot.problems.space import AlignmentProblem

    ap = AlignmentProblem(adata=adata)
    ap = ap.prepare(batch_key="dataset", policy="sequential")


    ap = ap.solve()
    G = ap[('visium','xenium')].solution
    
    with open(f"xen_hd_map/{region}_moscot_seg.pickle", 'wb') as f:
        pickle.dump(G,f)

