import sys
sys.path.append("..")

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pickle

import scanpy as sc
from scipy.spatial.distance import cdist

flag_8um = True
hd_flag = 'seg'

adata_hd = sc.read_h5ad('outs_04/segmented_outputs/seg.h5ad') 
adata_hd.var_names_make_unique()

adata_xen = sc.read_h5ad('Xenium_adata/IU04.h5ad')


import os,sys
resolved_path = os.path.realpath('..')
sys.path.append(resolved_path+'/moscot-framework_reproducibility')

from scipy.spatial.distance import cdist

from lib.fused_pgw import fused_partial_gromov_wasserstein, fused_partial_gromov_wasserstein_mass


   
for region in range(40):
    
    if f'G{region}' not in adata_hd.obs['glomerulus'].unique():
        continue

    adata_src = adata_hd[adata_hd.obs["glomerulus"] == f'G{region}'].copy()
    adata_tgt = adata_xen[adata_xen.obs["glomerulus"] == f'Selection {region}'].copy()
    
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

    pr = ap.problems[('visium','xenium')]

    M = cdist(pr.xy.data_src, pr.xy.data_tgt, metric='euclidean')
    M = (M-M.min())/(M.max()-M.min())
    #M = M.astype(np.float32)


    n,m = M.shape
    # p=np.ones(n)/max(n,m)
    # q=np.ones(m)/max(n,m)
    p=np.ones(n)/n
    q=np.ones(m)/m
    
    # p = p.astype(np.float32)
    # q = q.astype(np.float32)

    ref_structure = cdist(pr.x.data_src, pr.x.data_src, metric='euclidean')
    tgt_structure = cdist(pr.y.data_src, pr.y.data_src, metric='euclidean')
    ref_structure = (ref_structure-ref_structure.min())/(ref_structure.max()-ref_structure.min())
    tgt_structure = (tgt_structure-tgt_structure.min())/(tgt_structure.max()-tgt_structure.min())

    G0=fused_partial_gromov_wasserstein(M,ref_structure,tgt_structure,p=p,q=q,omega2=0.5, Lambda = 1.0)

    with open(f"xen_hd_map/glom{region}_spaot_seg.pickle", 'wb') as f:
        pickle.dump(G0,f)

