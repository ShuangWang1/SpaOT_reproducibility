import sys
sys.path.append("..")

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ST_PATH= "xenium_p2/xenium.h5ad"
MSI_PATH= "visium_p2/visium.h5ad"

st = ad.read_h5ad(ST_PATH)
msi = ad.read_h5ad(MSI_PATH)

#remove duplicate
msi = msi[:, ~msi.var_names.duplicated()].copy()

n_sample = 10000  # adjust based on memory
rng = np.random.default_rng(seed=42) 
idx_xenium = rng.choice(st.n_obs, n_sample, replace=False)

st = st[idx_xenium]

common_genes = list(set(st.var_names).intersection(msi.var_names))
st = st[:, common_genes].copy()
msi = msi[:, common_genes].copy()

st.obs["dataset"] = "xenium"
msi.obs["dataset"] = "visium"

import scanpy as sc
adata = sc.concat(
    {"xenium": st, "visium": msi},
    label="technology",
    join="inner",   # already inner, but safe
    merge="same"
)

from moscot.problems.space import AlignmentProblem

ap = AlignmentProblem(adata=adata)
ap = ap.prepare(batch_key="dataset", policy="sequential")

pr = ap.problems[('visium','xenium')]

import os,sys
resolved_path = os.path.realpath('..')
sys.path.append(resolved_path+'/moscot-framework_reproducibility')

from scipy.spatial.distance import cdist

from lib.fused_pgw import fused_partial_gromov_wasserstein, fused_partial_gromov_wasserstein_mass

M = cdist(pr.xy.data_src, pr.xy.data_tgt, metric='euclidean')
M = (M-M.min())/(M.max()-M.min())
ref_structure = cdist(pr.x.data_src, pr.x.data_src, metric='euclidean')
tgt_structure = cdist(pr.y.data_src, pr.y.data_src, metric='euclidean')

G0=fused_partial_gromov_wasserstein(M,ref_structure,tgt_structure,p=pr.a,q=pr.b,omega2=0.5, Lambda = 1.0)


import pickle
with open('fpgw_moscot_init.pickle','wb') as f:
    pickle.dump(G0,f)
    

def plot_match(adata_X, adata_Y, G0, top_k=2000, gap=50,
               rotate_deg=0, mirror_axis=None):

    # 1. Extract spatial coordinates
    XY_X = adata_X.obsm['spatial'].copy()
    XY_Y = adata_Y.obsm['spatial'].copy()
    
    if rotate_deg != 0:
        center = XY_Y.mean(axis=0)
        XY_Y_centered = XY_Y - center

        if rotate_deg == 90:
            XY_Y_rot = np.zeros_like(XY_Y_centered)
            XY_Y_rot[:,0] = -XY_Y_centered[:,1]  # x_new = -y
            XY_Y_rot[:,1] = XY_Y_centered[:,0]   # y_new = x
        elif rotate_deg == 180:
            XY_Y_rot = -XY_Y_centered
        elif rotate_deg == 270:
            XY_Y_rot = np.zeros_like(XY_Y_centered)
            XY_Y_rot[:,0] = XY_Y_centered[:,1]   # x_new = y
            XY_Y_rot[:,1] = -XY_Y_centered[:,0]  # y_new = -x
        else:
            raise ValueError("rotate_deg must be 0, 90, 180, or 270")

        XY_Y = XY_Y_rot + center

    # 3. Mirror if requested
    if mirror_axis == 'x':
        center_x = XY_Y[:,0].mean()
        XY_Y[:,0] = 2*center_x - XY_Y[:,0]
    elif mirror_axis == 'y':
        center_y = XY_Y[:,1].mean()
        XY_Y[:,1] = 2*center_y - XY_Y[:,1]

    # 2. Rescale XY_Y to match XY_X scale
    # Linear scaling by bounding box width
    minX_X, maxX_X = XY_X[:,0].min(), XY_X[:,0].max()
    minY_X, maxY_X = XY_X[:,1].min(), XY_X[:,1].max()
    
    minX_Y, maxX_Y = XY_Y[:,0].min(), XY_Y[:,0].max()
    minY_Y, maxY_Y = XY_Y[:,1].min(), XY_Y[:,1].max()
    
    scale_x = (maxX_X - minX_X) / (maxX_Y - minX_Y)
    scale_y = (maxY_X - minY_X) / (maxY_Y - minY_Y)
    
    XY_Y[:,0] = (XY_Y[:,0] - minX_Y) * scale_x
    XY_Y[:,1] = (XY_Y[:,1] - minY_Y) * scale_y

    # 3. Shift XY_Y to the right for visualization
    XY_Y[:,0] += (maxX_X - minX_X) + gap

    # 4. Plot cells
    plt.figure(figsize=(18, 12))
    plt.scatter(XY_X[:,0], XY_X[:,1], c='blue', s=5, label='Xenium')
    plt.scatter(XY_Y[:,0], XY_Y[:,1], c='red', s=5, label='Visium')

    # 5. Select top-k transport edges
    flat_indices = np.argsort(G0.flatten())[::-1][:top_k]
    rows, cols = np.unravel_index(flat_indices, G0.shape)
    weights = G0[rows, cols]
    weights_alpha = weights / weights.max()

    # 6. Draw transport lines
    for i, j, alpha in zip(rows, cols, weights_alpha):
        plt.plot(
            [XY_X[i,0], XY_Y[j,0]],
            [XY_X[i,1], XY_Y[j,1]],
            c='gray',
            alpha=alpha*0.7,
            linewidth=0.5
        )

    # 7. Final figure adjustments
    plt.gca().invert_yaxis()  # spatial convention
    plt.axis('off')
    plt.legend()
    plt.title("Transport plan between Xenium and Visium")
    plt.savefig('fpgw_plan_mass0.5.png',dpi=300)

#plot_match(adata_X, adata_Y, G0, top_k=500, gap=5000,rotate_deg=270)