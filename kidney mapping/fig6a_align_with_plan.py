import numpy as np
import copy

def align_adata_with_transport(
    adata_source,
    adata_target,
    P,
    spatial_key="spatial",
    allow_scaling=False
):
    """
    Align source AnnData to target AnnData using a transport plan.

    Parameters
    ----------
    adata_source : AnnData
    adata_target : AnnData
    P : (n_source, n_target) transport plan
    spatial_key : key in .obsm containing spatial coordinates
    allow_scaling : whether to estimate similarity transform (scale + rotation)

    Returns
    -------
    adata_aligned : aligned copy of adata_source
    transform : dict containing R, b, (optional s)
    """

    X = np.asarray(adata_source.obsm[spatial_key])
    Y = np.asarray(adata_target.obsm[spatial_key])

    # weights per source point
    w = P.sum(axis=1)
    w = w + 1e-12

    # barycentric projection of source onto target
    Y_bar = (P @ Y) / w[:, None]

    # weighted centroids
    mu_X = np.average(X, axis=0, weights=w)
    mu_Y = np.average(Y_bar, axis=0, weights=w)

    Xc = X - mu_X
    Yc = Y_bar - mu_Y

    # weighted covariance
    H = (Xc * w[:, None]).T @ Yc

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # ----- automatic reflection detection -----
    # if np.linalg.det(R) < 0:
    #     print('need to flip')
    #     Vt[-1, :] *= -1
    #     R = Vt.T @ U.T

    # optional scaling
    if allow_scaling:
        var_X = np.sum(w * np.sum(Xc**2, axis=1))
        s = np.sum(S) / (var_X + 1e-12)
    else:
        s = 1.0

    # translation
    b = mu_Y - s * (R @ mu_X)

    # apply transform
    X_aligned = (s * (R @ X.T)).T + b

    # return aligned copy
    adata_aligned = copy.deepcopy(adata_source)
    adata_aligned.obsm[spatial_key] = X_aligned

    transform = {"R": R, "b": b, "scale": s}

    return adata_aligned, transform




def align_adata_spatial_with_transport(
    X,
    Y,
    P,
    allow_scaling=False
):
    """
    Align source AnnData to target AnnData using a transport plan.

    Parameters
    ----------
    adata_source : AnnData
    adata_target : AnnData
    P : (n_source, n_target) transport plan
    spatial_key : key in .obsm containing spatial coordinates
    allow_scaling : whether to estimate similarity transform (scale + rotation)

    Returns
    -------
    adata_aligned : aligned copy of adata_source
    transform : dict containing R, b, (optional s)
    """
    # weights per source point
    w = P.sum(axis=1)
    w = w + 1e-12

    # barycentric projection of source onto target
    Y_bar = (P @ Y) / w[:, None]

    # weighted centroids
    mu_X = np.average(X, axis=0, weights=w)
    mu_Y = np.average(Y_bar, axis=0, weights=w)

    Xc = X - mu_X
    Yc = Y_bar - mu_Y

    # weighted covariance
    H = (Xc * w[:, None]).T @ Yc

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # optional scaling
    if allow_scaling:
        var_X = np.sum(w * np.sum(Xc**2, axis=1))
        s = np.sum(S) / (var_X + 1e-12)
    else:
        s = 1.0

    # translation
    b = mu_Y - s * (R @ mu_X)

    # apply transform
    X_aligned = (s * (R @ X.T)).T + b

    return X_aligned, Y


import scanpy as sc

adata_hd = sc.read_h5ad('outs_04/segmented_outputs/seg.h5ad')
flag_8um = False

adata_xen = sc.read_h5ad('Xenium_adata/IU04.h5ad')



#use biopsy sample #0 as an example to make the alignment with transport plan
region_dict = {0:3,1:2,2:0,3:1}
adata_source = adata_hd[adata_hd.obs["spatial_region"] == region_dict[0]].copy()
adata_target = adata_xen[adata_xen.obs["spatial_region"] == 0].copy()



import pickle
with open('xen_hd_map/0_spaot_new.pickle','rb') as f:
    P = pickle.load(f)

import os,sys
resolved_path = os.path.realpath('..')
sys.path.append(resolved_path+'/moscot-framework_reproducibility')

adata_aligned, transform = align_adata_with_transport(
    adata_source,
    adata_target,
    P.T,
    allow_scaling=True
)



import matplotlib.pyplot as plt
import numpy as np

def plot_alignment(
    adata_aligned,
    adata_target,
    spatial_key="spatial",
    source_color="blue",
    target_color="lightblue",
    source_label="Aligned Source",
    target_label="Target",
    point_size=10,
    alpha=0.7
):
    """
    Plot aligned source and target spatial coordinates together.
    """

    X = np.asarray(adata_aligned.obsm[spatial_key])
    Y = np.asarray(adata_target.obsm[spatial_key])

    plt.figure(figsize=(6, 6))

    plt.scatter(
        Y[:, 0], Y[:, 1],
        s=point_size,
        c=target_color,
        alpha=alpha,
        label=target_label
    )

    plt.scatter(
        X[:, 0], X[:, 1],
        s=point_size,
        c=source_color,
        alpha=alpha,
        label=source_label
    )

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Spatial Alignment")
    plt.tight_layout()
    plt.show()

plot_alignment(
    adata_aligned,
    adata_target,
    spatial_key="spatial",
    source_color="red",
    target_color="blue",
    source_label="Aligned Source",
    target_label="Target",
    point_size=10,
    alpha=0.7
)