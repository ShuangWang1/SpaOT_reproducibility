"""
spatial_plausibility.py

Compute spatially-aware alignment assessment metrics for Visium -> Xenium assignments.

Functions implemented:
 - ensure_row_normalized
 - gaussian_kernel
 - compute_wsim
 - compute_entropy_effective_count
 - estimate_cell_density
 - compute_expected_cell_count
 - compute_inside_fraction
 - compute_hull_ratio
 - compute_interspot_conflict
 - build_cell_graph
 - conductance_for_spot
 - morans_I
 - composite_plausibility

Includes:
 - demo_synthetic() to create a toy dataset and compare a "good" vs "bad" mapping
 - run_on_user_data() showing how to load P, coords, sim from files

"""

import scanpy as sc
import numpy as np


#adata_hd = sc.read_h5ad('outs_04/binned_outputs/square_008um/8um.h5ad')
adata_hd = sc.read_h5ad('outs_04/segmented_outputs/seg.h5ad')

adata_xen = sc.read_h5ad('Xenium_adata/IU04.h5ad')

from scipy.spatial.distance import cdist
    
import scipy.sparse as sp
def _to_numpy(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)
    
def per_gene_cosine_under_coupling(
    adata_source,   # e.g., Xenium (cells)
    adata_target,   # e.g., Visium (spots)
    P,              # shape (n_target, n_source)
    log1p=True,
    scale=True,
    eps=1e-12,
    return_pearson=False
):
    """
    Per-gene cosine similarity between adata_target.X and the barycentric
    projection of adata_source.X via coupling P.

    Returns:
        dict with:
          - 'genes': np.array of common gene names in order
          - 'cosine': np.array of per-gene cosine similarities
          - (optional) 'pearson': np.array of per-gene Pearson correlations
          - 'summary': dict with aggregates (mean/median)
    """
    a_src = adata_source.copy()
    a_tgt = adata_target.copy()
    a_src.var_names_make_unique()
    a_tgt.var_names_make_unique()

    if log1p:
        sc.pp.log1p(a_src); sc.pp.log1p(a_tgt)
    if scale:
        # scale per gene so cosine is not dominated by highly variable genes
        sc.pp.scale(a_src, zero_center=True, max_value=None)
        sc.pp.scale(a_tgt, zero_center=True, max_value=None)

    # Intersect genes and align order
    common = a_src.var_names.intersection(a_tgt.var_names)
    if len(common) == 0:
        raise ValueError("No shared genes between datasets.")
    common = common.sort_values()
    a_src = a_src[:, common].copy()
    a_tgt = a_tgt[:, common].copy()

    Xs = _to_numpy(a_src.X)  # (n_source, G)
    Xt = _to_numpy(a_tgt.X)  # (n_target, G)

    P = P.toarray() if sp.issparse(P) else np.asarray(P)
    n_t, n_s = Xt.shape[0], Xs.shape[0]
    if P.shape != (n_t, n_s):
        raise ValueError(f"P shape {P.shape} must be (n_target={n_t}, n_source={n_s}).")

    # Barycentric projection of source -> target
    row_sums = P.sum(axis=1, keepdims=True) + eps
    Xs_proj = (P @ Xs) / row_sums  # (n_target, G)

    # Per-gene cosine: cos(a,b) = (a·b) / (||a|| ||b||)
    num = (Xt * Xs_proj).sum(axis=0)
    Xt_norm = np.linalg.norm(Xt, axis=0) + eps
    Xp_norm = np.linalg.norm(Xs_proj, axis=0) + eps
    cos = num / (Xt_norm * Xp_norm)

    out = {
        "genes": np.asarray(common.values, dtype=str),
        "cosine": cos,
        "summary": {"mean_cosine": float(np.mean(cos)), "median_cosine": float(np.median(cos))}
    }

    if return_pearson:
        pear = np.array([pearsonr(Xt[:, g], Xs_proj[:, g])[0] for g in range(Xt.shape[1])])
        out["pearson"] = pear
        out["summary"].update(
            mean_pearson=float(np.nanmean(pear)),
            median_pearson=float(np.nanmedian(pear))
        )

    return out

import numpy as np
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix, ConvexHull
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

# ------------------ Utilities ------------------

def ensure_row_normalized(P, eps=1e-12):
    """Normalize rows of P to sum to 1. Returns float array."""
    P = np.asarray(P, dtype=float)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return P / (row_sums + eps)

def gaussian_kernel(d, sigma):
    """Elementwise gaussian kernel for distance array d and bandwidth sigma."""
    return np.exp(- (d**2) / (2.0 * float(sigma)**2))

# ------------------ Core metrics ------------------

def compute_wsim(P, sim, spot_coords, cell_coords, sigma=None):
    """
    Spatially weighted similarity per spot and global mean.
    P: n_spots x n_cells assignment weights (rows will be normalized inside).
    sim: n_spots x n_cells similarity matrix (e.g., cosine similarity).
    spot_coords: (n_spots,2)
    cell_coords: (n_cells,2)
    sigma: bandwidth for Gaussian spatial kernel. If None, estimated heuristically.
    Returns: (wsim_per_spot, wsim_global_mean, sigma_used)
    """
    P = ensure_row_normalized(P)
    d = distance_matrix(spot_coords, cell_coords)
    if sigma is None:
        # heuristic: median of minimal distances from spots to cells
        mins = np.min(d, axis=1)
        sigma = float(np.median(mins) if np.median(mins) > 0 else 1.0)
    w = gaussian_kernel(d, sigma)
    num = (P * sim * w).sum(axis=1)
    den = (P * w).sum(axis=1)
    den[den == 0] = 1e-12
    wsim_per_spot = num / den
    return wsim_per_spot, float(np.mean(wsim_per_spot)), sigma

def compute_entropy_effective_count(P):
    """
    Computes per-spot Shannon entropy H_i and effective count E_i = exp(H_i).
    Returns (H_array, E_array).
    """
    P = ensure_row_normalized(P)
    eps = 1e-12
    H = -np.sum(P * np.log(P + eps), axis=1)
    E = np.exp(H)
    return H, E

def estimate_cell_density(cell_coords, tissue_area=None):
    """
    Estimate cell density rho = n_cells / tissue_area.
    If tissue_area is None, approximate area by convex hull of cell coordinates.
    Returns (rho, tissue_area_used).
    """
    n_cells = cell_coords.shape[0]
    if tissue_area is None:
        if n_cells < 3:
            tissue_area = 1.0
        else:
            hull = ConvexHull(cell_coords)
            tissue_area = float(hull.volume)
    rho = float(n_cells) / float(tissue_area)
    return rho, tissue_area

def compute_expected_cell_count(spot_radius, rho):
    """
    Expected number of cells in a spot of given radius (or radii).
    spot_radius: scalar or array length n_spots
    rho: cells per unit area
    Returns (expected_count_array, spot_area_array)
    """
    spot_radius = np.asarray(spot_radius, dtype=float)
    area = np.pi * (spot_radius**2)
    expected = rho * area
    return expected, area

def compute_inside_fraction(P, spot_coords, cell_coords, spot_radius):
    """
    PF_i = sum_j p_ij * I(distance <= spot_radius_i)
    Returns (PF_array, outside_penalty_array = 1 - PF)
    """
    P = ensure_row_normalized(P)
    d = distance_matrix(spot_coords, cell_coords)
    # handle scalar or array radius
    spot_radius = np.asarray(spot_radius)
    if spot_radius.ndim == 0:
        inside = (d <= spot_radius).astype(float)
    else:
        inside = (d <= spot_radius[:, None]).astype(float)
    PF = (P * inside).sum(axis=1)
    outside_penalty = 1.0 - PF
    return PF, outside_penalty

def compute_hull_ratio(P, spot_coords, spot_radius, tau=0.01):
    """
    For each spot, take cells with p_ij > tau, compute their convex hull area,
    and return hull_area / spot_area. If fewer than 3 points selected, hull_area approximated small.
    Returns (hull_ratio_array, hull_area_array)
    """
    P = ensure_row_normalized(P)
    n_cells = P.shape[1]
    hull_ratios = np.zeros(n_cells)
    hull_areas = np.zeros(n_cells)
    spot_radius = np.asarray(spot_radius)
    for i in range(n_cells):
        idx = np.where(P[:, i] > tau)[0]
        if idx.size < 3:
            hull_area = 0.0
        else:
            try:
                hull = ConvexHull(spot_coords[idx])
                hull_area = float(hull.volume)
            except Exception:
                hull_area = 1e-8
        if spot_radius.ndim == 0:
            spot_area = np.pi * spot_radius**2
        else:
            spot_area = np.pi * (spot_radius[i]**2)
        hull_areas[i] = hull_area
        hull_ratios[i] = hull_area / (spot_area + 1e-12)
    return hull_ratios, hull_areas

def compute_interspot_conflict(P, cell_coords, cell_radius):
    """
    Compute per-cell conflict S_j (spread of contributing spot centers) normalized by typical spot radius.
    Also computes a global conflict scalar (mass-weighted).
    Returns (cell_conflict_array, global_conflict_scalar)
    """
    P = np.asarray(P, dtype=float)
    m_i = P.sum(axis=1) + 1e-12
    # weighted center of spots for each cell
    weighted_centers = (cell_coords.T @ P.T).T / m_i[:, None]
    n_spots = P.shape[0]
    S2 = np.zeros(n_spots)
    for i in range(n_spots):
        diffs = cell_coords - weighted_centers[i]
        sqd = np.sum(diffs**2, axis=1)
        S2[i] = np.sum(P[i, :] * sqd) / m_i[i]
    S = np.sqrt(S2)
    if np.ndim(cell_radius) == 0:
        rtyp = float(cell_radius)
    else:
        rtyp = float(np.median(cell_radius))
    cell_conflict = S / (rtyp + 1e-12)
    global_conflict = float(np.sum(m_i * np.maximum(0.0, cell_conflict - 1.0)) / np.sum(m_i))
    return cell_conflict, global_conflict

def build_cell_graph(cell_coords, k=8, sigma=None):
    """
    Build a symmetric kNN graph weighted by Gaussian kernel.
    Returns (W, sigma_used) where W is dense numpy array of shape (n_cells,n_cells).
    """
    knn = kneighbors_graph(cell_coords, n_neighbors=k, mode='connectivity', include_self=False)
    A = knn.toarray().astype(float)
    D = distance_matrix(cell_coords, cell_coords)
    if sigma is None:
        dvals = D[A == 1]
        sigma = float(np.median(dvals) if dvals.size > 0 else 1.0)
    W = np.exp(- (D**2) / (2.0 * sigma**2)) * A
    W = (W + W.T) / 2.0
    return W, sigma

def conductance_for_spot(P, cell_coords, k=8, tau=0.05):
    """
    For each spot, define assigned set S_i = {j: p_ij > tau}, compute conductance phi(S_i)
    on a kNN graph of cells. Returns (phi_array, sigma_used_for_graph).
    """
    W, sigma = build_cell_graph(cell_coords, k=k)
    d = W.sum(axis=1)
    n_spots = P.shape[0]
    phis = np.zeros(n_spots)
    for i in range(n_spots):
        S_idx = np.where(P[i] > tau)[0]
        if S_idx.size == 0:
            phis[i] = 1.0
            continue
        mask = np.zeros(W.shape[0], dtype=bool)
        mask[S_idx] = True
        cut = W[np.ix_(mask, ~mask)].sum()
        volS = d[mask].sum()
        volComp = d[~mask].sum()
        denom = min(volS, volComp) + 1e-12
        phis[i] = float(cut / denom)
    return phis, sigma

def morans_I(residuals, cell_coords, W=None, k=8):
    """
    Compute Moran's I statistic for residuals over a cell adjacency graph.
    residuals: vector length n_cells
    If W is None, build kNN W.
    Returns scalar I.
    """
    res = np.asarray(residuals, dtype=float)
    n = res.size
    if W is None:
        W, _ = build_cell_graph(cell_coords, k=k)
    Wsum = float(W.sum())
    if Wsum == 0:
        return 0.0
    res_mean = res.mean()
    num = float(((W * np.outer(res - res_mean, res - res_mean))).sum())
    den = float(((res - res_mean)**2).sum()) + 1e-12
    I = (n / Wsum) * (num / den)
    return float(I)

# ------------------ Composite plausibility ------------------

def composite_plausibility(wsim, outside_penalty, CR, conductance, weights=None):
    """
    Combine normalized components into a plausibility score in [0,1].
    Inputs: arrays of length n_spots
    Returns: plaus_array and dict of normalized components.
    """
    a = np.asarray(wsim, dtype=float)
    # normalize wsim min-max
    if (a.max() - a.min()) < 1e-12:
        a_n = np.ones_like(a) * 0.5
    else:
        a_n = (a - a.min()) / (a.max() - a.min())
    b_n = 1.0 - np.clip(outside_penalty, 0.0, 1.0)
    beta = 1.0
    c_n = np.exp(-beta * np.clip(np.asarray(CR, dtype=float) - 1.0, 0.0, None))
    d = np.asarray(conductance, dtype=float)
    med = float(np.median(d) + 1e-12)
    d_n = 1.0 / (1.0 + d / med)
    if weights is None:
        weights = np.array([0.25, 0.25, 0.25, 0.25])
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    plaus = weights[0]*a_n + weights[1]*b_n + weights[2]*c_n + weights[3]*d_n
    comps = {'wsim_n': a_n, 'inside_n': b_n, 'cr_n': c_n, 'cond_n': d_n}
    return plaus, comps

def run_on_user_data(P, spot_coords, cell_coords, sim=None, spot_radius=None,cell_radius=None, save_csv_prefix=None):
    """
    Run all metrics on user-supplied data.
    - P: n_spots x n_cells assignment matrix (rows will be normalized)
    - spot_coords: (n_spots,2)
    - cell_coords: (n_cells,2)
    - sim: optional n_spots x n_cells similarity matrix. If None, WSim will be skipped.
    - spot_radius: scalar or array (n_spots). If None, a heuristic radius is used.
    - cell_radius: scalar or array (n_cells). If None, a heuristic radius is used.
    - save_csv_prefix: if provided, writes {prefix}_spot_metrics.csv and {prefix}_cell_metrics.csv
    Returns a dict with dataframes and global metrics.
    """
    n_spots = int(P.shape[0])
    n_cells = int(P.shape[1])
    if spot_radius is None:
        # heuristic: median distance between neighboring spots
        if n_spots > 1:
            sd = distance_matrix(spot_coords, spot_coords)
            np.fill_diagonal(sd, np.inf)
            spot_radius = float(np.median(np.min(sd, axis=1)))
        else:
            spot_radius = 5.0
            
    
    if cell_radius is None:
        # heuristic: median distance between neighboring cells
        if n_cells > 1:
            sd = distance_matrix(cell_coords, cell_coords)
            np.fill_diagonal(sd, np.inf)
            cell_radius = float(np.median(np.min(sd, axis=1)) )
        else:
            cell_radius = 5.0

    results = {}
    Pnorm = ensure_row_normalized(P)
    if sim is not None:
        wsim_per_spot, wsim_global, sigma = compute_wsim(Pnorm, sim, spot_coords, cell_coords, sigma=None)
    else:
        wsim_per_spot = np.full(n_spots, np.nan)
        wsim_global, sigma = np.nan, None

    H, E = compute_entropy_effective_count(Pnorm)
    rho, tissue_area = estimate_cell_density(cell_coords)
    expected_n, spot_area = compute_expected_cell_count(spot_radius, rho)
    CR = E / (expected_n + 1e-12)
    #PF, outside_penalty = compute_inside_fraction(Pnorm, spot_coords, cell_coords, spot_radius)
    hull_ratios, hull_areas = compute_hull_ratio(Pnorm, spot_coords, spot_radius, tau=0.01)
    cell_conflict, global_conflict = compute_interspot_conflict(Pnorm, cell_coords, cell_radius)
    #phis, graph_sigma = conductance_for_spot(Pnorm, cell_coords, k=8, tau=0.05)

    # For Moran's I we need cell-level predicted expression/residuals. If sim provided, predict a scalar profile:
    # if sim is not None:
    #     # Simple predicted cell scalar: P.T @ (spot-wise mean similarity) as proxy
    #     spot_scalar = np.nanmean(sim, axis=1)
    #     predicted_cell_scalar = Pnorm.T @ spot_scalar
    #     # residuals relative to actual if user provides a cell-level scalar (not available); skip if not useful
    #     moran_I_val = morans_I(predicted_cell_scalar - predicted_cell_scalar.mean(), cell_coords, W=None, k=8)
    # else:
    #     predicted_cell_scalar = np.zeros(n_cells)
    #     moran_I_val = np.nan

    #plaus, comps = composite_plausibility(wsim_per_spot, outside_penalty, CR, phis)

    spot_df = pd.DataFrame({
        #'wsim': wsim_per_spot,
        'H': H,
        'E': E,
        'expected_n': expected_n,
        'CR': CR,
        #'PF': PF,
        #'outside_penalty': outside_penalty,
        #'conductance': phis,
        #'plausibility': plaus,
        'conflict': cell_conflict
    })
    cell_df = pd.DataFrame({
        'cell_x': cell_coords[:, 0],
        'cell_y': cell_coords[:, 1],
        'hull_area': hull_areas,
        'hull_ratio': hull_ratios
    })

    # global_metrics = {
    #     'wsim_global': wsim_global,
    #     'sigma_wsim': sigma,
    #     'rho': rho,
    #     'tissue_area': tissue_area,
    #     'global_conflict': global_conflict,
    #     #'moran_I_predicted_scalar': moran_I_val,
    #     #'graph_sigma': graph_sigma
    # }

    if save_csv_prefix is not None:
        spot_df.to_csv(f'{save_csv_prefix}_spot_metrics.csv', index=False)
        cell_df.to_csv(f'{save_csv_prefix}_cell_metrics.csv', index=False)

    results['spot_df'] = spot_df
    results['cell_df'] = cell_df
    #results['global'] = global_metrics
    #results['Pnorm'] = Pnorm
    return results

# ------------------ If executed as main, run demo and produce plots ------------------


adata_hd_seg = sc.read_h5ad('outs_04/segmented_outputs/seg.h5ad')
coords = adata_hd_seg.obsm['spatial']
source_glom_labels = pd.read_csv('Xenium_and_visium_glom_annotation/IU04_visium_Gloms.csv')    

source_glom_labels = source_glom_labels.set_index('Barcode')

adata_hd_seg.obs["glomerulus"] = (
    source_glom_labels["Gloms"]
    .reindex(adata_hd_seg.obs_names)
    .fillna("None")
)
import pickle

source_glom_labels = pd.read_csv('Xenium_and_visium_glom_annotation/IU04_visium_Gloms.csv')    

glom_poly = pd.read_csv('Xenium_and_visium_glom_annotation/IU04_xenium_gloms.csv')

from shapely.geometry import Polygon

polygons = {}

for sel, group in glom_poly.groupby("Selection"):
    coords = group[["X", "Y"]].values
    polygons[sel] = Polygon(coords)
    
cell_coords = adata_xen.obsm["spatial"]

from shapely.geometry import Point
import numpy as np

cell_glom = np.array(["None"] * adata_xen.n_obs, dtype=object)

for i, (x, y) in enumerate(cell_coords):
    point = Point(x, y)
    
    for sel, poly in polygons.items():
        if poly.contains(point):
            cell_glom[i] = sel
            break  # stop after first match
        
adata_xen.obs["glomerulus"] = cell_glom

#region_dict = {0:3,1:1,2:0,3:2}
# glom for seg
region_dict = {0:3,1:2,2:0,3:1}

for region in [0,1,2,3]:
    
    with open(f'xen_hd_map/{region}_moscot_seg.pickle','rb') as f:
        mos = pickle.load(f)
    G_mos = np.array(mos.transport_matrix)
    
    
    with open(f'xen_hd_map/{region}_moscot_init_seg.pickle','rb') as f:
        G0 = pickle.load(f)
        


    adata_X = adata_hd[adata_hd.obs["spatial_region"] == region_dict[region]].copy()
    adata_Y = adata_xen[adata_xen.obs["spatial_region"] == region].copy()

    df1 = run_on_user_data(G0.T, adata_X.obsm['spatial'], adata_Y.obsm['spatial'], sim=None, spot_radius=None, save_csv_prefix=None)
    
    with open(f'Assessment/{region}_fpgw_seg_newHull&CellConflict.pkl','wb') as f:
        pickle.dump(df1,f)
        
    df2 = run_on_user_data(G_mos.T, adata_X.obsm['spatial'], adata_Y.obsm['spatial'], sim=None, spot_radius=None, save_csv_prefix=None)
    
    with open(f'Assessment/{region}_moscot_seg_newHull&CellConflict.pkl','wb') as f:
        pickle.dump(df2,f)