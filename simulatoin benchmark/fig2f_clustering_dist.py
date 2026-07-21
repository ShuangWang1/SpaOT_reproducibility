import os, sys
resolved_path = os.path.realpath('..')
sys.path.append(resolved_path)
import time
import numpy as np

# ── Global seeds — set before any library that draws random numbers ───────────
SEED = 42
np.random.seed(SEED)
import torch
torch.manual_seed(SEED)
from lib.fgw.graph import Graph
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import adjusted_mutual_info_score
from lib.graph_clustering import *
import copy
import networkx as nx


X, y, V = build_dataset(5, type_list=[1, 2, 3],seed1=42,seed2=42)
for x in X:
    x.N = len(x.values())
X = np.array(X)

X1 = [copy.deepcopy(x) for x in X]
per = 0.5
num_graphs = int(len(X) * per)
eta = 0.3
for i in range(num_graphs):
    x1 = X1[i]
    add_outliers(x1, eta=eta, gap=4, seed=SEED + i)
print(x1.nodes().keys())
X1 = np.array(X1)

# Precompute all shortest-path distances
for x in X:
    C = x.distance_matrix(method='shortest_path', force_recompute=True)
for x in X1:
    x.distance_matrix(method='shortest_path', force_recompute=True)
X1 = np.array(X1)
connected = []
for x in X1:
    connected.append(nx.is_connected(x.nx_graph))

alpha = 0.50

def reorder(lst, k=3):
    result = []
    for i in range(k):
        result.extend(lst[i::k])
    return result

lst = list(range(15))
print(reorder(lst))

X1 = reorder(X1)
structural_information = [x.distance_matrix(method='shortest_path', force_recompute=True) for x in X1]





def _struct_to_spatial(C, n_components=2):
    """Derive 2-D spatial coordinates from a structure/cost matrix via MDS."""
    from sklearn.manifold import MDS
    C = np.array(C, dtype=float)
    C = np.clip(C, 0, None)          # ensure non-negative
    np.fill_diagonal(C, 0)
    denom = C.max()
    C_norm = C / (denom + 1e-8)
    return MDS(n_components=n_components, dissimilarity='precomputed',
               random_state=0, n_init=1).fit_transform(C_norm)
    
def cdist_fpgw(X_features, X_structure, Y_features, Y_structure, alpha,metric='sqeuclidean',measure='fgw',N_list=None):

    n_X = len(X_features)
    n_Y = len(Y_features)

    # TMP #
    for i in range(n_X):
        assert np.linalg.norm(X_structure[i] - X_structure[i].T) < 1e-5, "Wooops X"
    for j in range(n_Y):
        assert np.linalg.norm(Y_structure[j] - Y_structure[j].T) < 1e-5, "Wooops Y"


    dists = np.empty((n_X, n_Y))
    for i in range(n_X):
        for j in range(n_Y):
            dist_features_ij = cdist(np.array(X_features[i]).reshape((len(X_features[i]), -1)),
                                     np.array(Y_features[j]).reshape((len(Y_features[j]), -1)),
                                     metric=metric)
            C1,C2=X_structure[i],Y_structure[j]
            if measure=='fgw':
                p = np.ones(len(X_features[i])) / len(X_features[i])
                q = np.ones(len(Y_features[j])) / len(Y_features[j])
                dists[i, j] = ot.gromov.fused_gromov_wasserstein2(
                    dist_features_ij, C1, C2, p, q,
                    loss_fun='square_loss', alpha=alpha, log=False
                )
   
            if measure=='fmpgw':
                p=np.ones(len(X_features[i]))/N_list[i]
                q=np.ones(len(Y_features[j]))/N_list[j] # Transport all mass of Y (centers)
                #mass=min(p.sum(),q.sum())
                #dist_features_ij = (dist_features_ij - dist_features_ij.min()) / (dist_features_ij.max() - dist_features_ij.min() + 1e-8)
                C_max = max(C1.max(), C2.max())
                C2 = C2 / (C_max + 1e-8)
                C1 = C1 / (C_max + 1e-8)
                transport= fused_partial_gromov_wasserstein(dist_features_ij,C1,C2,p,q,omega2=alpha,mass=None,
                                                                  loss_fun='square_loss',verbose=False,log=False,Lambda=1.0)
                cost=fused_pgw_cost(dist_features_ij,C1,C2,transport,alpha,loss_fun='square_loss')
                dists[i,j]=cost
                
            if measure=='fugw':
                p=np.ones(len(X_features[i]))/N_list[i]
                q=np.ones(len(Y_features[j]))/N_list[j] # Transport all mass of Y (centers)
                #mass=min(p.sum(),q.sum())
                rho,eps=1.0,0.05

                gamma,_,log=ot.gromov.fused_unbalanced_gromov_wasserstein(Cx=C1, Cy=C2,wx=p,wy=q,reg_marginals=rho, epsilon=eps,
                                                                divergence="kl",unbalanced_solver="mm",alpha=alpha,M=dist_features_ij,log=True)
                cost=log['fugw_cost']
                dists[i,j]=cost

            if measure=='moscot':
                import anndata as ad
                import pandas as pd
                import scanpy as sc
                import warnings
                warnings.filterwarnings('ignore')
                from moscot.problems.space import AlignmentProblem
                fi = np.array(X_features[i]).reshape(-1, 1)
                fj = np.array(Y_features[j]).reshape(-1, 1)
                d1 = ad.AnnData(X=fi, var=pd.DataFrame(index=['feat_0']))
                d1.obsm['spatial'] = _struct_to_spatial(C1)
                d2 = ad.AnnData(X=fj, var=pd.DataFrame(index=['feat_0']))
                d2.obsm['spatial'] = _struct_to_spatial(C2)
                adata = sc.concat({'src': d1, 'tgt': d2},
                                  label='dataset', join='inner', merge='same')
                ap = AlignmentProblem(adata=adata)
                ap = ap.prepare(batch_key='dataset', policy='sequential')
                ap = ap.solve()
                dists[i, j] = float(ap[('src', 'tgt')].solution.cost)

            if measure=='paste2':
                import anndata as ad
                import pandas as pd
                import warnings
                warnings.filterwarnings('ignore')
                from paste2 import PASTE2
                fi = np.array(X_features[i]).reshape(-1, 1)
                fj = np.array(Y_features[j]).reshape(-1, 1)
                d1 = ad.AnnData(X=fi, var=pd.DataFrame(index=['feat_0']))
                d1.obsm['spatial'] = _struct_to_spatial(C1)
                d2 = ad.AnnData(X=fj, var=pd.DataFrame(index=['feat_0']))
                d2.obsm['spatial'] = _struct_to_spatial(C2)
                s = len(X_features[i])/N_list[i] if len(X_features[i])/N_list[i] <= 1 else 1.0
                _, cost = PASTE2.partial_pairwise_align(
                    d1, d2, s=s, return_obj=True,
                    dissimilarity='euclidean', verbose=False)
                dists[i, j] = float(cost)

    return dists


for m in ['fmpgw','fgw','moscot','paste2']:
    dists = cdist_fpgw(X_features=[x.values() for x in X1],
                        X_structure=structural_information,
                        Y_features=[x.values() for x in X1],
                        Y_structure=structural_information,
                        alpha=0.5,
                        N_list=[x.N for x in X1],
                        measure=m)

    with open(f'graph_cluster_dist/{m}_alpha_0.5.npy', 'wb') as f:
        np.save(f, dists)
        
