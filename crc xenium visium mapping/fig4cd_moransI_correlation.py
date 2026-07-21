import scanpy as sc
import numpy as np

import squidpy as sq

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import pickle
import seaborn as sns

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

sq.gr.spatial_neighbors(adata_X)
sq.gr.spatial_autocorr(
    adata_X,
    mode="moran",
    genes=adata_X.var_names,
    n_perms=100,
    n_jobs=1,
)
adata_X.uns["moranI"].head(10)

xen_moran = {}
xen_moran['original'] = adata_X.uns["moranI"]


#load moscot transport plan and compute morans I for xenium after mapping
with open('moscot_plan.pickle','rb') as f:
    mos = pickle.load(f)
G_mos = np.array(mos.transport_matrix)

row_sums = G_mos.T.sum(axis=1, keepdims=True) 
Xs_proj = (G_mos.T @ adata_Y.X) / row_sums  

adata_X_mos = adata_X.copy()
adata_X_mos.X = Xs_proj

sq.gr.spatial_neighbors(adata_X_mos)
sq.gr.spatial_autocorr(
    adata_X_mos,
    mode="moran",
    genes=adata_X_mos.var_names,
    n_perms=100,
    n_jobs=1,
)
xen_moran['moscot'] = adata_X_mos.uns["moranI"]


#load fpgw transport plan
with open('fpgw_moscot_init','rb') as f:
    G0 = pickle.load(f)

row_sums = G0.T.sum(axis=1, keepdims=True) 
Xs_proj = (G0.T @ adata_Y.X) / row_sums  

adata_X_fpgw = adata_X.copy()
adata_X_fpgw.X = Xs_proj

sq.gr.spatial_neighbors(adata_X_fpgw)
sq.gr.spatial_autocorr(
    adata_X_fpgw,
    mode="moran",
    genes=adata_X_fpgw.var_names,
    n_perms=100,
    n_jobs=1,
)
xen_moran['fpgw'] = adata_X_fpgw.uns["moranI"]

xen_moran['moscot'] = xen_moran['moscot'].sort_index()
xen_moran['fpgw'] = xen_moran['fpgw'].sort_index()
xen_moran['original'] = xen_moran['original'].sort_index()

tmp = xen_moran['original']['I'].tolist()
tmp1 = xen_moran['fpgw']['I'].tolist()
tmp2 = xen_moran['moscot']['I'].tolist()

np.corrcoef(tmp2, tmp)[0, 1]
np.corrcoef(tmp1, tmp)[0, 1]


sq.gr.spatial_neighbors(adata_Y)
sq.gr.spatial_autocorr(
    adata_Y,
    mode="moran",
    genes=adata_Y.var_names,
    n_perms=100,
    n_jobs=1,
)

vis_moran = {}
vis_moran['original'] = adata_Y.uns["moranI"]

row_sums = G_mos.sum(axis=1, keepdims=True) 
Xs_proj = (G_mos @ adata_X.X) / row_sums  

adata_Y_mos = adata_Y.copy()
adata_Y_mos.X = Xs_proj

sq.gr.spatial_neighbors(adata_Y_mos)
sq.gr.spatial_autocorr(
    adata_Y_mos,
    mode="moran",
    genes=adata_Y_mos.var_names,
    n_perms=100,
    n_jobs=1,
)
vis_moran['moscot'] = adata_Y_mos.uns["moranI"]


row_sums = G0.sum(axis=1, keepdims=True) 
Xs_proj = (G0 @ adata_X.X) / row_sums  

adata_Y_fpgw = adata_Y.copy()
adata_Y_fpgw.X = Xs_proj

sq.gr.spatial_neighbors(adata_Y_fpgw)
sq.gr.spatial_autocorr(
    adata_Y_fpgw,
    mode="moran",
    genes=adata_Y_fpgw.var_names,
    n_perms=100,
    n_jobs=1,
)
vis_moran['fpgw'] = adata_Y_fpgw.uns["moranI"]
vis_moran['moscot'] = vis_moran['moscot'].sort_index()
vis_moran['fpgw'] = vis_moran['fpgw'].sort_index()
vis_moran['original'] = vis_moran['original'].sort_index()

tmp = vis_moran['original']['I'].tolist()
tmp1 = vis_moran['fpgw']['I'].tolist()
tmp2 = vis_moran['moscot']['I'].tolist()

np.corrcoef(tmp2, tmp)[0, 1]
np.corrcoef(tmp1, tmp)[0, 1]


data = xen_moran
data_vis = vis_moran

  
y_vis = data_vis['original']['I'].tolist()
y_from_xen_moscot = data_vis['moscot']['I'].tolist()
y_from_vis_fpgw = data['fpgw']['I'].tolist()
y_from_vis_moscot = data['moscot']['I'].tolist()
y_from_xen_fpgw = data_vis['fpgw']['I'].tolist()
y_xen = data['original']['I'].tolist()


y_from_xen_fpgw = np.array(y_from_xen_fpgw)
y_from_xen_moscot = np.array(y_from_xen_moscot)
y_xen = np.array(y_xen)
y_from_vis_fpgw = np.array(y_from_vis_fpgw)
y_from_vis_moscot = np.array(y_from_vis_moscot)
y_vis = np.array(y_vis)

x = np.arange(len(y_vis)) 
idx = np.argsort(y_xen)

# reorder everything
x_sorted  = x[idx]
y1_sorted = y_vis[idx]
y2_sorted = y_from_vis_fpgw[idx]
y3_sorted = y_from_vis_moscot[idx]
y4_sorted = y_xen[idx]

# plt.plot(x, y4_sorted, label='original xenium')
# plt.plot(x, y1_sorted, label='original visium')
# plt.plot(x, y2_sorted, label='reconstrcuted xenium by fpgw')
# plt.plot(x, y3_sorted, label='reconstrcuted xenium by moscot')

# plt.xlabel('Gene')
# plt.ylabel('Morans I')
# plt.legend()
# plt.savefig('tmp_v2x.png')
# plt.close()


sns.set_theme(style="white", context="paper", font_scale=1.3)

plt.figure(figsize=(7, 5))

# Originals (grey)
plt.plot(x, y4_sorted, 
         label='Original Xenium',
         linewidth=2,
         alpha=0.7,
         color='gray')

plt.plot(x, y1_sorted, 
         label='Original Visium',
         linewidth=2,
         alpha=0.7,
         color='lightgray')

# Reconstructed (colored, highlighted)
plt.plot(x, y2_sorted, 
         label='Xenium reconstructed (FPGW)',
         linewidth=2.5,
         color='#ff7f0e')

plt.plot(x, y3_sorted, 
         label='Xenium reconstructed (MOSCOT)',
         linewidth=2.5,
         color='#1f77b4')

# Labels
plt.xlabel('Gene Rank')
plt.ylabel("Moran's I")

# Legend
plt.legend(
    frameon=False,
    loc='lower right',
    fontsize=6,           # smaller text
    handlelength=1.0,     # shorter line samples
    handletextpad=0.5,    # space between line and text
    borderaxespad=0.3,    # distance from axes border
    labelspacing=0.3     # vertical spacing between entries
)

# Layout
plt.tight_layout()

# Save high quality
plt.savefig('v2x.png', dpi=300, bbox_inches='tight')
plt.close()



x = np.arange(len(y_xen)) 
idx = np.argsort(y_vis)

# reorder everything
x_sorted  = x[idx]
y1_sorted = y_xen[idx]
y2_sorted = y_from_xen_fpgw[idx]
y3_sorted = y_from_xen_moscot[idx]
y4_sorted = y_vis[idx]

# plt.plot(x, y4_sorted, label='visium original')
# plt.plot(x, y1_sorted, label='xenium original')
# plt.plot(x, y2_sorted, label='visium reconstrcuted fpgw')
# plt.plot(x, y3_sorted, label='visium reconstrcuted moscot')

# plt.xlabel('Gene')
# plt.ylabel('Morans I')
# plt.legend()
# plt.savefig('tmp_x2v.png')
# plt.close()


sns.set_theme(style="white", context="paper", font_scale=1.3)

plt.figure(figsize=(7, 5))

# Originals (grey)
plt.plot(x, y4_sorted, 
         label='Original Visium',
         linewidth=2,
         alpha=0.7,
         color='gray')

plt.plot(x, y1_sorted, 
         label='Original Xenium',
         linewidth=2,
         alpha=0.7,
         color='lightgray')

# Reconstructed (colored, highlighted)
plt.plot(x, y2_sorted, 
         label='Visium reconstructed (FPGW)',
         linewidth=2.5,
         color='#ff7f0e')

plt.plot(x, y3_sorted, 
         label='Visium reconstructed (MOSCOT)',
         linewidth=2.5,
         color='#1f77b4')

# Labels
plt.xlabel('Gene Rank')
plt.ylabel("Moran's I")

# Legend
plt.legend(
    frameon=False,
    loc='lower right',
    fontsize=6,           # smaller text
    handlelength=1.0,     # shorter line samples
    handletextpad=0.5,    # space between line and text
    borderaxespad=0.3,    # distance from axes border
    labelspacing=0.3     # vertical spacing between entries
)

# Layout
plt.tight_layout()

# Save high quality
plt.savefig('x2v.png', dpi=300, bbox_inches='tight')
plt.close()


#summary bar plots

# Data
methods = ["Xenium (Reference)", "moscot", "SpaOT", "SpaOT After QC"]

visium_corr = [0.69, 0.27, 0.71, 0.80]
xenium_corr = [0.69, 0.49, 0.69, 0.74]

colors = ["gray", "tab:blue", "orange", '#ff7f0e']

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# Plot 1: Visium
axes[0].bar(methods, visium_corr, color=colors)
axes[0].set_title("Visium")
axes[0].set_ylabel("Pearson Correlation")
axes[0].set_ylim(0, 1)

for i, v in enumerate(visium_corr):
    axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center")

# Plot 2: Xenium
methods = ["Visium (Reference)", "moscot", "SpaOT", "SpaOT After QC"]
axes[1].bar(methods, xenium_corr, color=colors)
axes[1].set_title("Xenium")
axes[1].set_ylim(0, 1)

for i, v in enumerate(xenium_corr):
    axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center")

for ax in axes:
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('bar_new.png',dpi=300)