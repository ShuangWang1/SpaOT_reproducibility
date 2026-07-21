#integration
import pickle
import scanpy as sc
import pandas as pd
import numpy as np

adata = sc.read_h5ad('cerebellum/Cerebellum-MAGIC-seq.h5ad')
adata_protein = sc.read_h5ad('cerebellum/Cerebellum-PLATO.h5ad')
adata_meta = sc.read_h5ad('cerebellum/Cerebellum-MALDI-MSI.h5ad')

with open('cerebellum_cca_seq_protein_res/step3_fpgw_plan.pickle', 'rb') as f:
    G01 = pickle.load(f)
    
with open('cerebellum_cca_seq_meta_res/cerebellum_fpgw_plan.pickle', 'rb') as f:
    G02 = pickle.load(f)

with open('cerebellum_cca_seq_meta_res/benchmark_moscot_step3_plan.pickle', 'rb') as f:
    G02_moscot = pickle.load(f)

with open('cerebellum_cca_seq_meta_res/benchmark_paste2_step3_plan.pickle', 'rb') as f:
    G02_paste2 = pickle.load(f)
 
protein_reconstructed = (G01.T @ adata_protein.X) * len(adata)
meta_reconstructed = (G02.T @ adata_meta.X) * len(adata)
meta_reconstructed_paste2 = (G02_paste2.T @ adata_meta.X) * len(adata)
meta_reconstructed_moscot = (G02_moscot.T @ adata_meta.X) * len(adata)


# RNA
rna = adata.X
gene_names = adata.var_names

# Protein
protein = protein_reconstructed
protein_names = adata_protein.var_names

# Metabolite
meta = meta_reconstructed
meta_names = adata_meta.var_names

# Convert to DataFrame for convenience
rna_df = pd.DataFrame(rna.toarray(), columns=gene_names)
protein_df = pd.DataFrame(protein, columns=protein_names)
meta_df = pd.DataFrame(meta, columns=meta_names)

genes = ['Mbp','Ttr','mt-Nd2','Chd9']
proteins = ['Cldn11','Clic6','Grm1','Fubp1']
metas = ['mz845.67185','mz811.5554','mz787.54715','mz909.56922']

from scipy.stats import spearmanr, ranksums
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

results_gp = []

for g in genes:
    for p in proteins:
        if g in rna_df.columns and p in protein_df.columns:
            rho, pval = spearmanr(rna_df[g], protein_df[p])
            results_gp.append([g, p, rho, pval])

gp_corr = pd.DataFrame(results_gp, columns=['gene','protein','rho','pval'])
print(gp_corr.sort_values('rho', ascending=False))


#plot
# 1. Pivot the data to wide format
heatmap_data = gp_corr.pivot(index='gene', columns='protein', values='rho')

# Reorder rows and columns
heatmap_data = heatmap_data.reindex(index=genes, columns=proteins)

# 2. Plot the heatmap
plt.figure(figsize=(10, 8)) # Adjust size based on number of genes/proteins
sns.heatmap(heatmap_data, cmap='coolwarm', vmin=-1, vmax=1, annot=False)
plt.title('Gene-Protein Spearman Correlation')
plt.grid(False)
plt.savefig('gene_prot_corr.png',dpi=300)


results_pm = []

for p in proteins:
    for m in metas:
        if p in protein_df.columns and m in meta_df.columns:
            rho, pval = spearmanr(protein_df[p], meta_df[m])
            results_pm.append([p, m, rho, pval])

pm_corr = pd.DataFrame(results_pm, columns=['protein','metabolite','rho','pval'])
print(pm_corr.sort_values('rho', ascending=False))



#plot
heatmap_data = pm_corr.pivot(index='protein', columns='metabolite', values='rho')
heatmap_data = heatmap_data.reindex(index=proteins, columns=metas)

plt.figure(figsize=(10, 8)) # Adjust size based on number of genes/proteins
sns.heatmap(heatmap_data, cmap='coolwarm', vmin=-1, vmax=1, annot=False)
plt.title('Protein-Metabolite Spearman Correlation')
plt.grid(False)
plt.savefig('protein_meta_corr.png', dpi=300)


results_gm = []

for g in genes:
    for m in metas:
        if g in rna_df.columns and m in meta_df.columns:
            rho, pval = spearmanr(rna_df[g], meta_df[m])
            results_gm.append([g, m, rho, pval])

gm_corr = pd.DataFrame(results_gm, columns=['gene','metabolite','rho','pval'])


#plot
heatmap_data = gm_corr.pivot(index='gene', columns='metabolite', values='rho')
heatmap_data = heatmap_data.reindex(index=genes, columns=metas)

plt.figure(figsize=(10, 8)) # Adjust size based on number of genes/proteins
sns.heatmap(heatmap_data, cmap='coolwarm', vmin=-1, vmax=1, annot=False)
plt.title('Gene-Metabolite Spearman Correlation')
plt.grid(False)
plt.savefig('gene_meta_corr.png')



#2 pathway-styple scoring

sc.settings.set_figure_params(dpi=300, dpi_save=300)

def safe_mean(df, cols):
    cols = [c for c in cols if c in df.columns]
    return df[cols].mean(axis=1)

for myelin_genes, myelin_proteins, myelin_metas in zip(genes, proteins, metas):
    
    
    adata.obs['myelin_score_rna_orginal'] = rna_myelin.values
    sc.pl.spatial(adata, color='myelin_score_rna_orginal', cmap='viridis',save=f'_{myelin_genes}_score_rna_original.png',s=15)

    # Normalize
    scaler = StandardScaler()
    rna_myelin = scaler.fit_transform(rna_myelin.values.reshape(-1,1)).flatten()
    protein_myelin = scaler.fit_transform(protein_myelin.values.reshape(-1,1)).flatten()
    meta_myelin = scaler.fit_transform(meta_myelin.values.reshape(-1,1)).flatten()

    combined_myelin = (rna_myelin + protein_myelin + meta_myelin) / 3

    adata.obs['myelin_score'] = combined_myelin
    adata.obs['myelin_score_rna'] = rna_myelin

    sc.pl.spatial(adata, color='myelin_score', cmap='viridis',save=f'_{myelin_genes[0]}_score.png',s=15)
    sc.pl.spatial(adata, color='myelin_score_rna', cmap='viridis',save=f'_{myelin_genes[0]}_score_rna.png',s=15)



########plot original gene, protein and meta

for gene, protein, meta in zip(genes, proteins, metas):

    adata.obs[f'{gene}_score'] = combined_myelin
    adata.obs['myelin_score_rna'] = rna_myelin

    sc.pl.spatial(adata, color='myelin_score', cmap='viridis',save=f'_{myelin_genes[0]}_score.png',s=15)
    sc.pl.spatial(adata, color='myelin_score_rna', cmap='viridis',save=f'_{myelin_genes[0]}_score_rna.png',s=15)







#moran's I for RNA, protein and meta
#RNA and metabolism first
import squidpy as sq
import anndata as ad

sq.gr.spatial_neighbors(adata)
sq.gr.spatial_autocorr(
    adata,
    mode="moran",
    genes=genes,
    n_perms=100,
    n_jobs=1,
)

adata_from_meta = ad.AnnData(
    X=meta_reconstructed,
    var=adata_meta.var.copy(),
    obsm={"spatial": adata.obsm["spatial"].copy()}
)
sq.gr.spatial_neighbors(adata_from_meta)
sq.gr.spatial_autocorr(
    adata_from_meta,
    mode="moran",
    genes=metas,
    n_perms=100,
    n_jobs=1,
)

adata_from_meta_moscot = ad.AnnData(
    X=meta_reconstructed,
    var=adata_meta.var.copy(),
    obsm={"spatial": adata.obsm["spatial"].copy()}
)
sq.gr.spatial_neighbors(adata_from_meta_moscot)
sq.gr.spatial_autocorr(
    adata_from_meta_moscot,
    mode="moran",
    genes=metas,
    n_perms=100,
    n_jobs=1,
)

adata_from_meta_paste2 = ad.AnnData(
    X=meta_reconstructed,
    var=adata_meta.var.copy(),
    obsm={"spatial": adata.obsm["spatial"].copy()}
)
sq.gr.spatial_neighbors(adata_from_meta_paste2)
sq.gr.spatial_autocorr(
    adata_from_meta,
    mode="moran",
    genes=metas,
    n_perms=100,
    n_jobs=1,
)



adata_from_meta.uns["moranI"]


sq.gr.spatial_neighbors(adata_meta)
sq.gr.spatial_autocorr(
    adata_meta,
    mode="moran",
    genes=metas,
    n_perms=100,
    n_jobs=1,
)
adata_meta.uns["moranI"]



# adata.uns["moranI"]
#                I  pval_norm  var_norm  pval_z_sim  pval_sim   var_sim  pval_norm_fdr_bh  pval_z_sim_fdr_bh  pval_sim_fdr_bh
# Ttr     0.964798        0.0  0.000311         0.0  0.009901  0.000633               0.0                0.0         0.009901
# Mbp     0.859606        0.0  0.000311         0.0  0.009901  0.000578               0.0                0.0         0.009901
# mt-Nd2  0.701004        0.0  0.000311         0.0  0.009901  0.000368               0.0                0.0         0.009901
# Chd9    0.597060        0.0  0.000311         0.0  0.009901  0.000438               0.0                0.0         0.009901


# adata_meta.uns["moranI"]
#                     I  pval_norm  var_norm  pval_z_sim  pval_sim   var_sim  pval_norm_fdr_bh  pval_z_sim_fdr_bh  pval_sim_fdr_bh
# mz845.67185  0.895593        0.0  0.000132         0.0  0.009901  0.000241               0.0                0.0         0.009901
# mz787.54715  0.805866        0.0  0.000132         0.0  0.009901  0.000175               0.0                0.0         0.009901
# mz909.56922  0.740089        0.0  0.000132         0.0  0.009901  0.000186               0.0                0.0         0.009901
# mz811.5554   0.451554        0.0  0.000132         0.0  0.009901  0.000147               0.0                0.0         0.009901

# adata_from_meta.uns["moranI"]
#                     I  pval_norm  var_norm  pval_z_sim  pval_sim   var_sim  pval_norm_fdr_bh  pval_z_sim_fdr_bh  pval_sim_fdr_bh
# mz845.67185  0.857943        0.0  0.000187         0.0  0.009901  0.000423               0.0                0.0         0.009901
# mz787.54715  0.820723        0.0  0.000187         0.0  0.009901  0.000460               0.0                0.0         0.009901
# mz909.56922  0.704236        0.0  0.000187         0.0  0.009901  0.000376               0.0                0.0         0.009901
# mz811.5554   0.628060        0.0  0.000187         0.0  0.009901  0.000353               0.0                0.0         0.009901



# adata_from_meta_mos.uns["moranI"]
#                     I  pval_norm  var_norm  pval_z_sim  pval_sim   var_sim  pval_norm_fdr_bh  pval_z_sim_fdr_bh  pval_sim_fdr_bh
# mz811.5554   0.947793        0.0  0.000187         0.0  0.009901  0.000559               0.0                0.0         0.009901
# mz787.54715  0.939996        0.0  0.000187         0.0  0.009901  0.000444               0.0                0.0         0.009901
# mz845.67185  0.933805        0.0  0.000187         0.0  0.009901  0.000384               0.0                0.0         0.009901
# mz909.56922  0.886320        0.0  0.000187         0.0  0.009901  0.000403               0.0                0.0         0.009901

# adata_from_meta_paste2.uns["moranI"]
#                     I  pval_norm  var_norm  pval_z_sim  pval_sim   var_sim  pval_norm_fdr_bh  pval_z_sim_fdr_bh  pval_sim_fdr_bh
# mz845.67185  0.865727        0.0  0.000187         0.0  0.009901  0.000480               0.0                0.0         0.009901
# mz787.54715  0.826329        0.0  0.000187         0.0  0.009901  0.000543               0.0                0.0         0.009901
# mz909.56922  0.716236        0.0  0.000187         0.0  0.009901  0.000386               0.0                0.0         0.009901
# mz811.5554   0.629165        0.0  0.000187         0.0  0.009901  0.000433               0.0                0.0         0.009901


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# x-axis order: metabolite-aligned features
features = ["Mbp / mz845", "Ttr / mz787", "mt-Nd2 / mz909", "Chd9 / mz811"]

data = pd.DataFrame({
    "feature": features,

    "RNA": [0.964798, 0.859606, 0.701004, 0.597060],
    "Metabolite": [0.895593, 0.805866, 0.740089, 0.451554],
    "SpaOT reconstructed RNA": [0.857943, 0.820723, 0.704236, 0.628060],
    "Moscot reconstructed RNA": [0.947793, 0.939996, 0.933805, 0.886320],
    "PASTE2 reconstructed RNA": [0.865727, 0.826329, 0.716236, 0.629165],
})

methods = [
    "RNA",
    "Metabolite",
    "SpaOT reconstructed RNA",
    "Moscot reconstructed RNA",
    "PASTE2 reconstructed RNA",
]

colors = ["gray","lightgray", "orange", "#1f77b4", "#008fee"]

x = np.arange(len(features))
width = 0.15

fig, ax = plt.subplots(figsize=(12, 5))

for i, method in enumerate(methods):
    ax.bar(
        x + (i - (len(methods)-1)/2) * width,
        data[method],
        width=width,
        label=method,
        edgecolor="black",
        linewidth=0.3,
        color=colors[i]
    )

ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45, ha="right")
ax.set_ylabel("Moran's I")
ax.set_xlabel("Metabolite feature")
ax.legend(title="Object", frameon=False)
ax.axhline(0, color="black", lw=0.8)
ax.legend(
    title="Object",
    frameon=False,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5)  # 👈 pushes legend outside right
)

plt.tight_layout()
plt.savefig('bar.png',dpi=300)