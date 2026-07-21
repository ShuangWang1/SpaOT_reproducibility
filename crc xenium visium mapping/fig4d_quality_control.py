import scanpy as sc
import numpy as np

adata = sc.read_h5ad("xenium_p2/xenium.h5ad")
adata_8um = sc.read_h5ad("visium_p2/visium.h5ad")

common_genes = list(set(adata.var_names).intersection(adata_8um.var_names))
adata = adata[:, common_genes].copy()

#remove duplicate
adata_8um = adata_8um[:, ~adata_8um.var_names.duplicated()].copy()
adata_8um = adata_8um[:, common_genes].copy()

sc.pp.calculate_qc_metrics(adata, percent_top=None,inplace=True)


sc.pl.violin(adata, ['total_counts', 'n_genes_by_counts'], jitter=0.4,save='xenium_qc.png')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(adata.obs['total_counts'], bins=100)
axes[0].set_title('Total Counts')
axes[0].set_xlabel('Counts')
axes[0].set_ylabel('Cells')

axes[1].hist(adata.obs['n_genes_by_counts'], bins=100)
axes[1].set_title('Genes per Cell')
axes[1].set_xlabel('Genes')
axes[1].set_ylabel('Cells')

plt.tight_layout()
plt.savefig('xenium_qc_hist.png', dpi=300)
plt.show()

sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

# ---------------------------
# 4. Apply Visium/Xenium-style thresholds
# These are common thresholds (you can adjust depending on your dataset)
# ---------------------------

for data_source in ['Xenium', 'Visium']:
    if data_source == 'Visium':
        min_genes = 5        # minimum detected genes per spot
        max_genes = 350      # maximum detected genes per spot (remove likely doublets)
        min_counts = 300       # minimum total counts per spot
        max_counts = 12000     # maximum total counts per spot (remove likely doublets)
        
        
        adata_filtered = adata_8um[
            (adata_8um.obs['n_genes_by_counts'] >= min_genes) &
            (adata_8um.obs['n_genes_by_counts'] <= max_genes) &
            (adata_8um.obs['total_counts'] >= min_counts) &
            (adata_8um.obs['total_counts'] <= max_counts),
            :
        ].copy()
    else:    
        min_genes = 5        # minimum detected genes per spot
        max_genes = 250      # maximum detected genes per spot (remove likely doublets)
        min_counts = 10      # minimum total counts per spot
        max_counts = 750     # maximum total counts per spot (remove likely doublets)

        adata_filtered = adata[
            (adata.obs['n_genes_by_counts'] >= min_genes) &
            (adata.obs['n_genes_by_counts'] <= max_genes) &
            (adata.obs['total_counts'] >= min_counts) &
            (adata.obs['total_counts'] <= max_counts),
            :
        ].copy()

    # ---------------------------
    # 5. Optional: remove doublets using Scrublet
    # ---------------------------

    # ---------------------------
    # 6. Normalize and log-transform (standard Visium workflow)
    # ---------------------------
    sc.pp.normalize_total(adata_filtered, target_sum=1e4)
    sc.pp.log1p(adata_filtered)

    # ---------------------------
    # 7. Save cleaned data
    # ---------------------------
    adata_filtered.write(f'{data_source}_cleaned.h5ad')