# Cerebellum Mapping Reproducibility

This directory reproduces the multi-omics mouse cerebellum integration analyses presented in **Figure 5**. The workflow aligns sequencing spots with protein, metabolite, and histology image representations, then learns transport plans to reconstruct all modalities within a shared spatial coordinate system.

## Expected Inputs

The scripts expect the following preprocessed cerebellum AnnData files:

- `cerebellum/Cerebellum-MAGIC-seq.h5ad`
- `cerebellum/Cerebellum-PLATO.h5ad`
- `cerebellum/Cerebellum-MALDI-MSI.h5ad`

All three preprocessed AnnData objects can be downloaded from the Flow2Spatial repository:

https://github.com/bioinfo-biols/Flow2Spatial/tree/main/datasets

The image embedding scripts can regenerate the tile-center coordinates and DINO image embeddings when the raw H&E image is stored within the sequencing AnnData object.

## Scripts

- `fig5b_generate_tile_centers.py`  
  Generates a grid of H&E tile centers covering the tissue region based on sequencing coordinates.

- `fig5b_img_encoder_dino.py`  
  Crops H&E image tiles and extracts DINO image embeddings.

- `fig5b_cerebellum_cca_seq_protein_benchmark.py`  
  Benchmarks sequencing-to-protein mapping using CCA-enhanced PASTE2 and related baseline methods.

- `fig5b_cerebellum_cca_seq_metabolism_benchmark.py`  
  Benchmarks sequencing-to-metabolite mapping.

- `fig5b_cerebellum_cca_seq_img_benchmark.py`  
  Benchmarks image-to-sequencing mapping using DINO image embeddings.

- `fig5de_cerebellum_intergration_downstream.py`  
  Reconstructs protein and metabolite features in the sequencing spatial coordinate system and performs downstream analyses, including correlation analyses and summary visualizations.

## Suggested Run Order

```bash
python "reproductivity/cerebellum mapping/fig5b_generate_tile_centers.py"
python "reproductivity/cerebellum mapping/fig5b_img_encoder_dino.py"
python "reproductivity/cerebellum mapping/fig5b_cerebellum_cca_seq_protein_benchmark.py"
python "reproductivity/cerebellum mapping/fig5b_cerebellum_cca_seq_metabolism_benchmark.py"
python "reproductivity/cerebellum mapping/fig5b_cerebellum_cca_seq_img_benchmark.py"
python "reproductivity/cerebellum mapping/fig5de_cerebellum_intergration_downstream.py"
```

## Main Outputs

Running the workflow generates:

- Tile-center coordinate files and DINO image embedding files.
- Stepwise GW, CCA, and FGW transport plans.
- Per-cluster mapping accuracy evaluations.
- Spatial modality transfer visualizations.
- Gene–protein, protein–metabolite, and gene–metabolite correlation analyses.

## Method Overview

The benchmark scripts follow a three-stage alignment strategy:

1. **Initial structural alignment** using spatial information alone.
2. **Canonical Correlation Analysis (CCA)** to learn a shared latent representation across modalities.
3. **Final fused alignment** that combines spatial structure with the learned CCA features through fused Gromov–Wasserstein optimization.

This workflow enables accurate cross-modal correspondence while preserving the spatial organization of the mouse cerebellum.