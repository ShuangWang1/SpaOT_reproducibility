# Simulation Benchmark Reproducibility

This directory reproduces the synthetic robustness experiments presented in **Figure 2** of the manuscript. These experiments evaluate whether SpaOT can distinguish meaningful structural correspondences from outlier noise and whether the resulting optimal transport distances effectively separate graphs with different community structures.

## Scripts

- `fig2ac_spaOT_object_mapping.py`  
  Generates synthetic crescent (two-moon) point clouds, introduces outlier and structural noise, benchmarks SpaOT, moscot, PASTE2, and POT-based baselines, and saves the object-mapping visualizations used in **Figure 2a–c**.

- `fig2e_clustering_benchmark.py`  
  Generates synthetic graph datasets containing one, two, and three communities, injects outlier nodes, performs graph clustering using OT-derived distances, estimates graph barycenters, and saves the clustering and barycenter visualizations for **Figure 2e**.

- `fig2f_clustering_dist.py`  
  Computes pairwise graph distance matrices for the synthetic graph benchmark and generates the distance heatmaps shown in **Figure 2f**.

## Suggested Run Order

```bash
python "reproducibility/simulatoin benchmark/fig2ac_spaOT_object_mapping.py"
python "reproducibility/simulatoin benchmark/fig2e_clustering_benchmark.py"
python "reproducibility/simulatoin benchmark/fig2f_clustering_dist.py"
```

## Main Outputs

Running the workflow generates:

- Object-mapping visualizations (`*.png`).
- Graph clustering and barycenter figures under `result/`.
- Pairwise graph distance heatmaps comparing SpaOT with baseline optimal transport methods.

## Notes

- Fixed random seeds are used where appropriate to ensure reproducibility of the synthetic datasets and benchmark results.
- The directory name intentionally retains the original typo (`simulatoin`) to remain consistent with the project structure and existing scripts.