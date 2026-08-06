# CRC Xenium–Visium Mapping Reproducibility

This directory reproduces the **same-omics, cross-technology mapping analysis** between **Xenium** and **Visium** spatial transcriptomics data for the colorectal cancer (CRC) dataset presented in **Figure 4**.

The workflow performs:

- Mapping single-cell Xenium data to Visium spots using **SpaOT/FPGW**
- Evaluating spatial autocorrelation consistency via **Moran's I**
- Performing downstream differential expression (DE) analysis
- Running Reactome pathway and Gene Ontology (GO) enrichment analysis
- Visualizing reconstructed marker-gene expression and spatial transfer results

---

## Required Input Data

The following preprocessed datasets are required:

```text
xenium_p2/xenium.h5ad
visium_p2/visium.h5ad
```

Both datasets can be downloaded from the 10x Genomics colorectal cancer dataset:

> https://www.10xgenomics.com/platforms/visium/product-family/dataset-human-crc

Generated transport plans, intermediate files, and analysis outputs are written locally during execution.

---

## Scripts

| Script | Description |
|---------|-------------|
| `fig4b_xenium_visium_map.py` | Computes the SpaOT/FPGW transport plan between Xenium and Visium and generates the mapping visualization. |
| `fig4d_quality_control.py` | Produces quality-control plots for the Xenium and Visium datasets. |
| `fig4cd_moransI_correlation.py` | Reconstructs cross-technology expression profiles and compares spatial autocorrelation using Moran's I. |
| `fig4fg_mapped_xenium_DEGs.py` | Transfers (imputes) Xenium expression profiles using the transport plan and performs differential expression analysis on mapped cell groups. |
| `fig4fg_DEG_pathway_GO_run.r` | Performs Reactome pathway and Gene Ontology enrichment analysis on the identified differentially expressed genes. |
| `fig4hi_marker_xen_vis_mapping.py` | Visualizes reconstructed marker-gene expression and representative spatial mapping results. |

---

## Recommended Execution Order

Run the scripts from the repository root in the following order:

```bash
python "reproducibility/crc xenium visium mapping/fig4b_xenium_visium_map.py"
python "reproducibility/crc xenium visium mapping/fig4d_quality_control.py"
python "reproducibility/crc xenium visium mapping/fig4cd_moransI_correlation.py"
python "reproducibility/crc xenium visium mapping/fig4fg_mapped_xenium_DEGs.py"
Rscript "reproducibility/crc xenium visium mapping/fig4fg_DEG_pathway_GO_run.r"
python "reproducibility/crc xenium visium mapping/fig4hi_marker_xen_vis_mapping.py"
```

---

## Main Outputs

The workflow produces the following outputs:

- **Transport plans**
  - SpaOT/FPGW transport matrices (e.g., `fpgw_plan_mass0.5.pickle`)
- **Mapping visualizations**
  - Cross-technology mapping figures
  - Marker-gene reconstruction plots
- **Spatial autocorrelation analysis**
  - Moran's I comparison figures
- **Differential expression analysis**
  - DEG tables for mapped Xenium cell groups
- **Functional enrichment**
  - Reactome pathway enrichment results
  - Gene Ontology (GO) enrichment results

---

## Notes

- The scripts currently contain several hard-coded input/output paths and figure names.
- It is recommended to execute the workflow from the **repository root**. If running from another location, update the file paths accordingly.
- Intermediate files (including transport plans) are reused by downstream analyses; therefore, the scripts should be executed in the recommended order.