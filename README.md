# SpaOT Reproducibility Guide

This repository contains the complete reproducibility workflows for all experiments presented in the manuscript. Each case study corresponds to one figure (or figure panel) and demonstrates a different application of **SpaOT/FPGW** for robust optimal transport on spatial biological data.

The workflows are organized into five independent case studies. Each directory contains its own README with detailed instructions, required datasets, and script descriptions.

---

# Repository Overview

| Figure | Case Study | Application | Primary Data |
|---------|------------|-------------|--------------|
| **Fig. 2** | Simulation Benchmark | Synthetic robustness evaluation | Simulated point clouds & graphs |
| **Fig. 3** | Diagnostic Stratification | Patient/sample similarity and disease stratification | Spatial proteomics |
| **Fig. 4** | CRC Xenium–Visium Mapping | Same-omics cross-technology mapping | Spatial transcriptomics |
| **Fig. 5** | Cerebellum Mapping | Cross-modal multi-omics integration | Sequencing, protein, metabolite & histology |
| **Fig. 6** | Kidney Mapping | Cross-technology mapping and downstream communication analysis | Xenium & Visium HD |

---

# Case Studies

## 1. Simulation Benchmark (Figure 2)

Evaluates the robustness of **SpaOT/FPGW** on synthetic datasets.

The benchmark investigates whether SpaOT correctly distinguishes biologically meaningful structure from outlier noise while preserving graph topology. It also evaluates whether transport costs can separate graphs with different community structures.

### Highlights

- Synthetic object matching
- Robustness to outlier and structural noise
- Graph distance benchmarking
- Graph clustering
- Graph barycenter estimation

**Input**

No external datasets are required. All synthetic datasets are generated automatically.

---

## 2. Diagnostic Stratification (Figure 3)

Demonstrates how SpaOT/FPGW transport distances can serve as biologically meaningful similarity measures between spatial proteomics samples.

The workflow computes pairwise sample distances, performs disease and molecular subtype classification, constructs FPGW barycenters, and evaluates translational prediction across cohorts.

### Highlights

- Patient similarity computation
- Intra-cohort classification
- Cross-cohort prediction
- Patient/region/subtype barycenters
- Disease stratification

**Required datasets**

- Jackson breast cancer cohort
- METABRIC breast cancer cohort
- CRC imaging mass cytometry dataset

Dataset download links are provided in the case-study README.

---

## 3. CRC Xenium–Visium Mapping (Figure 4)

Reproduces the same-omics, cross-technology mapping between Xenium single-cell data and Visium spatial transcriptomics.

The workflow aligns Xenium cells to Visium spots, reconstructs spatial gene expression, evaluates spatial autocorrelation, and performs downstream biological analyses.

### Highlights

- Xenium → Visium mapping
- Moran's I consistency analysis
- Differential expression
- Reactome enrichment
- Gene Ontology analysis
- Spatial visualization

**Required datasets**

10x Genomics Human Colorectal Cancer Xenium/Visium dataset.

---

## 4. Cerebellum Mapping (Figure 5)

Demonstrates cross-modal spatial integration between sequencing, protein, metabolite, and histology image modalities.

The workflow aligns multiple molecular modalities into a shared spatial coordinate system using SpaOT/FPGW with Canonical Correlation Analysis (CCA).

### Highlights

- Sequence → protein mapping
- Sequence → metabolite mapping
- Image → sequencing mapping
- Multi-modal reconstruction
- Cross-modal correlation analysis

**Required datasets**

Flow2Spatial mouse cerebellum datasets.

---

## 5. Kidney Mapping (Figure 6)

Reproduces Xenium-to-Visium HD mapping for chronic kidney disease (CKD) tissue.

The workflow compares SpaOT/FPGW against moscot, evaluates mapping quality at tissue and glomerulus levels, reconstructs gene expression, and performs downstream cell-cell communication analysis.

### Highlights

- Xenium ↔ Visium HD mapping
- Mapping quality evaluation
- Glomerulus-level analysis
- Expression imputation
- COMMOT cell-cell communication analysis

**Required datasets**

Preprocessed CKD Xenium and Visium HD datasets.

> **Note:** The original dataset is currently not publicly available.

---

# General Workflow

Although each case study is independent, most follow a similar pipeline:

```text
Input datasets
        │
        ▼
Preprocessing (if required)
        │
        ▼
SpaOT/FPGW transport plan
        │
        ▼
Mapping / Alignment
        │
        ▼
Evaluation
        │
        ▼
Downstream biological analysis
        │
        ▼
Figures and quantitative results
```
---

# Directory Structure

```text
reproducibility/
│
├── simulation benchmark/
│
├── diagnostic stratification/
│
├── crc xenium visium mapping/
│
├── cerebellum mapping/
│
└── kidney mapping/
```

Each directory contains:

- A dedicated README
- Reproducibility scripts
- Intermediate outputs
- Generated figures

---

# Running the Experiments

Each case study is self-contained and can be executed independently.

For each directory:

1. Download the required datasets.
2. Follow the preprocessing instructions (if applicable).
3. Execute the scripts in the recommended order provided in the corresponding README.
4. Inspect the generated figures and quantitative outputs.

---

# Notes

- Many scripts contain hard-coded input/output paths. Running the workflows from the repository root is recommended.
- Large-scale optimal transport computations (particularly sample-distance calculations) are designed to support batched or parallel execution.
- Intermediate transport plans generated in earlier steps are reused by downstream analyses; therefore, the recommended execution order should be followed within each case study.
- Random seeds are fixed where appropriate to improve reproducibility of stochastic experiments.

---

# Citation

If you use this repository in your research, please cite the accompanying manuscript.
