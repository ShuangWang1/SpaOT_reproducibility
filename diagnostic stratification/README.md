
---

# Diagnostic Stratification Reproducibility

This case study reproduces the sample-distance, classification, and barycenter analyses presented in **Fig. 3**. The workflow uses **SpaOT/FPGW transport costs** as biologically informed distances between spatial proteomics samples and evaluates whether these distances improve disease and molecular subtype stratification.

## Required Input Data

The scripts expect the following preprocessed datasets:

* `data/BZ.h5ad` — Breast cancer (Jackson cohort)
* `data/metabric.h5ad` — Breast cancer (METABRIC cohort)
* `data/CRC_clusters_neighborhoods_markers.csv` — Colorectal cancer (CRC)

### Dataset Downloads

* **Jackson cohort:** [https://doi.org/10.5281/zenodo.4607374](https://doi.org/10.5281/zenodo.4607374)
* **METABRIC cohort:** [https://idr.openmicroscopy.org/](https://idr.openmicroscopy.org/)
* **CRC cohort:** [https://data.mendeley.com/datasets/mpjzbtfgfr/1](https://data.mendeley.com/datasets/mpjzbtfgfr/1)

### Preprocessing

The Jackson and METABRIC datasets require an additional preprocessing step before running the analysis. The preprocessing script is provided in `breast_cancer_preprocess.py`.

Example:

```bash
python preprocess.py --dataset all
```

The datasets themselves are **not included** in this repository.

---

## Scripts

| Script                                 | Description                                                                                                   |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `fig3cd&fg&hi_sample_distance.py`      | Computes pairwise SpaOT/FPGW distances for CRC and breast cancer samples.                                     |
| `fig3cd&fg_repeated_cv.py`             | Performs repeated cross-validation using intra-cohort distance features for the Jackson and METABRIC cohorts. |
| `fig3d_barycenter_patient.py`          | Computes patient-level FPGW barycenters from CRC tissue cores.                                                |
| `fig3d_barycenter_region.py`           | Computes region-level FPGW barycenters from CRC tissue cores.                                                 |
| `fig3d_sample_distance_bary.py`        | Computes distances between samples and their barycenter representations.                                      |
| `fig3e_barycenter_CLR_DII.py`          | Computes subtype-level CRC barycenters for the CLR and DII groups.                                            |
| `fig3e_barycenter_crc_CLR_DII_plot.py` | Visualizes CRC subtype barycenters and selected feature channels.                                             |
| `fig3hi_sample_distance_inter.py`      | Computes cross-cohort SpaOT/FPGW distances between the METABRIC and Jackson cohorts.                          |
| `fig3hi_repeated_cv_inter.py`          | Performs repeated cross-cohort classification for the translational stratification task.                      |

---

## Suggested Execution Order

### 1. Compute Sample-Level Distances

Pairwise distance computation is the most computationally intensive step. For example, the Jackson cohort requires computing **559 × 599 = 334,841** sample pairs. We recommend running the computation in batches by specifying the sample-pair index range using `--start` and `--end`.

```bash
python "reproductivity/diagnostic stratification/fig3cd&fg&hi_sample_distance.py" \
    --dataset BZ --start 0 --end 1000

python "reproductivity/diagnostic stratification/fig3hi_sample_distance_inter.py" \
    --start 0 --end 1000
```

---

### 2. Compute CRC Barycenters

```bash
python "reproductivity/diagnostic stratification/fig3d_barycenter_patient.py" \
    --patient 0

python "reproductivity/diagnostic stratification/fig3d_barycenter_region.py" \
    --patient 0

python "reproductivity/diagnostic stratification/fig3e_barycenter_CLR_DII.py"
```

---

### 3. Compute Barycenter Distances and Generate Visualizations

```bash
python "reproductivity/diagnostic stratification/fig3d_sample_distance_bary.py"

python "reproductivity/diagnostic stratification/fig3e_barycenter_crc_CLR_DII_plot.py"
```

---

### 4. Run Repeated Classification

```bash
python "reproductivity/diagnostic stratification/fig3cd&fg_repeated_cv.py"

python "reproductivity/diagnostic stratification/fig3hi_repeated_cv_inter.py"
```

---

## Main Outputs

The workflow produces:

* Pairwise SpaOT/FPGW distance matrices for intra- and inter-cohort analyses.
* Patient-, region-, and subtype-level CRC FPGW barycenter representations.
* Repeated cross-validation performance metrics, including **AUROC** and **AUPR**.
* Barycenter visualizations and classification summary figures.

---

## Notes

* Several scripts are designed for batched execution using either patient indices or sample-pair index ranges.
* Before launching large-scale jobs, check the argument parser at the beginning of each script to determine the available command-line options.
* The sample-distance computations are the most time-consuming part of the pipeline and are intended to be executed in parallel across multiple jobs or computing nodes.
