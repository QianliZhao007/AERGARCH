# AERGARCH SPX Example

This repository contains a cleaned SPX-only example for Realized GARCH models with high-dimensional realized measures and autoencoder-based dimension reduction.

The original research code was developed in notebooks. This version keeps the notebook workflow, while moving reusable functions into Python modules under `src/`.

## What This Example Includes

- Processed SPX example data loading and preparation.
- LW-RealGARCH and FW-RealGARCH examples.
- AVG-RealGARCH and single-measure RV5 Realized GARCH examples.
- PC-RealGARCH and IC-RealGARCH examples.
- A 12-12-1 rolling autoencoder example for SPX.
- AE-RealGARCH using the rolling encoded SPX series.
- VaR, ES loss, and MCS evaluation sections.

## Repository Structure

```text
AERGARCH.ipynb              Main notebook for the SPX example
src/                        Python helper modules
data/spx/                   Processed SPX example data
requirements.txt            Python package list
environment.yml             Conda environment record
ENVIRONMENT.md              Environment notes and checked package versions
```

## Data

This GitHub example uses processed SPX data stored under `data/spx/`.

The notebook loads these files through `src/data_loader.py` and returns notebook-ready dictionaries for the SPX market, including realized measures, in-sample and out-of-sample splits, and demeaned return series.

## Environment

The code was checked in the `l_nl_ae_rgarch` conda environment. The main package versions are recorded in `ENVIRONMENT.md`.

A typical setup is:

```powershell
conda env create -f environment.yml
conda activate l_nl_ae_rgarch
```

If the exact environment cannot be recreated, install the core packages listed in `requirements.txt`.

## Running The Notebook

Open and run:

```text
AERGARCH.ipynb
```

The notebook uses SPX as the example market. It produces model-specific result dictionaries such as:

```python
lw_allparams_df["SPX"]
fw_allparams_df["SPX"]
avg_allparams_df["SPX"]
rv5_allparams_df["SPX"]
pca_allparams_df["SPX"]
ica_allparams_df["SPX"]
ae_allparams_df["SPX"]
```

The main forecast variance series is stored in the `F_var` column. Predictive negative log-likelihood values are stored in the `Each_f_nLL` column.




