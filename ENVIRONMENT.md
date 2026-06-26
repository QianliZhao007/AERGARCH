# Environment

The code was checked with the existing conda environment `l_nl_ae_rgarch` on
Windows.

## Checked Versions

| Package | Version |
|---|---|
| Python | 3.11.5 |
| PyTorch | 2.10.0.dev20251118+cu128 |
| CUDA used by PyTorch | 12.8 |
| NumPy | 1.24.3 |
| pandas | 1.5.3 |
| SciPy | 1.13.1 |
| scikit-learn | 1.3.0 |
| arch | 8.0.0 |
| joblib | 1.2.0 |
| matplotlib | 3.7.3 |
| seaborn | 0.12.2 |

PyTorch reports CUDA as available in this environment.

## Windows Note

In this environment, PyTorch should be imported before NumPy/pandas/scikit-learn
to avoid a Windows DLL load-order error involving `torch/lib/c10.dll`. The
notebook already imports `torch` first.

## Smoke Tests

From the project root:

```powershell
conda run -n l_nl_ae_rgarch python scripts\smoke_notebook_start.py
python scripts\smoke_model_sections_spx.py
conda run -n l_nl_ae_rgarch python scripts\smoke_autoencoder_spx.py
conda run -n l_nl_ae_rgarch python scripts\smoke_risk_sections.py
```

The full rolling-window estimation is much slower than these smoke tests.

