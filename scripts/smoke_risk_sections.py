"""Smoke test for VaR/ES loss and MCS section variables."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    notebook_path = PROJECT_ROOT / "AERGARCH.ipynb"
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    index = pd.bdate_range("2020-01-01", periods=160)
    returns = pd.Series(
        np.linspace(-2.0, 2.0, len(index)) / 10,
        index=index,
        name="return",
    )

    model_names = ["lw", "nlw", "avg", "ae", "pca", "ica", "rv5"]
    model_dicts = {}
    for i, name in enumerate(model_names):
        f_var = pd.Series(0.8 + 0.02 * i + np.linspace(0.01, 0.05, len(index)), index=index)
        each_nll = pd.Series(1.0 + 0.05 * i + np.linspace(0.0, 0.2, len(index)), index=index)
        model_dicts[name] = {
            "SPX": pd.DataFrame(
                {
                    "F_var": f_var,
                    "Each_f_nLL": each_nll,
                }
            )
        }

    env = {
        "__name__": "__main__",
        "np": np,
        "pd": pd,
        "market_list_2": ["SPX"],
        "returns_outsample": {"SPX": returns},
        "lw_allparams_df": model_dicts["lw"],
        "nlw_allparams_df": model_dicts["nlw"],
        "avg_allparams_df": model_dicts["avg"],
        "ae_allparams_df": model_dicts["ae"],
        "pca_allparams_df": model_dicts["pca"],
        "ica_allparams_df": model_dicts["ica"],
        "rv5_allparams_df": model_dicts["rv5"],
    }

    markers = [
        "from src.risk_sections import *",
        "covid_qloss_result, covid_jointloss_result",
        "mcs_results = run_mcs_all_markets",
        "covid_mcs_results = run_mcs_all_markets",
    ]

    for marker in markers:
        for idx, cell in enumerate(nb["cells"]):
            source = "".join(cell.get("source", []))
            if cell.get("cell_type") == "code" and marker in source:
                exec(source, env)
                print(f"cell with marker {marker!r} ok")
                break
        else:
            raise RuntimeError(f"Could not find notebook cell containing: {marker}")

    required = [
        "allmodels_params_vares_markets",
        "covid_qloss_result",
        "covid_jointloss_result",
        "df_mcs_qloss",
        "df_mcs_jointloss",
        "df_mcs_pnll",
        "df_covid_mcs_qloss",
        "df_covid_mcs_jointloss",
        "df_covid_mcs_pnll",
    ]
    for name in required:
        value = env[name]
        if hasattr(value, "shape"):
            print(name, value.shape)
        else:
            print(name, type(value))

    assert "VaR" in env["lw_allparams_df"]["SPX"].columns
    assert "ES" in env["lw_allparams_df"]["SPX"].columns
    assert "QLoss" in env["ae_allparams_df"]["SPX"].columns
    assert "JointLoss" in env["ae_allparams_df"]["SPX"].columns
    print("risk section smoke ok")


if __name__ == "__main__":
    main()
