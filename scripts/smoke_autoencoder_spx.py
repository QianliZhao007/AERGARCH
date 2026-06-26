"""Fast smoke test for the SPX Autoencoder and AE-RealGARCH sections."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: F401

from src.data_loader import get_spx_data
from src.model_sections import slice_model_inputs
from src.autoencoder_sections import (
    DEFAULT_AE_ARCHITECTURE_12121,
    DEFAULT_AE_HPARAMS,
    collect_ae_results,
    device,
    run_ae_realgarch_spx,
    run_rolling_ae_spx,
)


def main():
    spx_data = get_spx_data()
    market_list, rms_in, rms_out, ret_in, ret_out = slice_model_inputs(
        spx_data["market_list_2"],
        spx_data["rms_insample"],
        spx_data["rms_outsample"],
        spx_data["returns_insample"],
        spx_data["returns_outsample"],
        insample_tail=60,
        outsample_head=2,
    )

    hparams = DEFAULT_AE_HPARAMS.copy()
    hparams.update(
        {
            "FINETUNE_EPOCHS": 3,
            "PATIENCE": 2,
            "MIN_DELTA": 0.0,
        }
    )

    print("device", device)
    print("smoke data", market_list, rms_in["SPX"].shape, rms_out["SPX"].shape)

    all_market_encoded_series, all_market_weights = run_rolling_ae_spx(
        market_list,
        rms_in,
        rms_out,
        architecture=DEFAULT_AE_ARCHITECTURE_12121,
        hparams=hparams,
        use_cache=False,
        save=False,
    )
    print(
        "ae rolling ok",
        all_market_encoded_series["SPX"].shape,
        len(all_market_weights["SPX"]),
    )

    ae_rgarch_spx = run_ae_realgarch_spx(
        market_list,
        all_market_encoded_series,
        ret_in,
        ret_out,
        use_cache=False,
        save=False,
    )
    ae_allparams_df, ae_forecast_nll = collect_ae_results(ae_rgarch_spx, verbose=False)
    print("ae rgarch ok", ae_allparams_df["SPX"].shape, ae_forecast_nll)
    print("autoencoder smoke ok")


if __name__ == "__main__":
    main()
