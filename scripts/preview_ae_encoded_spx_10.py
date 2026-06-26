"""Preview SPX Autoencoder encoded series for the first 10 rolling windows."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import torch first in this conda environment to avoid Windows DLL load-order issues.
import torch  # noqa: F401

from src.autoencoder_sections import (
    DEFAULT_AE_ARCHITECTURE_12121,
    DEFAULT_AE_HPARAMS,
    device,
    run_rolling_ae_spx,
)
from src.data_loader import get_spx_data


N_ROLLING_WINDOWS = 10


def main():
    spx_data = get_spx_data()
    market = "SPX"

    rms_insample = {market: spx_data["rms_insample"][market]}

    # rolling_ae_for_market loops over range(n_outsample + 1), so 9 OOS rows gives
    # rolling windows 0, 1, ..., 9.
    rms_outsample = {
        market: spx_data["rms_outsample"][market].head(N_ROLLING_WINDOWS - 1)
    }

    hparams = DEFAULT_AE_HPARAMS.copy()

    print("device:", device)
    print("architecture:", DEFAULT_AE_ARCHITECTURE_12121)
    print("in-sample rows:", len(rms_insample[market]))
    print("rolling windows:", N_ROLLING_WINDOWS)

    all_market_encoded_series, _ = run_rolling_ae_spx(
        [market],
        rms_insample,
        rms_outsample,
        architecture=DEFAULT_AE_ARCHITECTURE_12121,
        hparams=hparams,
        use_cache=False,
        save=False,
    )

    encoded = all_market_encoded_series[market].copy()
    encoded.index = rms_insample[market].index
    encoded.columns = [f"window_{i}" for i in range(encoded.shape[1])]

    output_dir = PROJECT_ROOT / "outputs" / "preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "spx_ae_encoded_first_10_windows.csv"
    encoded.to_csv(output_path)

    print(encoded.head(10))
    print(f"saved to: {output_path}")


if __name__ == "__main__":
    main()
