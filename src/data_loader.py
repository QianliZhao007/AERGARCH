"""Data preparation helpers for the SPX-only GitHub example."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "oxfordmanrealizedvolatilityindices.csv"
SPX_OUTPUT_DIR = PROJECT_ROOT / "data" / "spx"
SPLIT_DATE = "2018-01-01"

DROP_COLUMNS = ["open_to_close", "open_price", "nobs", "close_time", "open_time"]


def load_oxfordman_data(csv_path=RAW_DATA_PATH):
    """Load the raw Oxford-Man realized volatility data."""
    raw_data = pd.read_csv(csv_path)
    raw_data = raw_data.rename(columns={"Unnamed: 0": "Date"})
    raw_data = raw_data.drop(columns=DROP_COLUMNS, errors="ignore")
    return raw_data


def prepare_market_data(raw_data, symbol=".SPX"):
    """Prepare one market using the same cleaning steps as the notebook."""
    market_data = raw_data.loc[raw_data["Symbol"] == symbol].copy()
    if market_data.empty:
        raise ValueError(f"No rows found for symbol {symbol!r}.")

    market_data["Date"] = market_data["Date"].str[0:11]
    market_data["Date"] = pd.to_datetime(market_data["Date"])
    market_data.set_index("Date", inplace=True)

    market_data["close_price_shifted"] = market_data["close_price"].shift(1)
    market_data["return"] = 100 * (
        np.log(market_data["close_price"])
        - np.log(market_data["close_price_shifted"])
    )

    market_data = market_data.drop(columns=["close_price", "close_price_shifted"])
    market_data = market_data.iloc[1:, 1:]
    market_data.iloc[:, :-1] = market_data.iloc[:, :-1] * 10000
    market_data = market_data[~((market_data.iloc[:, :-1] == 0).all(axis=1))]
    market_data = market_data[~((market_data.iloc[:, :-1] < 0).any(axis=1))]
    market_data = market_data[~((market_data.iloc[:, :-1] == 0).any(axis=1))]
    market_data = market_data.dropna()

    return market_data


def split_market_data(market_data, split_date=SPLIT_DATE):
    """Split realized measures and demeaned returns into in/out-of-sample sets."""
    split_timestamp = pd.Timestamp(split_date)

    insample = market_data[market_data.index < split_timestamp]
    outsample = market_data[market_data.index >= split_timestamp]

    returns_insample = insample["return"] - np.average(insample["return"])
    returns_outsample = outsample["return"] - np.average(outsample["return"])

    return insample, outsample, returns_insample, returns_outsample


def save_spx_data(
    data_spx,
    insample,
    outsample,
    returns_insample,
    returns_outsample,
    output_dir=SPX_OUTPUT_DIR,
):
    """Save the SPX data files used by the notebook."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_spx.to_csv(output_dir / "data_SPX.csv")
    insample.to_csv(output_dir / "insample_SPX.csv")
    outsample.to_csv(output_dir / "outsample_SPX.csv")
    returns_insample.to_csv(output_dir / "returns_insample_SPX.csv")
    returns_outsample.to_csv(output_dir / "returns_outsample_SPX.csv")


def prepare_spx_data(csv_path=RAW_DATA_PATH, output_dir=SPX_OUTPUT_DIR, save=True):
    """Prepare SPX data and return notebook-compatible dictionaries."""
    raw_data = load_oxfordman_data(csv_path)
    data_spx = prepare_market_data(raw_data, symbol=".SPX")
    insample, outsample, ret_insample, ret_outsample = split_market_data(data_spx)

    if save:
        save_spx_data(data_spx, insample, outsample, ret_insample, ret_outsample, output_dir)

    return {
        "market_list_2": ["SPX"],
        "data_SPX": data_spx,
        "rms_insample": {"SPX": insample.iloc[:, :-1]},
        "rms_outsample": {"SPX": outsample.iloc[:, :-1]},
        "returns_insample": {"SPX": ret_insample.rename("return")},
        "returns_outsample": {"SPX": ret_outsample.rename("return")},
        "insample": {"SPX": insample},
        "outsample": {"SPX": outsample},
    }


def load_saved_spx_data(output_dir=SPX_OUTPUT_DIR):
    """Load previously saved SPX files and return notebook-compatible dictionaries."""
    output_dir = Path(output_dir)

    data_spx = pd.read_csv(output_dir / "data_SPX.csv", index_col=0)
    insample = pd.read_csv(output_dir / "insample_SPX.csv", index_col=0)
    outsample = pd.read_csv(output_dir / "outsample_SPX.csv", index_col=0)
    ret_insample = pd.read_csv(
        output_dir / "returns_insample_SPX.csv", index_col=0
    )["return"]
    ret_outsample = pd.read_csv(
        output_dir / "returns_outsample_SPX.csv", index_col=0
    )["return"]

    for frame in (data_spx, insample, outsample, ret_insample, ret_outsample):
        frame.index = pd.to_datetime(frame.index)

    return {
        "market_list_2": ["SPX"],
        "data_SPX": data_spx,
        "rms_insample": {"SPX": insample.iloc[:, :-1]},
        "rms_outsample": {"SPX": outsample.iloc[:, :-1]},
        "returns_insample": {"SPX": ret_insample.rename("return")},
        "returns_outsample": {"SPX": ret_outsample.rename("return")},
        "insample": {"SPX": insample},
        "outsample": {"SPX": outsample},
    }


def get_spx_data(csv_path=RAW_DATA_PATH, output_dir=SPX_OUTPUT_DIR):
    """Prepare SPX from raw data when available, otherwise load saved SPX files."""
    if Path(csv_path).exists():
        return prepare_spx_data(csv_path=csv_path, output_dir=output_dir, save=True)
    return load_saved_spx_data(output_dir=output_dir)


if __name__ == "__main__":
    prepared = get_spx_data()
    market = prepared["market_list_2"][0]
    print(f"Prepared {market}")
    print(f"In-sample rows: {len(prepared['rms_insample'][market])}")
    print(f"Out-of-sample rows: {len(prepared['rms_outsample'][market])}")
    print(f"Saved files to: {SPX_OUTPUT_DIR}")
