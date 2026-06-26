"""Fast smoke test for SPX model-section wrappers before the Autoencoder."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_spx_data
from src.model_sections import (
    build_avg_measures,
    collect_lw_results,
    collect_nlw_results,
    collect_single_measure_results,
    run_avg_realgarch_spx,
    run_ica_realgarch_spx,
    run_lw_realgarch_spx,
    run_nlw_realgarch_spx,
    run_pca_ica_spx,
    run_pca_realgarch_spx,
    run_rv5_realgarch_spx,
    slice_model_inputs,
)


def main():
    spx_data = get_spx_data()
    market_list, rms_in, rms_out, ret_in, ret_out = slice_model_inputs(
        spx_data["market_list_2"],
        spx_data["rms_insample"],
        spx_data["rms_outsample"],
        spx_data["returns_insample"],
        spx_data["returns_outsample"],
        insample_tail=80,
        outsample_head=2,
    )
    print("smoke data", market_list, rms_in["SPX"].shape, rms_out["SPX"].shape)

    lw = run_lw_realgarch_spx(
        market_list, rms_in, rms_out, ret_in, ret_out, use_cache=False, save=False
    )
    lw_df, lw_nll, lw_syn = collect_lw_results(lw, rms_in, verbose=False)
    print("lw ok", lw_df["SPX"].shape, len(lw_nll), len(lw_syn["SPX"]))

    nlw = run_nlw_realgarch_spx(
        market_list, rms_in, rms_out, ret_in, ret_out, use_cache=False, save=False
    )
    nlw_df, nlw_nll, nlw_syn = collect_nlw_results(nlw, rms_in, verbose=False)
    print("nlw ok", nlw_df["SPX"].shape, len(nlw_nll), len(nlw_syn["SPX"]))

    rms_avg_in, rms_avg_out = build_avg_measures(market_list, rms_in, rms_out)
    avg = run_avg_realgarch_spx(
        market_list, rms_avg_in, rms_avg_out, ret_in, ret_out, use_cache=False, save=False
    )
    avg_df, avg_nll = collect_single_measure_results(avg, "AVG", verbose=False)
    print("avg ok", avg_df["SPX"].shape, len(avg_nll))

    rv5 = run_rv5_realgarch_spx(
        market_list, rms_in, rms_out, ret_in, ret_out, use_cache=False, save=False
    )
    rv5_df, rv5_nll = collect_single_measure_results(rv5, "RV5", verbose=False)
    print("rv5 ok", rv5_df["SPX"].shape, len(rv5_nll))

    pca_results, ica_results = run_pca_ica_spx(
        market_list, rms_in, rms_out, use_cache=False, save=False
    )
    print("pca/ica series ok", pca_results["SPX"].shape, ica_results["SPX"].shape)

    pca = run_pca_realgarch_spx(
        market_list, pca_results, ret_in, ret_out, use_cache=False, save=False
    )
    pca_df, pca_nll = collect_single_measure_results(pca, "PCA", verbose=False)
    print("pca rgarch ok", pca_df["SPX"].shape, len(pca_nll))

    ica = run_ica_realgarch_spx(
        market_list, ica_results, ret_in, ret_out, use_cache=False, save=False
    )
    ica_df, ica_nll = collect_single_measure_results(ica, "ICA", verbose=False)
    print("ica rgarch ok", ica_df["SPX"].shape, len(ica_nll))
    print("model sections smoke ok")


if __name__ == "__main__":
    main()
