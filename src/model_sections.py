"""SPX-only runners for the main AERGARCH notebook sections.

The functions here wrap the original model functions without changing their
estimation logic. They keep the old notebook variable shapes so later sections
can still use the same dictionaries.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .aergarch_functions import (
    lw_rgarch_lg_fit_pred_process,
    nlw_rgarch_lg_fit_pred_process,
    rgarch_lg_fit_pred_process,
    rgarch_lg_dr_fit_pred_process,
    rolling_pca_ica,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "spx"
MARKET = "SPX"

WEIGHT_COL_NAMES = [
    "rk_th2",
    "medrv",
    "rsv",
    "rv5",
    "rk_twoscale",
    "rsv_ss",
    "bv_ss",
    "rk_parzen",
    "bv",
    "rv10",
    "rv10_ss",
    "rv5_ss",
]

__all__ = [
    "RESULTS_DIR",
    "MARKET",
    "WEIGHT_COL_NAMES",
    "slice_model_inputs",
    "build_avg_measures",
    "run_lw_realgarch_spx",
    "collect_lw_results",
    "run_nlw_realgarch_spx",
    "collect_nlw_results",
    "run_avg_realgarch_spx",
    "collect_single_measure_results",
    "run_rv5_realgarch_spx",
    "run_pca_ica_spx",
    "run_pca_realgarch_spx",
    "run_ica_realgarch_spx",
]


def _result_path(filename):
    return RESULTS_DIR / filename


def _load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def _save_pickle(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def _get_or_compute(filename, compute_fn, use_cache=True, overwrite=False, save=True):
    path = _result_path(filename)
    if use_cache and path.exists() and not overwrite:
        return _load_pickle(path)

    result = compute_fn()
    if save:
        _save_pickle(result, path)
    return result


def _model_markets(market_list):
    return list(market_list)


def slice_model_inputs(
    market_list,
    rms_insample,
    rms_outsample,
    returns_insample,
    returns_outsample,
    insample_tail=120,
    outsample_head=3,
):
    """Create a small real-data slice for fast smoke checks."""
    sliced_rms_insample = {}
    sliced_rms_outsample = {}
    sliced_returns_insample = {}
    sliced_returns_outsample = {}

    for market in market_list:
        sliced_rms_insample[market] = rms_insample[market].tail(insample_tail)
        sliced_rms_outsample[market] = rms_outsample[market].head(outsample_head)
        sliced_returns_insample[market] = returns_insample[market].tail(insample_tail)
        sliced_returns_outsample[market] = returns_outsample[market].head(outsample_head)

    return (
        _model_markets(market_list),
        sliced_rms_insample,
        sliced_rms_outsample,
        sliced_returns_insample,
        sliced_returns_outsample,
    )


def _run_realgarch_for_measure(
    market_list,
    measures_insample,
    measures_outsample,
    returns_insample,
    returns_outsample,
    fit_function,
):
    results = []
    for market in _model_markets(market_list):
        results.append(
            fit_function(
                market,
                measures_insample[market],
                measures_outsample[market],
                returns_insample[market],
                returns_outsample[market],
            )
        )
    return results


def run_lw_realgarch_spx(
    market_list,
    rms_insample,
    rms_outsample,
    returns_insample,
    returns_outsample,
    use_cache=True,
    overwrite=False,
    save=True,
):
    return _get_or_compute(
        "lw_rgarch_spx.pkl",
        lambda: _run_realgarch_for_measure(
            market_list,
            rms_insample,
            rms_outsample,
            returns_insample,
            returns_outsample,
            lw_rgarch_lg_fit_pred_process,
        ),
        use_cache=use_cache,
        overwrite=overwrite,
        save=save,
    )


def collect_lw_results(results, rms_insample, verbose=True):
    allparams = {}
    forecast_nll = []
    synxt_insample = {}

    for result in results:
        (
            name,
            insample_weights,
            insample_params,
            fit_nll,
            forecast_nll_value,
            eachstep_forecast_nll,
            forecast_var,
            con_var,
            params_df,
            syn_xt_list,
            weights_frame,
        ) = result

        omega, beta, gamma, xi, phi, tau1, tau2, sigma_mu = insample_params
        forecast_nll.append((name, forecast_nll_value))
        test_index = params_df.index

        final_market_df = pd.concat(
            [
                pd.Series(forecast_var, index=test_index, name="F_var"),
                pd.Series(
                    eachstep_forecast_nll, index=test_index, name="Each_f_nLL"
                ),
                pd.Series(
                    syn_xt_list[len(con_var) - 1 : -1],
                    index=test_index,
                    name="Syn_xt",
                ),
                params_df,
                pd.DataFrame(
                    weights_frame,
                    index=test_index,
                    columns=WEIGHT_COL_NAMES[: weights_frame.shape[1]],
                ),
            ],
            axis=1,
        )

        allparams[name] = final_market_df
        synxt_insample[name] = pd.Series(
            syn_xt_list[: len(con_var)],
            index=rms_insample[name].index,
            name=name,
        )

        if verbose:
            print(f"LW-RGARCH-lg-{name}")
            print(f"omega_{name}: {round(omega, 4)}")
            print(f"beta_{name}: {round(beta, 4)}")
            print(f"gamma_{name}: {round(gamma, 4)}")
            print(f"xi_{name}: {round(xi, 4)}")
            print(f"phi_{name}: {round(phi, 4)}")
            print(f"tau1_{name}: {round(tau1, 4)}")
            print(f"tau2_{name}: {round(tau2, 4)}")
            print(f"sigma_mu_{name}: {round(sigma_mu, 4)}")
            print(f"weights_{name}: {insample_weights}")
            print(f"LW-RGARCH-lg-{name} Trained negative log-likelihood: {round(fit_nll, 4)}")
            print(f"LW-RGARCH-lg-{name} Predictive negative log-likelihood: {round(forecast_nll_value, 4)}")
            print("-" * 80)

    return allparams, forecast_nll, synxt_insample


def run_nlw_realgarch_spx(
    market_list,
    rms_insample,
    rms_outsample,
    returns_insample,
    returns_outsample,
    use_cache=True,
    overwrite=False,
    save=True,
):
    return _get_or_compute(
        "nlw_rgarch_spx.pkl",
        lambda: _run_realgarch_for_measure(
            market_list,
            rms_insample,
            rms_outsample,
            returns_insample,
            returns_outsample,
            nlw_rgarch_lg_fit_pred_process,
        ),
        use_cache=use_cache,
        overwrite=overwrite,
        save=save,
    )


def collect_nlw_results(results, rms_insample, verbose=True):
    allparams = {}
    forecast_nll = []
    synxt_insample = {}

    for result in results:
        (
            name,
            insample_ab,
            insample_weights,
            insample_params,
            fit_nll,
            forecast_nll_value,
            eachstep_forecast_nll,
            forecast_var,
            con_var,
            params_df,
            syn_xt_list,
            ab_frame,
            weights_frame,
        ) = result

        omega, beta, gamma, xi, phi, tau1, tau2, sigma_mu = insample_params
        forecast_nll.append((name, forecast_nll_value))
        test_index = params_df.index

        final_market_df = pd.concat(
            [
                pd.Series(forecast_var, index=test_index, name="F_var"),
                pd.Series(
                    eachstep_forecast_nll, index=test_index, name="Each_f_nLL"
                ),
                pd.Series(
                    syn_xt_list[len(con_var) - 1 : -1],
                    index=test_index,
                    name="Syn_xt",
                ),
                params_df,
                pd.DataFrame(ab_frame, index=test_index, columns=["a", "b"]),
                pd.DataFrame(
                    weights_frame,
                    index=test_index,
                    columns=WEIGHT_COL_NAMES[: weights_frame.shape[1]],
                ),
            ],
            axis=1,
        )

        allparams[name] = final_market_df
        synxt_insample[name] = pd.Series(
            syn_xt_list[: len(con_var)],
            index=rms_insample[name].index,
            name=name,
        )

        if verbose:
            print(f"NLW-RGARCH-lg-{name}")
            print(f"omega_{name}: {round(omega, 4)}")
            print(f"beta_{name}: {round(beta, 4)}")
            print(f"gamma_{name}: {round(gamma, 4)}")
            print(f"xi_{name}: {round(xi, 4)}")
            print(f"phi_{name}: {round(phi, 4)}")
            print(f"tau1_{name}: {round(tau1, 4)}")
            print(f"tau2_{name}: {round(tau2, 4)}")
            print(f"sigma_mu_{name}: {round(sigma_mu, 4)}")
            print(f"ab_{name}: {insample_ab}")
            print(f"weights_{name}: {insample_weights}")
            print(f"NLW-RGARCH-lg-{name} Trained negative log-likelihood: {round(fit_nll, 4)}")
            print(f"NLW-RGARCH-lg-{name} Predictive negative log-likelihood: {round(forecast_nll_value, 4)}")
            print("-" * 80)

    return allparams, forecast_nll, synxt_insample


def build_avg_measures(market_list, rms_insample, rms_outsample):
    rms_insample_avg = {}
    rms_outsample_avg = {}
    for market in _model_markets(market_list):
        rms_insample_avg[market] = rms_insample[market].mean(axis=1)
        rms_outsample_avg[market] = rms_outsample[market].mean(axis=1)
    return rms_insample_avg, rms_outsample_avg


def run_avg_realgarch_spx(
    market_list,
    rms_insample_avg,
    rms_outsample_avg,
    returns_insample,
    returns_outsample,
    use_cache=True,
    overwrite=False,
    save=True,
):
    return _get_or_compute(
        "avg_rgarch_spx.pkl",
        lambda: _run_realgarch_for_measure(
            market_list,
            rms_insample_avg,
            rms_outsample_avg,
            returns_insample,
            returns_outsample,
            rgarch_lg_fit_pred_process,
        ),
        use_cache=use_cache,
        overwrite=overwrite,
        save=save,
    )


def run_rv5_realgarch_spx(
    market_list,
    rms_insample,
    rms_outsample,
    returns_insample,
    returns_outsample,
    use_cache=True,
    overwrite=False,
    save=True,
):
    rv5_insample = {}
    rv5_outsample = {}
    for market in _model_markets(market_list):
        rv5_insample[market] = rms_insample[market]["rv5"]
        rv5_outsample[market] = rms_outsample[market]["rv5"]

    return _get_or_compute(
        "rv5_rgarch_spx.pkl",
        lambda: _run_realgarch_for_measure(
            market_list,
            rv5_insample,
            rv5_outsample,
            returns_insample,
            returns_outsample,
            rgarch_lg_fit_pred_process,
        ),
        use_cache=use_cache,
        overwrite=overwrite,
        save=save,
    )


def collect_single_measure_results(results, model_label, verbose=True):
    allparams = {}
    forecast_nll = []

    for result in results:
        (
            name,
            insample_params,
            fit_nll,
            forecast_nll_value,
            eachstep_forecast_nll,
            forecast_var,
            params_df,
        ) = result

        omega, beta, gamma, xi, phi, tau1, tau2, sigma_mu = insample_params
        forecast_nll.append((name, forecast_nll_value))
        test_index = params_df.index

        final_market_df = pd.concat(
            [
                pd.Series(forecast_var, index=test_index, name="F_var"),
                pd.Series(
                    eachstep_forecast_nll, index=test_index, name="Each_f_nLL"
                ),
                params_df,
            ],
            axis=1,
        )
        allparams[name] = final_market_df

        if verbose:
            print(f"{model_label}-RGARCH-lg-{name}")
            print(f"omega_{name}: {round(omega, 4)}")
            print(f"beta_{name}: {round(beta, 4)}")
            print(f"gamma_{name}: {round(gamma, 4)}")
            print(f"xi_{name}: {round(xi, 4)}")
            print(f"phi_{name}: {round(phi, 4)}")
            print(f"tau1_{name}: {round(tau1, 4)}")
            print(f"tau2_{name}: {round(tau2, 4)}")
            print(f"sigma_mu_{name}: {round(sigma_mu, 4)}")
            print(f"{model_label}-RGARCH-lg-{name} Trained negative log-likelihood: {round(fit_nll, 4)}")
            print(f"{model_label}-RGARCH-lg-{name} Predictive negative log-likelihood: {round(forecast_nll_value, 4)}")
            print("-" * 80)

    return allparams, forecast_nll


def run_pca_ica_spx(
    market_list,
    rms_insample,
    rms_outsample,
    use_cache=True,
    overwrite=False,
    save=True,
):
    pca_path = _result_path("pca_results_spx.pkl")
    ica_path = _result_path("ica_results_spx.pkl")

    if use_cache and pca_path.exists() and ica_path.exists() and not overwrite:
        return _load_pickle(pca_path), _load_pickle(ica_path)

    pca_results = {}
    ica_results = {}
    for market in _model_markets(market_list):
        name, pca_df, ica_df = rolling_pca_ica(
            market,
            rms_insample[market],
            rms_outsample[market],
        )
        pca_results[name] = pca_df
        ica_results[name] = ica_df

    if save:
        _save_pickle(pca_results, pca_path)
        _save_pickle(ica_results, ica_path)

    return pca_results, ica_results


def _run_dr_realgarch(
    market_list,
    dimension_reduced_results,
    returns_insample,
    returns_outsample,
    filename,
    use_cache=True,
    overwrite=False,
    save=True,
):
    def compute():
        results = []
        for market in _model_markets(market_list):
            results.append(
                rgarch_lg_dr_fit_pred_process(
                    market,
                    dimension_reduced_results[market],
                    returns_insample[market],
                    returns_outsample[market],
                )
            )
        return results

    return _get_or_compute(
        filename,
        compute,
        use_cache=use_cache,
        overwrite=overwrite,
        save=save,
    )


def run_pca_realgarch_spx(
    market_list,
    pca_results,
    returns_insample,
    returns_outsample,
    use_cache=True,
    overwrite=False,
    save=True,
):
    return _run_dr_realgarch(
        market_list,
        pca_results,
        returns_insample,
        returns_outsample,
        "pca_rgarch_spx.pkl",
        use_cache=use_cache,
        overwrite=overwrite,
        save=save,
    )


def run_ica_realgarch_spx(
    market_list,
    ica_results,
    returns_insample,
    returns_outsample,
    use_cache=True,
    overwrite=False,
    save=True,
):
    return _run_dr_realgarch(
        market_list,
        ica_results,
        returns_insample,
        returns_outsample,
        "ica_rgarch_spx.pkl",
        use_cache=use_cache,
        overwrite=overwrite,
        save=save,
    )

