"""VaR/ES loss and MCS helpers for the SPX-only AERGARCH notebook."""

import numpy as np
import pandas as pd
from scipy.stats import norm


DEFAULT_ALPHA = 0.025
DEFAULT_MCS_ALPHA = 0.25
DEFAULT_MODEL_MAPPING = {
    "lw": "lw",
    "fw": "fw",
    "avg": "avg",
    "ae": "12-12-1 ae",
    "pca": "pca",
    "ica": "ica",
    "rv5": "rv5",
}
DEFAULT_MODEL_KEYS = ["lw", "fw", "avg", "ae", "pca", "ica", "rv5"]

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_MCS_ALPHA",
    "DEFAULT_MODEL_MAPPING",
    "DEFAULT_MODEL_KEYS",
    "qloss",
    "get_qloss_series",
    "jointloss",
    "get_jointloss_series",
    "build_model_data_dict",
    "add_var_es_to_models",
    "add_loss_series_to_models",
    "compute_covid_loss_tables",
    "run_mcs_for_market",
    "run_mcs_all_markets",
    "format_included",
]


def qloss(y_true, var_pred, alpha=DEFAULT_ALPHA):
    y = np.array(y_true).ravel()
    q = np.array(var_pred).ravel()

    resid = y - q
    indicator = (y <= q).astype(float)
    loss = (alpha - indicator) * resid

    return np.mean(loss)


def get_qloss_series(y_true, var_pred, alpha=DEFAULT_ALPHA):
    y = np.array(y_true).ravel()
    q = np.array(var_pred).ravel()

    resid = y - q
    indicator = (y <= q).astype(float)
    loss = (alpha - indicator) * resid

    return loss


def jointloss(y_true, var_pred, es_pred, alpha=DEFAULT_ALPHA):
    y = np.array(y_true).ravel()
    q = np.array(var_pred).ravel()
    es = np.array(es_pred).ravel()

    indicator = (y <= q).astype(float)
    term_1 = -np.log((alpha - 1) / es)
    numerator = (y - q) * (alpha - indicator)
    denominator = alpha * es
    term_2 = numerator / denominator
    return np.mean(term_1 - term_2)


def get_jointloss_series(y_true, var_pred, es_pred, alpha=DEFAULT_ALPHA):
    y = np.array(y_true).ravel()
    q = np.array(var_pred).ravel()
    es = np.array(es_pred).ravel()

    indicator = (y <= q).astype(float)
    term_1 = -np.log((alpha - 1) / es)
    numerator = (y - q) * (alpha - indicator)
    denominator = alpha * es
    term_2 = numerator / denominator
    return term_1 - term_2


def build_model_data_dict(
    lw_allparams_df,
    fw_allparams_df,
    avg_allparams_df,
    ae_allparams_df,
    pca_allparams_df,
    ica_allparams_df,
    rv5_allparams_df,
):
    return {
        "lw": lw_allparams_df,
        "fw": fw_allparams_df,
        "avg": avg_allparams_df,
        "ae": ae_allparams_df,
        "pca": pca_allparams_df,
        "ica": ica_allparams_df,
        "rv5": rv5_allparams_df,
    }


def add_var_es_to_models(model_data, markets_list, alpha=DEFAULT_ALPHA):
    var_factor = norm.ppf(alpha)
    es_factor = -norm.pdf(norm.ppf(alpha)) / alpha

    for market in markets_list:
        for model_dict in model_data.values():
            model_dict[market]["VaR"] = var_factor * np.sqrt(model_dict[market]["F_var"])
            model_dict[market]["ES"] = es_factor * np.sqrt(model_dict[market]["F_var"])

    return model_data


def add_loss_series_to_models(
    model_data,
    returns_outsample,
    markets_list,
    model_keys=None,
    alpha=DEFAULT_ALPHA,
):
    model_keys = DEFAULT_MODEL_KEYS if model_keys is None else model_keys

    for key in model_keys:
        for market in markets_list:
            df = model_data[key][market]

            actuals = returns_outsample[market].values.ravel()
            var_fore = df["VaR"].values.ravel()
            es_fore = df["ES"].values.ravel()

            qloss_series = get_qloss_series(actuals, var_fore, alpha=alpha)
            joint_series = get_jointloss_series(actuals, var_fore, es_fore, alpha=alpha)

            df["QLoss"] = qloss_series
            df["JointLoss"] = joint_series

    return model_data


def compute_covid_loss_tables(
    loaded_data,
    returns_outsample,
    markets_list,
    model_mapping=None,
    start_date="2020-01-30",
    end_date="2020-06-30",
    alpha=DEFAULT_ALPHA,
):
    model_mapping = DEFAULT_MODEL_MAPPING if model_mapping is None else model_mapping

    covid_qloss_rows = []
    covid_jointloss_rows = []

    for market in markets_list:
        q_data = {"Market": market}
        j_data = {"Market": market}

        actuals_series = returns_outsample[market].loc[start_date:end_date]
        actuals = actuals_series.values.ravel()

        for key, display_name in model_mapping.items():
            df = loaded_data[key][market].loc[start_date:end_date]
            var_fore = df["VaR"].values.ravel()
            es_fore = df["ES"].values.ravel()

            q_val = qloss(actuals, var_fore, alpha=alpha)
            j_val = jointloss(actuals, var_fore, es_fore, alpha=alpha)

            q_data[display_name] = q_val
            j_data[display_name] = j_val

        covid_qloss_rows.append(q_data)
        covid_jointloss_rows.append(j_data)

    cols = ["Market"] + list(model_mapping.values())

    covid_qloss_result = pd.DataFrame(covid_qloss_rows, columns=cols)
    covid_jointloss_result = pd.DataFrame(covid_jointloss_rows, columns=cols)

    return covid_qloss_result, covid_jointloss_result


def run_mcs_for_market(
    market_name,
    loss_col_name,
    alpha,
    input_data_dict,
    model_map,
    start_date=None,
    end_date=None,
):
    from arch.bootstrap import MCS

    loss_data = {}
    for key, display_name in model_map.items():
        series = input_data_dict[key][market_name][loss_col_name]
        if start_date and end_date:
            series = series.loc[start_date:end_date]
        loss_data[display_name] = series

    df_losses = pd.DataFrame(loss_data)
    mcs = MCS(df_losses, size=alpha, block_size=10)
    mcs.compute()
    return mcs.pvalues, mcs.included


def _coerce_pvalues(pvals):
    if isinstance(pvals, pd.DataFrame):
        return pvals.iloc[:, 0]
    return pvals


def run_mcs_all_markets(
    markets_list,
    loaded_data,
    model_mapping=None,
    mcs_alpha=DEFAULT_MCS_ALPHA,
    start_date=None,
    end_date=None,
):
    model_mapping = DEFAULT_MODEL_MAPPING if model_mapping is None else model_mapping

    mcs_qloss_included = {}
    mcs_jointloss_included = {}
    mcs_pnll_included = {}

    mcs_qloss_results = {}
    mcs_jointloss_results = {}
    mcs_pnll_results = {}

    for market in markets_list:
        q_res = run_mcs_for_market(
            market_name=market,
            loss_col_name="QLoss",
            alpha=mcs_alpha,
            input_data_dict=loaded_data,
            model_map=model_mapping,
            start_date=start_date,
            end_date=end_date,
        )
        if q_res:
            pvals, included = q_res
            mcs_qloss_results[market] = _coerce_pvalues(pvals)
            mcs_qloss_included[market] = included

        j_res = run_mcs_for_market(
            market_name=market,
            loss_col_name="JointLoss",
            alpha=mcs_alpha,
            input_data_dict=loaded_data,
            model_map=model_mapping,
            start_date=start_date,
            end_date=end_date,
        )
        if j_res:
            pvals, included = j_res
            mcs_jointloss_results[market] = _coerce_pvalues(pvals)
            mcs_jointloss_included[market] = included

        pnll_res = run_mcs_for_market(
            market_name=market,
            loss_col_name="Each_f_nLL",
            alpha=mcs_alpha,
            input_data_dict=loaded_data,
            model_map=model_mapping,
            start_date=start_date,
            end_date=end_date,
        )
        if pnll_res:
            pvals, included = pnll_res
            mcs_pnll_results[market] = _coerce_pvalues(pvals)
            mcs_pnll_included[market] = included

    df_mcs_qloss = pd.DataFrame.from_dict(mcs_qloss_results, orient="index")
    df_mcs_jointloss = pd.DataFrame.from_dict(mcs_jointloss_results, orient="index")
    df_mcs_pnll = pd.DataFrame.from_dict(mcs_pnll_results, orient="index")

    df_included_qloss = format_included(mcs_qloss_included).to_frame(
        name="Included_Models"
    )
    df_included_jointloss = format_included(mcs_jointloss_included).to_frame(
        name="Included_Models"
    )
    df_included_pnll = format_included(mcs_pnll_included).to_frame(
        name="Included_Models"
    )

    return {
        "mcs_qloss_included": mcs_qloss_included,
        "mcs_jointloss_included": mcs_jointloss_included,
        "mcs_pnll_included": mcs_pnll_included,
        "mcs_qloss_results": mcs_qloss_results,
        "mcs_jointloss_results": mcs_jointloss_results,
        "mcs_pnll_results": mcs_pnll_results,
        "df_mcs_qloss": df_mcs_qloss,
        "df_mcs_jointloss": df_mcs_jointloss,
        "df_mcs_pnll": df_mcs_pnll,
        "df_included_qloss": df_included_qloss,
        "df_included_jointloss": df_included_jointloss,
        "df_included_pnll": df_included_pnll,
    }


def format_included(included_dict):
    return pd.Series(included_dict).apply(lambda x: ", ".join(x))
