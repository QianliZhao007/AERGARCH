"""Core functions extracted from ``AERGARCH.ipynb``.

This module keeps the original notebook function names so the notebook can
import them directly while the project is gradually cleaned for GitHub.
"""

import random

try:
    import torch
except ModuleNotFoundError:
    torch = None

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gamma
from sklearn import decomposition


DEFAULT_SEED = 42

__all__ = [
    "DEFAULT_SEED",
    "set_seed",
    "pred_nll",
    "rgarch_params_constraints",
    "rgarch_log_nll",
    "condi_var_rgarch_lg",
    "rgarch_lg_rolling_forecast",
    "rgarch_lg_fit_pred_process",
    "para_constraint",
    "weight_constraint",
    "lw_rgarch_lg_nll",
    "condi_var_lw_rgarch_lg",
    "lw_rgarch_lg_rolling_forecast",
    "lw_rgarch_lg_fit_pred_process",
    "beta_weights",
    "nlw_rgarch_lg_nll",
    "rnlw_garch_constraint",
    "rnlw_garch_constraints",
    "rnlw_garch_bounds",
    "condi_var_nlw_rgarch_lg",
    "nlw_rgarch_lg_rolling_forecast",
    "nlw_rgarch_lg_fit_pred_process",
    "rgarch_lg_dr_rolling_forecast",
    "rgarch_lg_dr_fit_pred_process",
    "rolling_pca_ica",
]


def _pos_value(values, position):
    """Return one value by position for pandas or array-like inputs."""
    if hasattr(values, "iloc"):
        return values.iloc[position]
    return values[position]


def _slice_pos(values, start=None, stop=None):
    """Return a positional slice for pandas or array-like inputs."""
    if hasattr(values, "iloc"):
        return values.iloc[slice(start, stop)]
    return values[slice(start, stop)]


def _last_value(values):
    """Return the last value for pandas or array-like inputs."""
    return _pos_value(values, -1)


def set_seed(seed):
    """Set random seeds for reproducible CPU/GPU results."""
    np.random.seed(seed)
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pred_nll(returns, pred_var):
    """Predictive negative log-likelihood, omitting constant terms."""
    pred_ll = -np.sum(np.log(pred_var) + (returns**2 / pred_var))
    return -pred_ll


def rgarch_params_constraints(params, epsilon=1e-6):
    omega, beta, gamma_param, xi, phi, tau1, tau2, sigma_mu = params
    beta_gamma_phi = beta + gamma_param * phi
    c1 = beta_gamma_phi - epsilon
    c2 = 1 - beta_gamma_phi - epsilon
    return np.array([c1, c2])


def rgarch_log_nll(params, returns, real_var):
    omega, beta, gamma_param, xi, phi, tau1, tau2, sigma_mu = params

    rt = np.asarray(returns)
    xt = np.asarray(real_var)
    n = len(rt)

    ht = np.zeros(n)
    zt = np.zeros(n)
    mut = np.zeros(n)

    ht[0] = np.sum(rt**2) / n
    mut[0] = np.sum(xt) / n
    zt[0] = rt[0] / np.sqrt(ht[0])

    for i in range(1, n):
        ht[i] = np.exp(
            omega + beta * np.log(ht[i - 1]) + gamma_param * np.log(xt[i - 1])
        )
        zt[i] = rt[i] / np.sqrt(ht[i])
        mut[i] = (
            np.log(xt[i])
            - xi
            - phi * np.log(ht[i])
            - tau1 * zt[i]
            - tau2 * (zt[i] ** 2 - 1)
        )

    ll_garch = -np.sum(np.log(ht) + (rt**2 / ht))
    ll_measure = -np.sum(np.log(sigma_mu**2) + (mut**2) / (sigma_mu**2))
    return -(ll_garch + ll_measure)


def condi_var_rgarch_lg(rt_train, xt_train, params):
    omega, beta, gamma_param, xi, phi, tau1, tau2, sigma_mu = params

    rt = np.asarray(rt_train)
    xt = np.asarray(xt_train)
    n = len(rt)

    ht = np.zeros(n)
    zt = np.zeros(n)
    mut = np.zeros(n)

    ht[0] = np.sum(rt**2) / n
    mut[0] = np.sum(xt) / n
    zt[0] = rt[0] / np.sqrt(ht[0])

    for i in range(1, n):
        ht[i] = np.exp(
            omega + beta * np.log(ht[i - 1]) + gamma_param * np.log(xt[i - 1])
        )
        zt[i] = rt[i] / np.sqrt(ht[i])
        mut[i] = (
            np.log(xt[i])
            - xi
            - phi * np.log(ht[i])
            - tau1 * zt[i]
            - tau2 * (zt[i] ** 2 - 1)
        )
    return ht


def rgarch_lg_rolling_forecast(returns_train, returns_test, rv_train, rv_test, params):
    rgarch_bounds = [
        (0, 5),
        (0, 5),
        (0, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (0, 10),
    ]
    rgarch_constraints = [{"type": "ineq", "fun": rgarch_params_constraints}]

    n = len(returns_test)

    forecast_var_rg_lg = np.zeros(n)
    eachstep_forecast_nll = np.zeros(n)
    omega_list = np.zeros(n)
    beta_list = np.zeros(n)
    gamma_list = np.zeros(n)
    xi_list = np.zeros(n)
    phi_list = np.zeros(n)
    tau1_list = np.zeros(n)
    tau2_list = np.zeros(n)
    sigma_mu_list = np.zeros(n)

    current_estimates = params.copy()
    for t in range(n):
        rolling_rv = pd.concat([_slice_pos(rv_train, t), _slice_pos(rv_test, None, t)])
        rolling_returns = pd.concat(
            [_slice_pos(returns_train, t), _slice_pos(returns_test, None, t)]
        )

        result = minimize(
            rgarch_log_nll,
            current_estimates,
            args=(rolling_returns, rolling_rv),
            method="SLSQP",
            bounds=rgarch_bounds,
            constraints=rgarch_constraints,
        )

        current_estimates = result.x
        omega_new, beta_new, gamma_new, xi_new, phi_new, tau1_new, tau2_new, sigma_mu_new = (
            current_estimates
        )
        con_var_new = condi_var_rgarch_lg(rolling_returns, rolling_rv, current_estimates)

        if t == 0:
            garch_fit_nll = pred_nll(returns_train, con_var_new)
            insample_params = current_estimates

        forecast_var_rg_lg[t] = np.exp(
            omega_new
            + beta_new * np.log(con_var_new[-1])
            + gamma_new * np.log(_last_value(rolling_rv))
        )
        eachstep_forecast_nll[t] = pred_nll(
            _pos_value(returns_test, t), forecast_var_rg_lg[t]
        )
        omega_list[t] = omega_new
        beta_list[t] = beta_new
        gamma_list[t] = gamma_new
        xi_list[t] = xi_new
        phi_list[t] = phi_new
        tau1_list[t] = tau1_new
        tau2_list[t] = tau2_new
        sigma_mu_list[t] = sigma_mu_new

    return (
        insample_params,
        garch_fit_nll,
        forecast_var_rg_lg,
        eachstep_forecast_nll,
        omega_list,
        beta_list,
        gamma_list,
        xi_list,
        phi_list,
        tau1_list,
        tau2_list,
        sigma_mu_list,
    )


def rgarch_lg_fit_pred_process(
    market_name, measures_train, measures_test, returns_train, returns_test
):
    initial_params_rgarch = [0.2, 0.6, 0.2, 0.1, 0.5, 0.1, 0.1, 0.5]

    (
        insample_params,
        garch_fit_nll,
        forecast_var,
        eachstep_forecast_nll,
        omega_list,
        beta_list,
        gamma_list,
        xi_list,
        phi_list,
        tau1_list,
        tau2_list,
        sigma_mu_list,
    ) = rgarch_lg_rolling_forecast(
        returns_train, returns_test, measures_train, measures_test, initial_params_rgarch
    )

    overall_forecast_nll = pred_nll(returns_test, forecast_var)

    params_df = pd.DataFrame(
        {
            "Omega": omega_list,
            "Beta": beta_list,
            "Gamma": gamma_list,
            "Xi": xi_list,
            "Phi": phi_list,
            "Tau1": tau1_list,
            "Tau2": tau2_list,
            "Sigma_mu": sigma_mu_list,
        }
    )
    params_df.index = returns_test.index

    return (
        market_name,
        insample_params,
        garch_fit_nll,
        overall_forecast_nll,
        eachstep_forecast_nll,
        forecast_var,
        params_df,
    )


def para_constraint(estimates, num_dimen, epsilon=1e-6):
    beta, gamma_param, phi = (
        estimates[num_dimen + 1],
        estimates[num_dimen + 2],
        estimates[num_dimen + 4],
    )
    beta_gamma_phi = beta + gamma_param * phi
    c1 = beta_gamma_phi - epsilon
    c2 = 1 - beta_gamma_phi - epsilon
    return np.array([c1, c2])


def weight_constraint(estimates, num_dimen):
    weights = estimates[:num_dimen]
    return np.sum(weights) - 1


def lw_rgarch_lg_nll(estimates, rt_train, xt_train, num_dimen):
    weights = estimates[:num_dimen]
    params = estimates[num_dimen:]
    omega, beta, gamma_param, xi, phi, tau1, tau2, sigma_mu = params

    rt = np.asarray(rt_train)
    xt = np.dot(xt_train, weights)
    n = len(rt)

    ht = np.zeros(n)
    zt = np.zeros(n)
    mut = np.zeros(n)

    ht[0] = np.sum(rt**2) / n
    mut[0] = np.sum(xt) / n
    zt[0] = rt[0] / np.sqrt(ht[0])

    for i in range(1, n):
        ht[i] = np.exp(
            omega + beta * np.log(ht[i - 1]) + gamma_param * np.log(xt[i - 1])
        )
        zt[i] = rt[i] / np.sqrt(ht[i])
        mut[i] = (
            np.log(xt[i])
            - xi
            - phi * np.log(ht[i])
            - tau1 * zt[i]
            - tau2 * (zt[i] ** 2 - 1)
        )

    ll_garch = -np.sum(np.log(ht) + (rt**2 / ht))
    ll_measure = -np.sum(np.log(sigma_mu**2) + (mut**2 / sigma_mu**2))
    return -(ll_garch + ll_measure)


def condi_var_lw_rgarch_lg(estimates, rt_train, xt_train, num_dimen):
    weights = estimates[:num_dimen]
    params = estimates[num_dimen:]
    omega, beta, gamma_param, xi, phi, tau1, tau2, sigma_mu = params

    rt = np.asarray(rt_train)
    xt = np.dot(xt_train, weights)
    n = len(rt)

    ht = np.zeros(n)
    zt = np.zeros(n)
    mut = np.zeros(n)

    ht[0] = np.sum(rt**2) / n
    mut[0] = np.sum(xt) / n
    zt[0] = rt[0] / np.sqrt(ht[0])

    for i in range(1, n):
        ht[i] = np.exp(
            omega + beta * np.log(ht[i - 1]) + gamma_param * np.log(xt[i - 1])
        )
        zt[i] = rt[i] / np.sqrt(ht[i])
        mut[i] = (
            np.log(xt[i])
            - xi
            - phi * np.log(ht[i])
            - tau1 * zt[i]
            - tau2 * (zt[i] ** 2 - 1)
        )

    return ht


def lw_rgarch_lg_rolling_forecast(
    returns_train,
    returns_test,
    rv_train,
    rv_test,
    conditional_var,
    estimates,
    num_dimen,
):
    rlwgarch_bounds = [(0, 1)] * num_dimen + [
        (0, 5),
        (0, 5),
        (0, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (0, 10),
    ]
    rlwgarch_constraints = [
        {"type": "ineq", "fun": para_constraint, "args": (num_dimen,)},
        {"type": "eq", "fun": weight_constraint, "args": (num_dimen,)},
    ]

    weights = estimates[:num_dimen]
    params = estimates[num_dimen:]
    omega, beta, gamma_param, xi, phi, tau1, tau2, sigma_mu = params

    n = len(returns_test)
    n_train = len(returns_train)

    forecast_var_rg_lg = np.zeros(n)
    eachstep_forecast_nll = np.zeros(n)
    omega_list = np.zeros(n)
    beta_list = np.zeros(n)
    gamma_list = np.zeros(n)
    xi_list = np.zeros(n)
    phi_list = np.zeros(n)
    tau1_list = np.zeros(n)
    tau2_list = np.zeros(n)
    sigma_mu_list = np.zeros(n)
    syn_xt_list = np.zeros(n_train + n)
    weights_frame = np.zeros((n, num_dimen))

    xt_train = np.dot(rv_train, weights)
    initial_forecast_var = np.exp(
        omega + beta * np.log(conditional_var[-1]) + gamma_param * np.log(xt_train[-1])
    )
    initial_forecast_nll = pred_nll(_pos_value(returns_test, 0), initial_forecast_var)

    forecast_var_rg_lg[0] = initial_forecast_var
    eachstep_forecast_nll[0] = initial_forecast_nll
    omega_list[0] = omega
    beta_list[0] = beta
    gamma_list[0] = gamma_param
    xi_list[0] = xi
    phi_list[0] = phi
    tau1_list[0] = tau1
    tau2_list[0] = tau2
    sigma_mu_list[0] = sigma_mu
    syn_xt_list[:n_train] = xt_train
    weights_frame[0, :] = weights

    current_estimates = estimates.copy()

    for t in range(1, n):
        rolling_xt = np.vstack([rv_train.iloc[t:, :], rv_test.iloc[:t, :]])
        rolling_returns = pd.concat(
            [_slice_pos(returns_train, t), _slice_pos(returns_test, None, t)]
        )

        rlwgarch_result = minimize(
            lw_rgarch_lg_nll,
            current_estimates,
            args=(rolling_returns, rolling_xt, num_dimen),
            method="SLSQP",
            bounds=rlwgarch_bounds,
            constraints=rlwgarch_constraints,
        )

        current_estimates = rlwgarch_result.x
        omega_new, beta_new, gamma_new, xi_new, phi_new, tau1_new, tau2_new, sigma_mu_new = (
            rlwgarch_result.x[num_dimen:]
        )
        weights_new = rlwgarch_result.x[:num_dimen]

        xt_new = np.dot(rolling_xt[-1, :], weights_new)
        con_var_new = condi_var_lw_rgarch_lg(
            current_estimates, rolling_returns, rolling_xt, num_dimen
        )

        forecast_var_rg_lg[t] = np.exp(
            omega_new
            + beta_new * np.log(con_var_new[-1])
            + gamma_new * np.log(xt_new)
        )
        eachstep_forecast_nll[t] = pred_nll(
            _pos_value(returns_test, t), forecast_var_rg_lg[t]
        )

        omega_list[t] = omega_new
        beta_list[t] = beta_new
        gamma_list[t] = gamma_new
        xi_list[t] = xi_new
        phi_list[t] = phi_new
        tau1_list[t] = tau1_new
        tau2_list[t] = tau2_new
        sigma_mu_list[t] = sigma_mu_new
        syn_xt_list[n_train + t - 1] = xt_new
        weights_frame[t, :] = weights_new

    return (
        forecast_var_rg_lg,
        eachstep_forecast_nll,
        omega_list,
        beta_list,
        gamma_list,
        xi_list,
        phi_list,
        tau1_list,
        tau2_list,
        sigma_mu_list,
        syn_xt_list,
        weights_frame,
    )


def lw_rgarch_lg_fit_pred_process(
    market_name, measures_train, measures_test, returns_train, returns_test
):
    num_dimen = measures_train.shape[1]

    initial_weights = [1 / num_dimen] * num_dimen
    initial_params = [0.2, 0.6, 0.2, 0.1, 0.5, 0.1, 0.1, 0.5]
    initial_guess = initial_weights + initial_params

    if market_name == "KSE":
        initial_guess = [1 / num_dimen] * num_dimen + [
            0.02,
            0.15,
            0.7,
            0.1,
            0.1,
            0.1,
            0.1,
            1.0,
        ]

    rlwgarch_bounds = [(0, 1)] * num_dimen + [
        (0, 5),
        (0, 5),
        (0, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (0, 10),
    ]
    rlwgarch_constraints = [
        {"type": "ineq", "fun": para_constraint, "args": (num_dimen,)},
        {"type": "eq", "fun": weight_constraint, "args": (num_dimen,)},
    ]

    rlwgarch_result = minimize(
        lw_rgarch_lg_nll,
        initial_guess,
        args=(returns_train, measures_train, num_dimen),
        method="SLSQP",
        bounds=rlwgarch_bounds,
        constraints=rlwgarch_constraints,
    )

    optimal_weights = rlwgarch_result.x[:num_dimen]
    optimal_params = rlwgarch_result.x[num_dimen:]
    con_var = condi_var_lw_rgarch_lg(
        rlwgarch_result.x, returns_train, measures_train, num_dimen
    )
    rlwgarch_fit_nll = pred_nll(returns_train, con_var)

    (
        forecast_var,
        eachstep_forecast_nll,
        omega_list,
        beta_list,
        gamma_list,
        xi_list,
        phi_list,
        tau1_list,
        tau2_list,
        sigma_mu_list,
        syn_xt_list,
        weights_frame,
    ) = lw_rgarch_lg_rolling_forecast(
        returns_train,
        returns_test,
        measures_train,
        measures_test,
        con_var,
        rlwgarch_result.x,
        num_dimen,
    )

    overall_forecast_nll = pred_nll(returns_test, forecast_var)

    params_df = pd.DataFrame(
        {
            "Omega": omega_list,
            "Beta": beta_list,
            "Gamma": gamma_list,
            "Xi": xi_list,
            "Phi": phi_list,
            "Tau1": tau1_list,
            "Tau2": tau2_list,
            "Sigma_mu": sigma_mu_list,
        }
    )
    params_df.index = returns_test.index

    return (
        market_name,
        optimal_weights,
        optimal_params,
        rlwgarch_fit_nll,
        overall_forecast_nll,
        eachstep_forecast_nll,
        forecast_var,
        con_var,
        params_df,
        syn_xt_list,
        weights_frame,
    )


def beta_weights(paras, num_lags, normalise=True, include_gamma=False):
    a, b = paras
    j = np.arange(1, num_lags + 1)
    x = j / num_lags
    weights_kernel = (x ** (a - 1)) * ((1 - x) ** (b - 1))

    if normalise:
        weights_to_normalise = weights_kernel[:-1]
        weights = weights_to_normalise / np.sum(weights_to_normalise)
    elif include_gamma:
        weights = weights_kernel * gamma(a + b) / (gamma(a) * gamma(b))
    else:
        weights = weights_kernel

    return weights


def nlw_rgarch_lg_nll(estimates, returns, real_var, num_dimen):
    weights = beta_weights(estimates[:2], num_dimen + 1)
    params = estimates[2:]
    omega, beta, gamma_param, xi, phi, tau1, tau2, sigma_mu = params

    rt = np.asarray(returns)
    xt = np.dot(real_var, weights)
    n = len(rt)

    ht = np.zeros(n)
    zt = np.zeros(n)
    mut = np.zeros(n)

    ht[0] = np.sum(rt**2) / n
    mut[0] = np.average(xt)
    zt[0] = rt[0] / np.sqrt(ht[0])

    for i in range(1, n):
        zt[i] = rt[i] / np.sqrt(ht[i - 1])
        ht[i] = np.exp(
            omega + beta * np.log(ht[i - 1]) + gamma_param * np.log(xt[i - 1])
        )
        mut[i] = (
            np.log(xt[i])
            - xi
            - phi * np.log(ht[i])
            - tau1 * zt[i]
            - tau2 * (zt[i] ** 2 - 1)
        )

    ll_garch = -np.sum(np.log(ht) + (rt**2 / ht))
    ll_measure = -np.sum(np.log(sigma_mu**2) + (mut**2 / sigma_mu**2))
    return -(ll_garch + ll_measure)


def rnlw_garch_constraint(params, epsilon=1e-6):
    beta, gamma_param, phi = params[3], params[4], params[6]
    beta_gamma_phi = beta + gamma_param * phi
    c1 = beta_gamma_phi - epsilon
    c2 = 1 - beta_gamma_phi - epsilon
    return np.array([c1, c2])


rnlw_garch_constraints = [{"type": "ineq", "fun": rnlw_garch_constraint}]
rnlw_garch_bounds = [(1e-20, 1e10), (1e-20, 1e10)] + [
    (0, 5),
    (0, 5),
    (0, 5),
    (-5, 5),
    (-5, 5),
    (-5, 5),
    (-5, 5),
    (0, 10),
]


def condi_var_nlw_rgarch_lg(estimates, rt_train, xt_train, num_dimen):
    weights = beta_weights(estimates[:2], num_dimen + 1)
    params = estimates[2:]
    omega, beta, gamma_param, xi, phi, tau1, tau2, sigma_mu = params

    rt = np.asarray(rt_train)
    xt = np.dot(xt_train, weights)
    n = len(rt)

    ht = np.zeros(n)
    zt = np.zeros(n)
    mut = np.zeros(n)

    ht[0] = np.sum(rt**2) / n
    mut[0] = np.sum(xt) / n
    zt[0] = rt[0] / np.sqrt(ht[0])

    for i in range(1, n):
        ht[i] = np.exp(
            omega + beta * np.log(ht[i - 1]) + gamma_param * np.log(xt[i - 1])
        )
        zt[i] = rt[i] / np.sqrt(ht[i])
        mut[i] = (
            np.log(xt[i])
            - xi
            - phi * np.log(ht[i])
            - tau1 * zt[i]
            - tau2 * (zt[i] ** 2 - 1)
        )
    return ht


def nlw_rgarch_lg_rolling_forecast(
    returns_train,
    returns_test,
    rv_train,
    rv_test,
    conditional_var,
    estimates,
    num_dimen,
):
    weights = beta_weights(estimates[:2], num_dimen + 1)
    params = estimates[2:]
    omega, beta, gamma_param, xi, phi, tau1, tau2, sigma_mu = params

    n = len(returns_test)
    n_train = len(returns_train)

    forecast_var_rg_lg = np.zeros(n)
    eachstep_forecast_nll = np.zeros(n)
    omega_list = np.zeros(n)
    beta_list = np.zeros(n)
    gamma_list = np.zeros(n)
    xi_list = np.zeros(n)
    phi_list = np.zeros(n)
    tau1_list = np.zeros(n)
    tau2_list = np.zeros(n)
    sigma_mu_list = np.zeros(n)
    syn_xt_list = np.zeros(n_train + n)
    ab_frame = np.zeros((n, 2))
    weights_frame = np.zeros((n, num_dimen))

    xt_train = np.dot(rv_train, weights)
    initial_forecast_var = np.exp(
        omega + beta * np.log(conditional_var[-1]) + gamma_param * np.log(xt_train[-1])
    )
    initial_forecast_nll = pred_nll(_pos_value(returns_test, 0), initial_forecast_var)

    forecast_var_rg_lg[0] = initial_forecast_var
    eachstep_forecast_nll[0] = initial_forecast_nll
    omega_list[0] = omega
    beta_list[0] = beta
    gamma_list[0] = gamma_param
    xi_list[0] = xi
    phi_list[0] = phi
    tau1_list[0] = tau1
    tau2_list[0] = tau2
    sigma_mu_list[0] = sigma_mu

    syn_xt_list[:n_train] = xt_train
    ab_frame[0, :] = estimates[:2]
    weights_frame[0, :] = weights
    current_estimates = estimates.copy()

    for t in range(1, n):
        rolling_xt = np.vstack([rv_train.iloc[t:, :], rv_test.iloc[:t, :]])
        rolling_returns = pd.concat(
            [_slice_pos(returns_train, t), _slice_pos(returns_test, None, t)]
        )

        rnlw_garch_result = minimize(
            nlw_rgarch_lg_nll,
            current_estimates,
            args=(rolling_returns, rolling_xt, num_dimen),
            method="SLSQP",
            bounds=rnlw_garch_bounds,
            constraints=rnlw_garch_constraints,
        )

        current_estimates = rnlw_garch_result.x
        a_new, b_new = rnlw_garch_result.x[:2]
        omega_new, beta_new, gamma_new, xi_new, phi_new, tau1_new, tau2_new, sigma_mu_new = (
            rnlw_garch_result.x[2:]
        )
        weights_new = beta_weights((a_new, b_new), num_dimen + 1)
        con_var_new = condi_var_nlw_rgarch_lg(
            rnlw_garch_result.x, rolling_returns, rolling_xt, num_dimen
        )
        xt_new = np.dot(rolling_xt[-1, :], weights_new)

        forecast_var_rg_lg[t] = np.exp(
            omega_new
            + beta_new * np.log(con_var_new[-1])
            + gamma_new * np.log(xt_new)
        )
        eachstep_forecast_nll[t] = pred_nll(
            _pos_value(returns_test, t), forecast_var_rg_lg[t]
        )

        omega_list[t] = omega_new
        beta_list[t] = beta_new
        gamma_list[t] = gamma_new
        xi_list[t] = xi_new
        phi_list[t] = phi_new
        tau1_list[t] = tau1_new
        tau2_list[t] = tau2_new
        sigma_mu_list[t] = sigma_mu_new
        syn_xt_list[n_train + t - 1] = xt_new
        ab_frame[t, :] = [a_new, b_new]
        weights_frame[t, :] = weights_new

    return (
        forecast_var_rg_lg,
        eachstep_forecast_nll,
        omega_list,
        beta_list,
        gamma_list,
        xi_list,
        phi_list,
        tau1_list,
        tau2_list,
        sigma_mu_list,
        syn_xt_list,
        ab_frame,
        weights_frame,
    )


def nlw_rgarch_lg_fit_pred_process(
    market_name, measures_train, measures_test, returns_train, returns_test
):
    num_dimen = measures_train.shape[1]

    initial_ab = [1.5, 1.5]
    initial_params = [0.2, 0.6, 0.2, 0.1, 0.5, 0.1, 0.1, 0.5]
    initial_guess = initial_ab + initial_params

    rnlw_garch_result = minimize(
        nlw_rgarch_lg_nll,
        initial_guess,
        args=(returns_train, measures_train, num_dimen),
        method="SLSQP",
        bounds=rnlw_garch_bounds,
        constraints=rnlw_garch_constraints,
    )

    optimal_ab = rnlw_garch_result.x[:2]
    optimal_weights = beta_weights(optimal_ab, num_dimen + 1)
    optimal_params = rnlw_garch_result.x[2:]
    con_var = condi_var_nlw_rgarch_lg(
        rnlw_garch_result.x, returns_train, measures_train, num_dimen
    )
    rnlw_garch_fit_nll = pred_nll(returns_train, con_var)

    (
        forecast_var,
        eachstep_forecast_nll,
        omega_list,
        beta_list,
        gamma_list,
        xi_list,
        phi_list,
        tau1_list,
        tau2_list,
        sigma_mu_list,
        syn_xt_list,
        ab_frame,
        weights_frame,
    ) = nlw_rgarch_lg_rolling_forecast(
        returns_train,
        returns_test,
        measures_train,
        measures_test,
        con_var,
        rnlw_garch_result.x,
        num_dimen,
    )

    overall_forecast_nll = pred_nll(returns_test, forecast_var)

    params_df = pd.DataFrame(
        {
            "Omega": omega_list,
            "Beta": beta_list,
            "Gamma": gamma_list,
            "Xi": xi_list,
            "Phi": phi_list,
            "Tau1": tau1_list,
            "Tau2": tau2_list,
            "Sigma_mu": sigma_mu_list,
        }
    )
    params_df.index = returns_test.index

    return (
        market_name,
        optimal_ab,
        optimal_weights,
        optimal_params,
        rnlw_garch_fit_nll,
        overall_forecast_nll,
        eachstep_forecast_nll,
        forecast_var,
        con_var,
        params_df,
        syn_xt_list,
        ab_frame,
        weights_frame,
    )


def rgarch_lg_dr_rolling_forecast(
    dimension_reduced_dataframe, returns_insample, returns_outsample, params
):
    rgarch_bounds = [
        (0, 5),
        (0, 5),
        (0, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (0, 10),
    ]
    rgarch_constraints = [{"type": "ineq", "fun": rgarch_params_constraints}]

    n_fore = len(returns_outsample)
    forecast_ht_rg_lg = np.zeros(n_fore)
    eachstep_forecast_nll = np.zeros(n_fore)
    omega_list = np.zeros(n_fore)
    beta_list = np.zeros(n_fore)
    gamma_list = np.zeros(n_fore)
    xi_list = np.zeros(n_fore)
    phi_list = np.zeros(n_fore)
    tau1_list = np.zeros(n_fore)
    tau2_list = np.zeros(n_fore)
    sigma_mu_list = np.zeros(n_fore)

    current_params = params.copy()
    for t in range(n_fore):
        rolling_rv = dimension_reduced_dataframe.iloc[:, t]
        rolling_returns = pd.concat(
            [_slice_pos(returns_insample, t), _slice_pos(returns_outsample, None, t)]
        )

        result = minimize(
            rgarch_log_nll,
            current_params,
            args=(rolling_returns, rolling_rv),
            method="SLSQP",
            bounds=rgarch_bounds,
            constraints=rgarch_constraints,
        )

        current_params = result.x
        omega_new, beta_new, gamma_new, xi_new, phi_new, tau1_new, tau2_new, sigma_mu_new = (
            current_params
        )
        condi_var_new = condi_var_rgarch_lg(rolling_returns, rolling_rv, current_params)

        if t == 0:
            fit_nll = pred_nll(returns_insample, condi_var_new)
            insample_params = current_params

        forecast_ht_rg_lg[t] = np.exp(
            omega_new
            + beta_new * np.log(condi_var_new[-1])
            + gamma_new * np.log(rolling_rv.iloc[-1])
        )

        eachstep_forecast_nll[t] = pred_nll(
            _pos_value(returns_outsample, t), forecast_ht_rg_lg[t]
        )
        omega_list[t] = omega_new
        beta_list[t] = beta_new
        gamma_list[t] = gamma_new
        xi_list[t] = xi_new
        phi_list[t] = phi_new
        tau1_list[t] = tau1_new
        tau2_list[t] = tau2_new
        sigma_mu_list[t] = sigma_mu_new

    return (
        insample_params,
        fit_nll,
        forecast_ht_rg_lg,
        eachstep_forecast_nll,
        omega_list,
        beta_list,
        gamma_list,
        xi_list,
        phi_list,
        tau1_list,
        tau2_list,
        sigma_mu_list,
    )


def rgarch_lg_dr_fit_pred_process(
    name, dimension_reduced_dataframe, returns_insample, returns_outsample
):
    initial_params_rgarch = [0.2, 0.6, 0.2, 0.1, 0.5, 0.1, 0.1, 0.5]

    (
        insample_params,
        fit_nll,
        forecast_var,
        eachstep_forecast_nll,
        omega_list,
        beta_list,
        gamma_list,
        xi_list,
        phi_list,
        tau1_list,
        tau2_list,
        sigma_mu_list,
    ) = rgarch_lg_dr_rolling_forecast(
        dimension_reduced_dataframe,
        returns_insample,
        returns_outsample,
        initial_params_rgarch,
    )

    overall_forecast_nll = pred_nll(returns_outsample, forecast_var)

    params_df = pd.DataFrame(
        {
            "Omega": omega_list,
            "Beta": beta_list,
            "Gamma": gamma_list,
            "Xi": xi_list,
            "Phi": phi_list,
            "Tau1": tau1_list,
            "Tau2": tau2_list,
            "Sigma_mu": sigma_mu_list,
        }
    )
    params_df.index = returns_outsample.index

    return (
        name,
        insample_params,
        fit_nll,
        overall_forecast_nll,
        eachstep_forecast_nll,
        forecast_var,
        params_df,
    )


def rolling_pca_ica(market, rms_insample, rms_outsample):
    n_insample = rms_insample.shape[0]
    n_outsample = rms_outsample.shape[0]
    pca_series_matrix = np.zeros((n_insample, n_outsample + 1))
    ica_series_matrix = np.zeros((n_insample, n_outsample + 1))

    for t in range(n_outsample + 1):
        rms_df_np = np.vstack([rms_insample.iloc[t:, :], rms_outsample.iloc[:t, :]])

        pca = decomposition.PCA(n_components=1)
        pca.fit(rms_df_np)
        weights_pc = pca.components_
        pc_series = rms_df_np @ weights_pc.T
        pc_series_scaled = (
            (pc_series - pc_series.min())
            / (pc_series.max() - pc_series.min())
            * (rms_df_np.max() - rms_df_np.min())
            + rms_df_np.min()
        )
        pca_series_matrix[:, t] = pc_series_scaled.flatten()
        pca_series_df = pd.DataFrame(pca_series_matrix)

        ica = decomposition.FastICA(n_components=1)
        ica.fit(rms_df_np)
        weights_ic = np.abs(ica.components_)
        ic_series = rms_df_np @ weights_ic.T
        ic_series_scaled = (
            (ic_series - ic_series.min())
            / (ic_series.max() - ic_series.min())
            * (rms_df_np.max() - rms_df_np.min())
            + rms_df_np.min()
        )
        ica_series_matrix[:, t] = ic_series_scaled.flatten()
        ica_series_df = pd.DataFrame(ica_series_matrix)

    return market, pca_series_df, ica_series_df
