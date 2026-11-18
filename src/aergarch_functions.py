"""Core functions extracted from ``AERGARCH.ipynb``.

This module keeps the original notebook function names so the notebook can
import them directly while the project is gradually cleaned for GitHub.
"""

import numbers
import random

try:
    import torch
except ModuleNotFoundError:
    torch = None

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
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
    "FW_PARAMETER_NAMES",
    "beta_component_weights",
    "beta_mixture_weights",
    "fw_nll",
    "conditional_variance_fw",
    "fit_fw",
    "fit_predict_fw",
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


def _rgarch_inputs(returns, real_var):
    rt = np.asarray(returns, dtype=float)
    xt = np.asarray(real_var, dtype=float)
    if rt.ndim != 1 or rt.shape != xt.shape or not len(rt):
        raise ValueError("Returns and realized variance must be nonempty, equal-length vectors.")
    return rt, xt


def _rgarch_variance(params, returns, log_measure):
    omega, beta, gamma_param = params[:3]
    initial_variance = np.sum(returns**2) / len(returns)
    log_variance = np.empty(len(returns))
    log_variance[0] = np.log(initial_variance)
    if len(returns) > 1:
        log_variance[1:], _ = lfilter(
            [1.0], [1.0, -beta], omega + gamma_param * log_measure[:-1],
            zi=[beta * log_variance[0]],
        )
    variance = np.exp(log_variance)
    variance[0] = initial_variance
    return variance


def rgarch_log_nll(params, returns, real_var):
    _, _, _, xi, phi, tau1, tau2, sigma_mu = params
    rt, xt = _rgarch_inputs(returns, real_var)
    if (not np.isfinite(params).all() or not np.isfinite(rt).all()
            or not np.isfinite(xt).all() or np.any(xt <= 0) or sigma_mu == 0):
        return float("inf")

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        log_measure = np.log(xt)
        variance = _rgarch_variance(params, rt, log_measure)
        if not np.isfinite(variance).all() or np.any(variance <= 0):
            return float("inf")
        log_variance = np.log(variance)
        standardized_return = rt / np.sqrt(variance)
        residual = (
            log_measure - xi - phi * log_variance
            - tau1 * standardized_return - tau2 * (standardized_return**2 - 1)
        )
        residual[0] = np.sum(xt) / len(xt)
        sigma_squared = sigma_mu**2
        return_nll = np.sum(log_variance + rt**2 / variance)
        measure_nll = np.sum(np.log(sigma_squared) + residual**2 / sigma_squared)
        objective = return_nll + measure_nll
    return float(objective) if np.isfinite(objective) else float("inf")


def condi_var_rgarch_lg(rt_train, xt_train, params):
    rt, xt = _rgarch_inputs(rt_train, xt_train)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        return _rgarch_variance(params, rt, np.log(xt))


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
    synthetic_measure = np.dot(xt_train, estimates[:num_dimen])
    return rgarch_log_nll(estimates[num_dimen:], rt_train, synthetic_measure)


def condi_var_lw_rgarch_lg(estimates, rt_train, xt_train, num_dimen):
    synthetic_measure = np.dot(xt_train, estimates[:num_dimen])
    return condi_var_rgarch_lg(rt_train, synthetic_measure, estimates[num_dimen:])


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


FW_PARAMETER_NAMES = (
    "mix", "a1", "b1", "a2", "b2", "omega", "beta", "gamma",
    "xi", "phi", "tau1", "tau2", "sigma_mu",
)
_FW_INITIAL_ESTIMATES = (
    0.5, 1.5, 3.0, 3.0, 1.5, 0.2, 0.6, 0.2, 0.1, 0.5, 0.1, 0.1, 0.5,
)
_FW_BOUNDS = (
    ((0.0, 1.0),) + ((1e-20, 1e10),) * 4
    + ((0.0, 5.0),) * 3 + ((-5.0, 5.0),) * 4 + ((1e-8, 10.0),)
)
_FW_NUMERICAL_PENALTY = 1e100


def _fw_vector(values, length, name):
    values = np.asarray(values, dtype=float)
    if values.shape != (length,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain {length} finite values.")
    return values


def beta_component_weights(shape_params, num_measures):
    """Return normalized beta-kernel weights on d/(D+1), d=1,...,D."""
    if (isinstance(num_measures, bool) or not isinstance(num_measures, numbers.Integral)
            or num_measures < 1):
        raise ValueError("num_measures must be a positive integer.")
    a, b = _fw_vector(shape_params, 2, "shape_params")
    if a <= 0 or b <= 0:
        raise ValueError("Both beta shape parameters must be positive.")
    grid = np.arange(1, num_measures + 1, dtype=float) / (num_measures + 1)
    log_kernel = (a - 1) * np.log(grid) + (b - 1) * np.log1p(-grid)
    # Subtracting the maximum avoids underflow of all kernels for large shapes.
    kernels = np.exp(log_kernel - np.max(log_kernel))
    return kernels / kernels.sum()


def beta_mixture_weights(mixture_params, num_measures):
    """Return mix * beta(a1,b1) + (1-mix) * beta(a2,b2), each normalized."""
    mix, a1, b1, a2, b2 = _fw_vector(mixture_params, 5, "mixture_params")
    if not 0 <= mix <= 1:
        raise ValueError("mix must lie in [0, 1].")
    first = beta_component_weights((a1, b1), num_measures)
    second = beta_component_weights((a2, b2), num_measures)
    return mix * first + (1 - mix) * second


def _fw_estimates(estimates):
    estimates = _fw_vector(estimates, len(FW_PARAMETER_NAMES), "estimates")
    if not 0 <= estimates[0] <= 1 or np.any(estimates[1:5] <= 0):
        raise ValueError("Use mix in [0, 1] and four positive beta shape parameters.")
    if estimates[-1] <= 0:
        raise ValueError("sigma_mu must be positive.")
    return estimates


def _fw_persistence_constraint(estimates, epsilon=1e-6):
    """SLSQP inequalities enforcing epsilon <= beta+gamma*phi <= 1-epsilon."""
    estimates = _fw_vector(estimates, len(FW_PARAMETER_NAMES), "estimates")
    persistence = estimates[6] + estimates[7] * estimates[9]
    return np.array([persistence - epsilon, 1 - persistence - epsilon])


def _fw_training_data(returns, real_var):
    if isinstance(returns, pd.Series) and isinstance(real_var, pd.DataFrame):
        if not returns.index.equals(real_var.index):
            raise ValueError("Return and realized-measure dates must match exactly.")
    rt, rm = np.asarray(returns, dtype=float), np.asarray(real_var, dtype=float)
    if rt.ndim != 1 or len(rt) < 2:
        raise ValueError("Training returns must be a vector with at least two observations.")
    if rm.ndim != 2 or rm.shape[0] != len(rt) or rm.shape[1] < 1:
        raise ValueError("real_var must have one row per return and at least one column.")
    if not np.isfinite(rt).all() or not np.isfinite(rm).all() or np.any(rm <= 0):
        raise ValueError("Returns must be finite and realized measures finite and positive.")
    with np.errstate(over="ignore", under="ignore"):
        initial_variance = np.mean(rt**2)
    if not np.isfinite(initial_variance) or initial_variance <= 0:
        raise ValueError("The initial mean squared return must be finite and positive.")
    return rt, rm


def _fw_filter(estimates, rt, rm):
    weights = beta_mixture_weights(estimates[:5], rm.shape[1])
    synthetic = rm @ weights
    omega, beta, gamma = estimates[5:8]
    log_h = np.empty(len(rt))
    log_h[0] = np.log(np.mean(rt**2))
    log_h[1:], _ = lfilter(
        [1.0], [1.0, -beta], omega + gamma * np.log(synthetic[:-1]),
        zi=[beta * log_h[0]],
    )
    h = np.exp(log_h)
    if not np.isfinite(h).all() or np.any(h <= 0):
        raise FloatingPointError("Conditional variance became nonfinite or nonpositive.")
    return weights, synthetic, h


def conditional_variance_fw(estimates, returns, real_var):
    """Filter the conditional variance with mixture weights."""
    estimates = _fw_estimates(estimates)
    rt, rm = _fw_training_data(returns, real_var)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        return _fw_filter(estimates, rt, rm)[2]


def _fw_nll(estimates, rt, rm):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        try:
            _, synthetic, h = _fw_filter(estimates, rt, rm)
            xi, phi, tau1, tau2, sigma_mu = estimates[8:]
            residual = np.empty(len(rt))
            residual[0] = synthetic.mean()
            z = rt[1:] / np.sqrt(h[:-1])
            residual[1:] = (
                np.log(synthetic[1:]) - xi - phi * np.log(h[1:])
                - tau1 * z - tau2 * (z**2 - 1)
            )
            objective = np.sum(np.log(h) + rt**2 / h) + np.sum(
                np.log(sigma_mu**2) + residual**2 / sigma_mu**2
            )
            return float(objective) if np.isfinite(objective) else _FW_NUMERICAL_PENALTY
        except FloatingPointError:
            return _FW_NUMERICAL_PENALTY


def fw_nll(estimates, returns, real_var):
    """Evaluate the joint negative log-likelihood."""
    estimates = _fw_estimates(estimates)
    return _fw_nll(estimates, *_fw_training_data(returns, real_var))


def fit_fw(returns, real_var, *, initial_estimates=None, optimizer_options=None):
    """Jointly estimate five mixture and eight RealGARCH parameters using SLSQP.

    Raises RuntimeError if optimization fails or yields an invalid solution.
    The returned OptimizeResult additionally holds weights, conditional_variance,
    synthetic_measure, predictive_nll, persistence, and constraint_values.
    """
    rt, rm = _fw_training_data(returns, real_var)
    initial = _fw_estimates(_FW_INITIAL_ESTIMATES if initial_estimates is None else initial_estimates).copy()
    bounds = np.asarray(_FW_BOUNDS)
    if np.any(initial < bounds[:, 0]) or np.any(initial > bounds[:, 1]):
        raise ValueError("Initial estimates must satisfy the FW-RealGARCH parameter bounds.")
    options = {"maxiter": 1000, "ftol": 1e-7}
    options.update(optimizer_options or {})
    result = minimize(
        _fw_nll, initial, args=(rt, rm), method="SLSQP", bounds=_FW_BOUNDS,
        constraints=[{"type": "ineq", "fun": _fw_persistence_constraint}], options=options,
    )
    feasible = (
        np.isfinite(result.x).all() and np.isfinite(result.fun)
        and result.fun < _FW_NUMERICAL_PENALTY
        and np.all(result.x >= bounds[:, 0] - 1e-7)
        and np.all(result.x <= bounds[:, 1] + 1e-7)
        and np.all(_fw_persistence_constraint(result.x) >= -1e-7)
    )
    if not result.success or not feasible:
        raise RuntimeError(
            f"FW-RealGARCH optimization failed (status={result.status}, "
            f"objective={result.fun}, feasible={feasible}): {result.message}"
        )
    # SLSQP can finish a few ulps outside a bound. Project only an already
    # accepted solution, so it is also a valid starting point for the next window.
    bounded_estimates = np.clip(result.x, bounds[:, 0], bounds[:, 1])
    result.bounds_adjusted = bool(np.any(bounded_estimates != result.x))
    result.x = bounded_estimates
    result.fun = _fw_nll(result.x, rt, rm)
    if result.fun >= _FW_NUMERICAL_PENALTY or np.any(_fw_persistence_constraint(result.x) < -1e-7):
        raise RuntimeError("The fitted parameters are invalid after removing bound roundoff.")
    result.weights, result.synthetic_measure, result.conditional_variance = _fw_filter(result.x, rt, rm)
    result.predictive_nll = float(np.sum(np.log(result.conditional_variance) + rt**2 / result.conditional_variance))
    result.persistence = float(result.x[6] + result.x[7] * result.x[9])
    result.constraint_values = _fw_persistence_constraint(result.x)
    return result


def _fw_dated_pair(returns, measures, label):
    if not isinstance(returns, pd.Series) or not isinstance(measures, pd.DataFrame):
        raise TypeError(f"{label} inputs must be a pandas Series and DataFrame.")
    if not returns.index.equals(measures.index):
        raise ValueError(f"{label} return and measure dates must match exactly.")
    if not returns.index.is_unique or not returns.index.is_monotonic_increasing:
        raise ValueError(f"{label} dates must be unique and increasing.")
    if measures.shape[1] < 1 or not measures.columns.is_unique:
        raise ValueError(f"{label} realized-measure columns must be nonempty and unique.")
    rt, rm = returns.to_numpy(dtype=float), measures.to_numpy(dtype=float)
    if not np.isfinite(rt).all() or not np.isfinite(rm).all() or np.any(rm <= 0):
        raise ValueError(f"{label} data must be finite, with positive realized measures.")
    return rt, rm


def fit_predict_fw(
    market_name, measures_train, measures_test, returns_train, returns_test, *,
    n_forecasts=None, initial_estimates=None, optimizer_options=None,
):
    """Fit once, then refit fixed-length windows for one-step-ahead forecasts.

    At step t, the fitting window ends immediately before test observation t.
    n_forecasts=None uses every test observation; zero returns only the initial
    fit. Forecast rows contain all 13 estimates and the combined RM weights.
    """
    rt_train, rm_train = _fw_dated_pair(returns_train, measures_train, "Training")
    rt_test, rm_test = _fw_dated_pair(returns_test, measures_test, "Test")
    if not measures_train.columns.equals(measures_test.columns):
        raise ValueError("Training and test realized-measure columns must match in order.")
    if len(rt_train) < 2:
        raise ValueError("At least two training observations are required.")
    if len(rt_test) and returns_train.index[-1] >= returns_test.index[0]:
        raise ValueError("Test dates must follow all training dates without overlap.")
    count = len(rt_test) if n_forecasts is None else n_forecasts
    if (isinstance(count, bool) or not isinstance(count, numbers.Integral)
            or count < 0 or count > len(rt_test)):
        raise ValueError("n_forecasts must be an integer between zero and the test length.")
    initial_fit = fit_fw(
        returns_train, measures_train, initial_estimates=initial_estimates,
        optimizer_options=optimizer_options,
    )
    current_fit = initial_fit
    all_returns = np.concatenate([rt_train, rt_test[:count]])
    all_measures = np.vstack([rm_train, rm_test[:count]])
    columns = ["forecast_var", "eachstep_nll", "synthetic_rm_lag", *FW_PARAMETER_NAMES]
    columns += [f"weight_{name}" for name in measures_train.columns]
    forecast = pd.DataFrame(index=returns_test.index[:count], columns=columns, dtype=float)
    diagnostics = []

    def record_fit(fit, step, stage):
        diagnostics.append({
            "step": step, "stage": stage, "success": bool(fit.success),
            "status": int(fit.status), "message": str(fit.message), "iterations": int(fit.nit),
            "joint_nll": float(fit.fun), "persistence": float(fit.x[6] + fit.x[7] * fit.x[9]),
        })

    record_fit(initial_fit, 0, "initial")
    for t in range(count):
        if t:
            current_fit = fit_fw(
                all_returns[t:t + len(rt_train)], all_measures[t:t + len(rt_train)],
                initial_estimates=current_fit.x, optimizer_options=optimizer_options,
            )
            record_fit(current_fit, t, "rolling")
        omega, beta, gamma = current_fit.x[5:8]
        synthetic_lag = float(current_fit.synthetic_measure[-1])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            predicted = float(np.exp(
                omega + beta * np.log(current_fit.conditional_variance[-1])
                + gamma * np.log(synthetic_lag)
            ))
            loss = float(np.log(predicted) + rt_test[t]**2 / predicted)
        if not np.isfinite(predicted) or predicted <= 0 or not np.isfinite(loss):
            raise FloatingPointError(f"Invalid forecast for {market_name}, step {t}.")
        forecast.iloc[t] = [predicted, loss, synthetic_lag, *current_fit.x, *current_fit.weights]
    return {
        "market": market_name, "fit": initial_fit, "forecast": forecast,
        "insample_conditional_variance": pd.Series(initial_fit.conditional_variance, index=returns_train.index, name="conditional_variance"),
        "insample_synthetic_measure": pd.Series(initial_fit.synthetic_measure, index=returns_train.index, name="synthetic_measure"),
        "insample_weights": pd.Series(initial_fit.weights, index=measures_train.columns, name="weight"),
        "fit_nll": float(initial_fit.predictive_nll),
        "forecast_nll": float(forecast["eachstep_nll"].sum()),
        "optimizer_diagnostics": pd.DataFrame(diagnostics),
    }


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
