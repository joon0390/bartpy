import numpy as np
from typing import Tuple, Dict, Optional

from .psis import psis_smooth_weights


def _ensure_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """
    Split/conformal quantile with finite-sample correction.

    Uses the canonical ceil((n+1)*(1-alpha))/n order statistic.
    """
    r = np.asarray(residuals, dtype=float).reshape(-1)
    n = r.shape[0]
    if n == 0:
        raise ValueError("residuals must be non-empty")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)  # clamp to [1, n]
    # zero-based index for np.partition
    idx = k - 1
    r_part = np.partition(r, idx)
    return float(r_part[idx])


def build_intervals(pred_point: np.ndarray, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build symmetric conformal intervals around point predictions using width q.
    """
    pred = np.asarray(pred_point, dtype=float)
    return pred - q, pred + q


def loo_residuals_via_psis(
    y: np.ndarray,
    posterior_draws: np.ndarray,
    log_lik: np.ndarray,
    tail_fraction: float = 0.2,
    min_tail: int = 20,
    ionides_trunc: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute absolute LOO residuals using PSIS weights.

    Parameters
    - y: shape (n,), observed targets
    - posterior_draws: shape (n, S), posterior predictive draws for each observation
      (e.g., draws of y_i given theta_s)
    - log_lik: shape (n, S), log p(y_i | theta_s)

    Returns
    - residuals: shape (n,), |y_i - E_LOO[y_i]|
    - pareto_k: shape (n,), PSIS k diagnostics
    - loo_pred: shape (n,), E_LOO[y_i]
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    draws = np.asarray(posterior_draws, dtype=float)
    ll = np.asarray(log_lik, dtype=float)

    # Accept either (n, S) or (S, n) for draws, and same for ll
    if ll.ndim != 2:
        raise ValueError("log_lik must be 2D with shape (n, S) or (S, n)")

    # Try to coerce to (n, S)
    if draws.shape == ll.shape:
        D = draws
        L = ll
    elif draws.T.shape == ll.shape:
        D = draws.T
        L = ll
    elif draws.shape == ll.T.shape:
        D = draws
        L = ll.T
    elif draws.T.shape == ll.T.shape:
        D = draws.T
        L = ll.T
    else:
        raise ValueError("posterior_draws and log_lik must be broadcastable to the same 2D shape")

    if y.shape[0] != D.shape[0]:
        raise ValueError("y and posterior_draws must agree on number of observations (n)")

    n, S = D.shape
    loo_pred = np.empty(n, dtype=float)
    k_diag = np.empty(n, dtype=float)

    for i in range(n):
        w_i, k_i, _ = psis_smooth_weights(
            L[i], tail_fraction=tail_fraction, min_tail=min_tail, ionides_trunc=ionides_trunc
        )
        loo_pred[i] = float(np.dot(w_i, D[i]))
        k_diag[i] = k_i

    residuals = np.abs(y - loo_pred)
    return residuals, k_diag, loo_pred


def split_conformal_quantile(
    y_cal: np.ndarray,
    posterior_draws_cal: np.ndarray,
    alpha: float,
    agg: str = "median",
) -> float:
    """
    Split-conformal residual quantile on a calibration set.

    Parameters
    - y_cal: shape (n_cal,)
    - posterior_draws_cal: shape (n_cal, S)
    - alpha: miscoverage level
    - agg: 'mean' or 'median' for point summary from draws
    """
    y_cal = np.asarray(y_cal, dtype=float).reshape(-1)
    D = np.asarray(posterior_draws_cal, dtype=float)
    if D.ndim != 2 or y_cal.shape[0] != D.shape[0]:
        raise ValueError("posterior_draws_cal must have shape (n_cal, S) and match y_cal")

    if agg == "mean":
        pred = D.mean(axis=1)
    elif agg == "median":
        pred = np.median(D, axis=1)
    else:
        raise ValueError("agg must be 'mean' or 'median'")

    res = np.abs(y_cal - pred)
    return conformal_quantile(res, alpha)


def _normal_logpdf(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, eps)
    var = sigma ** 2
    return -0.5 * (np.log(2.0 * np.pi * var) + ((y - mu) ** 2) / var)


def prepare_bart_loglik_and_draws(
    model,
    use_original_scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare (y, posterior_draws, log_lik) from a fitted bartpy.sklearnmodel.SklearnModel.

    Returns
    - y: shape (n,)
    - posterior_draws: shape (n, S) using per-sample fitted means f_{i,s}
    - log_lik: shape (n, S), Normal log p(y_i | f_{i,s}, sigma_s)

    Notes
    - Requires model to be fitted with store_in_sample_predictions=True so that
      per-sample in-sample predictions and model objects are stored.
    - The per-sample sigma is read from each stored Model in model._model_samples.
    """
    # Basic checks
    if not hasattr(model, "_prediction_samples") or model._prediction_samples is None:
        raise ValueError("Model does not have stored in-sample prediction samples. Fit with store_in_sample_predictions=True.")
    if not hasattr(model, "_model_samples") or model._model_samples is None:
        raise ValueError("Model does not have stored per-sample models. Enable model storing in sampler (default True).")

    # In-sample fitted means per sample: shape (S, n) in normalized scale
    pred_samples_norm = np.asarray(model._prediction_samples, dtype=float)
    if pred_samples_norm.ndim != 2:
        raise ValueError("_prediction_samples must be 2D (S, n)")
    S, n = pred_samples_norm.shape

    # Convert to original scale if requested
    if use_original_scale:
        pred_samples = model.data.y.unnormalize_y(pred_samples_norm)
        y_obs = model.data.y.unnormalized_y
    else:
        pred_samples = pred_samples_norm
        y_obs = model.data.y.values

    # Gather per-sample sigma (original scale if use_original_scale)
    sigma_s = []
    for m in model._model_samples:
        sig = m.sigma.current_unnormalized_value() if use_original_scale else m.sigma.current_value()
        sigma_s.append(sig)
    sigma_s = np.asarray(sigma_s, dtype=float)
    if sigma_s.shape[0] != S:
        raise ValueError("Number of sigma samples does not match number of prediction samples")

    # Build matrices with shape (n, S)
    posterior_draws = pred_samples.T  # (n, S)
    log_lik = _normal_logpdf(y_obs[:, None], posterior_draws, sigma_s[None, :])
    return y_obs, posterior_draws, log_lik
