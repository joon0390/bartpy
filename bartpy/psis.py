import numpy as np
from typing import Tuple, Optional, Dict

try:
    from scipy.stats import genpareto as _gpd
    from scipy.special import logsumexp as _logsumexp
except Exception:  # scipy may not be present at edit time, but is in requirements
    _gpd = None
    _logsumexp = None


def _ensure_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 1:
        return a.reshape(-1)
    return a


def psis_smooth_weights(
    log_lik_i: np.ndarray,
    tail_fraction: float = 0.2,
    min_tail: int = 20,
    ionides_trunc: bool = True,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """
    Compute PSIS-smoothed importance weights for a single observation i
    from per-draw log-likelihoods log p(y_i | theta_s).

    Returns normalized weights w_s (sum to 1), Pareto k estimate, and diagnostics.

    Parameters
    - log_lik_i: shape (S,), log p(y_i | theta_s)
    - tail_fraction: fraction of largest raw weights to smooth (default 0.2)
    - min_tail: minimum number of tail points (default 20)
    - ionides_trunc: apply Ionides S^{3/4} truncation
    - eps: floor to avoid zeros

    Notes
    - Requires SciPy for GPD fitting; else falls back to no-smoothing (standard SNIS),
      which may be unstable for heavy tails.
    """

    ll = _ensure_1d(np.asarray(log_lik_i, dtype=float))
    S = ll.shape[0]
    if S == 0:
        raise ValueError("log_lik_i must be non-empty")

    # Importance ratios proportional to 1 / p(y_i | theta_s)
    # Compute in log space for stability: log r_s = -log p(y_i|theta_s)
    log_r = -ll
    log_r -= np.max(log_r)  # stabilize
    r = np.exp(log_r)
    r = np.maximum(r, eps)

    # Determine tail size
    m = max(min_tail, int(np.floor(tail_fraction * S)))
    m = min(m, S)  # cannot exceed S
    if m == 0:
        w = r / np.sum(r)
        ess = 1.0 / np.sum(w ** 2)
        return w, float('nan'), {"ess": float(ess), "tail_m": float(m)}

    # Sort weights and extract tail
    order = np.argsort(r)
    r_sorted = r[order]
    tail = r_sorted[-m:]
    u = r_sorted[-m]  # threshold (minimum of tail)

    k_hat = np.nan
    if _gpd is not None and m >= 5 and np.any(tail > u):
        # exceedances
        y = tail - u
        try:
            # Fit GPD with loc fixed at 0
            c, loc, scale = _gpd.fit(y, floc=0.0)
            k_hat = float(c)

            # Expected order statistics via GPD quantiles
            p = (np.arange(1, m + 1) - 0.5) / m  # in (0,1)
            y_smooth = _gpd.ppf(p, c, loc=0.0, scale=scale)
            y_smooth = np.clip(y_smooth, 0.0, np.inf)
            tail_smooth = u + y_smooth

            # Replace tail portion with smoothed values
            r_sorted[-m:] = tail_smooth
        except Exception:
            # If fitting fails, proceed without smoothing
            pass

    # Ionides truncation to control variance
    if ionides_trunc:
        tau = (S ** 0.75) * np.mean(r_sorted)
        r_sorted = np.minimum(r_sorted, tau)

    # Unsort back to original draw order
    r_smoothed = np.empty_like(r_sorted)
    r_smoothed[order] = r_sorted

    # Normalize to self-normalized IS weights
    total = np.sum(r_smoothed)
    total = max(total, eps)
    w = r_smoothed / total

    ess = 1.0 / np.sum(w ** 2)
    return w, k_hat, {"ess": float(ess), "tail_m": float(m)}


def psis_loo_lpd(
    log_lik_matrix: np.ndarray,
    tail_fraction: float = 0.2,
    min_tail: int = 20,
    ionides_trunc: bool = True,
    return_weights: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute PSIS-LOO log predictive densities and diagnostics.

    Parameters
    - log_lik_matrix: shape (n, S), per-observation per-draw log-likelihoods
    - return_weights: if True, include per-point weight arrays (object dtype)

    Returns dict with keys:
    - lpd_loo: shape (n,), log p(y_i | y_-i) approximations
    - pareto_k: shape (n,), Pareto k diagnostics
    - ess: shape (n,), effective sample size per point
    - weights: optional, list of arrays of shape (S,) per i
    - elpd_loo: scalar sum of lpd_loo
    """

    if _logsumexp is None:
        # Minimal fallback for logsumexp to avoid SciPy hard dependency here
        def _lse(a):
            a = np.asarray(a)
            m = np.max(a)
            return m + np.log(np.sum(np.exp(a - m)))
    else:
        def _lse(a):
            return float(_logsumexp(a))

    ll = np.asarray(log_lik_matrix, dtype=float)
    if ll.ndim != 2:
        raise ValueError("log_lik_matrix must have shape (n, S)")
    n, S = ll.shape

    lpd_loo = np.empty(n, dtype=float)
    pareto_k = np.empty(n, dtype=float)
    ess = np.empty(n, dtype=float)
    weights_out = [] if return_weights else None

    for i in range(n):
        w_i, k_i, diag = psis_smooth_weights(
            ll[i], tail_fraction=tail_fraction, min_tail=min_tail, ionides_trunc=ionides_trunc
        )
        # log p_loo(y_i) = log sum_s w_i[s] * p(y_i | theta_s)
        a = ll[i] + np.log(np.maximum(w_i, 1e-300))
        lpd_loo[i] = _lse(a)
        pareto_k[i] = k_i
        ess[i] = diag.get("ess", np.nan)
        if return_weights:
            weights_out.append(w_i)

    out: Dict[str, np.ndarray] = {
        "lpd_loo": lpd_loo,
        "pareto_k": pareto_k,
        "ess": ess,
        "elpd_loo": np.sum(lpd_loo),
    }
    if return_weights:
        out["weights"] = np.array(weights_out, dtype=object)
    return out

