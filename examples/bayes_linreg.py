import numpy as np
from typing import Tuple


def design_matrix(X: np.ndarray) -> np.ndarray:
    """
    Build a flexible basis for nonlinear regression using simple features.
    Includes: intercept, x1, x2, x1^2, x2^2, x1*x2, sin(x1), cos(x1).
    Accepts X with shape (n, d>=1). If d==1, x2 terms are dropped.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    x1 = X[:, 0]
    cols = [np.ones(n), x1, x1 ** 2, np.sin(x1), np.cos(x1)]
    if d >= 2:
        x2 = X[:, 1]
        cols.extend([x2, x2 ** 2, x1 * x2])
    Phi = np.column_stack(cols)
    return Phi


def bayes_linreg_posterior(
    Phi: np.ndarray,
    y: np.ndarray,
    tau2: float = 10.0,
    a0: float = 2.0,
    b0: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Conjugate Normal-Inverse-Gamma posterior for linear regression.

    Prior: beta | sigma^2 ~ N(beta0, sigma^2 V0), sigma^2 ~ InvGamma(a0,b0), with beta0=0, V0=tau2 I.

    Returns: (beta_n, Vn, a_n, b_n) where
      beta | sigma^2, y ~ N(beta_n, sigma^2 Vn), sigma^2 | y ~ InvGamma(a_n, b_n)
    """
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, p = Phi.shape
    V0 = tau2 * np.eye(p)
    V0_inv = np.eye(p) / tau2
    XtX = Phi.T @ Phi
    Xty = Phi.T @ y
    Vn_inv = V0_inv + XtX
    Vn = np.linalg.inv(Vn_inv)
    beta_n = Vn @ Xty  # since beta0=0
    a_n = a0 + 0.5 * n
    b_n = b0 + 0.5 * (y @ y - beta_n @ Vn_inv @ beta_n)
    return beta_n, Vn, a_n, b_n


def sample_posterior(
    beta_n: np.ndarray,
    Vn: np.ndarray,
    a_n: float,
    b_n: float,
    S: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw S samples of (beta, sigma). Returns (beta_draws, sigma_draws).
    sigma is returned as standard deviation (sqrt of variance).
    """
    p = beta_n.shape[0]
    # Inv-Gamma(a,b) draw via 1/Gamma(a, 1/b)
    sigma2 = 1.0 / rng.gamma(shape=a_n, scale=1.0 / b_n, size=S)
    sigma = np.sqrt(sigma2)
    # beta | sigma^2 ~ N(beta_n, sigma^2 Vn)
    L = np.linalg.cholesky(Vn)
    z = rng.normal(size=(S, p))
    beta_draws = beta_n[None, :] + (sigma[:, None] * (z @ L.T))
    return beta_draws, sigma


def posterior_draws_and_loglik(
    X: np.ndarray,
    y: np.ndarray,
    S: int = 500,
    tau2: float = 10.0,
    a0: float = 2.0,
    b0: float = 1.0,
    seed: int = 0,
    return_pred: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build posterior mean draws f_{i,s} and per-observation log-likelihood matrix for PSIS.
    If return_pred is True, also returns posterior predictive draws y_pred (n,S).

    Returns
    - if return_pred is False (default): (y_obs, posterior_draws(n,S), log_lik(n,S))
    - if return_pred is True:  (y_obs, posterior_draws(n,S), log_lik(n,S), y_pred(n,S))
    """
    rng = np.random.default_rng(seed)
    Phi = design_matrix(X)
    beta_n, Vn, a_n, b_n = bayes_linreg_posterior(Phi, y, tau2=tau2, a0=a0, b0=b0)
    beta_draws, sigma = sample_posterior(beta_n, Vn, a_n, b_n, S=S, rng=rng)
    # Predictive mean draws f = Phi @ beta
    f = Phi @ beta_draws.T  # (n, S)
    # Log-likelihood per observation given (beta_s, sigma_s): Normal(y | f, sigma_s)
    y_obs = y.reshape(-1, 1)
    var = (sigma**2)[None, :]
    log_lik = -0.5 * (np.log(2.0 * np.pi * var) + ((y_obs - f) ** 2) / var)
    if return_pred:
        y_pred = f + rng.normal(size=f.shape) * sigma[None, :]
        return y.reshape(-1), f, log_lik, y_pred
    else:
        return y.reshape(-1), f, log_lik


def posterior_draws_loglik_and_pred(
    X: np.ndarray,
    y: np.ndarray,
    S: int = 500,
    tau2: float = 10.0,
    a0: float = 2.0,
    b0: float = 1.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper that ALWAYS returns posterior predictive draws as the 4th output.
    Returns (y_obs, f(n,S), log_lik(n,S), y_pred(n,S)).

    This mirrors posterior_draws_and_loglik(..., return_pred=True) without needing the flag.
    """
    y_obs, f, log_lik, y_pred = posterior_draws_and_loglik(
        X, y, S=S, tau2=tau2, a0=a0, b0=b0, seed=seed, return_pred=True
    )
    return y_obs, f, log_lik, y_pred
