import numpy as np
from matplotlib import pyplot as plt

from bartpy.sklearnmodel import SklearnModel
from bartpy.conformal_bayes import (
    prepare_bart_loglik_and_draws,
    loo_residuals_via_psis,
    conformal_quantile,
    build_intervals,
)


def make_sine(n=200, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, 1))
    f = np.sin(X[:, 0])
    y = f + rng.normal(0.0, noise, size=n)
    return X, y, f


def main():
    X, y, f_true = make_sine(n=250, noise=0.25, seed=42)

    # Fit BART and store per-sample predictions
    model = SklearnModel(
        n_trees=100,
        n_chains=2,
        n_samples=200,
        n_burn=200,
        thin=0.5,
        store_in_sample_predictions=True,
        store_acceptance_trace=True,
        n_jobs=1,
    )
    model.fit(X, y)

    # Prepare matrices for PSIS-LOO
    y_obs, y_draws, log_lik = prepare_bart_loglik_and_draws(model)

    # PSIS-LOO residuals and intervals
    residuals, k_diag, loo_mean = loo_residuals_via_psis(y_obs, y_draws, log_lik)
    q = conformal_quantile(residuals, alpha=0.1)  # 90% coverage target

    # Point predictions (posterior mean under full data)
    y_hat = loo_mean  # Using LOO mean as point summary for training points
    lo, hi = build_intervals(y_hat, q)

    coverage = np.mean((y_obs >= lo) & (y_obs <= hi))
    print(f"Empirical coverage (train, PSIS-LOO conformal, 90% nominal): {coverage:.3f}")

    # Diagnostics plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(k_diag[np.isfinite(k_diag)], bins=20, color="#4472c4")
    axes[0].set_title("Pareto k (PSIS)")
    axes[0].axvline(0.5, color='r', linestyle='--', label='0.5')
    axes[0].axvline(0.7, color='orange', linestyle='--', label='0.7')
    axes[0].legend()

    order = np.argsort(X[:, 0])
    axes[1].plot(X[order, 0], y_obs[order], 'k.', alpha=0.5, label='y')
    axes[1].plot(X[order, 0], y_hat[order], color='#2ca02c', label='LOO mean')
    axes[1].fill_between(
        X[order, 0], lo[order], hi[order], color='#9bd39b', alpha=0.4, label='Conformal 90%'
    )
    axes[1].set_title("PSIS-Conformal intervals (train)")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

