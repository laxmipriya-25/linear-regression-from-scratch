"""
ridge_regression.py

STAGE 13: Ridge Regression (L2-regularized linear regression), implemented
from scratch with NumPy - no sklearn.Ridge.

This is a SEPARATE module. It does NOT replace or modify
multiple_linear_regression.py, which remains the plain (unregularized)
multiple linear regression implementation. ridge_regression.py reuses
multiple_linear_regression.py's compute_hypothesis() and
compute_gradients() (the plain MSE gradient), and simply adds the L2
penalty on top.

RIDGE OBJECTIVE:
    Loss = MSE + lambda * sum(w^2)

IMPORTANT - THE INTERCEPT IS NOT REGULARIZED:
Only the feature weights (w) are penalized, never the bias/intercept (b).
This is standard practice: the intercept just shifts predictions up or
down and doesn't contribute to overfitting the way large feature weights
do, so there's no reason to shrink it.

    gradient_w = MSE_gradient_w + 2 * lambda * w   <- penalty added here
    gradient_b = MSE_gradient_b                     <- unchanged, no penalty
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from utils import train_test_split
from metrics import mean_squared_error, root_mean_squared_error, r_squared
from scaling import standardize_features
from multiple_linear_regression import (
    load_multiple_dataset,
    compute_hypothesis,
    compute_gradients,
    gradient_descent as plain_gradient_descent,
    FEATURE_NAMES,
)


def ridge_gradient_descent(
    X, y, learning_rate=0.01, n_iterations=2000, lambda_=0.1, verbose=False
):
    """
    Train Ridge Regression parameters (w, b) using batch gradient descent
    on the L2-regularized objective: Loss = MSE + lambda * sum(w^2).

    The bias term b is NOT regularized - only the weights w are.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
    y : numpy.ndarray, shape (n_samples,)
    learning_rate : float
        Must be > 0.
    n_iterations : int
        Must be > 0.
    lambda_ : float
        L2 regularization strength. Must be >= 0. lambda_=0 reduces this
        exactly to plain (unregularized) gradient descent.
    verbose : bool
        If True, print loss at sensible intervals.

    Returns
    -------
    w : numpy.ndarray, shape (n_features,)
    b : float
    loss_history : list of float
        The RIDGE loss (MSE + lambda*sum(w^2)) recorded at every
        iteration - NOT plain MSE - so the regularization term's
        contribution is visible in the recorded training curve.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError(f"X must be a 2D numpy array, got shape "
                          f"{getattr(X, 'shape', None)}.")
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError(f"y must be a 1D numpy array, got shape "
                          f"{getattr(y, 'shape', None)}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have incompatible shapes: {X.shape[0]} vs {y.shape[0]} samples."
        )
    if lambda_ < 0:
        raise ValueError(f"lambda_ must be >= 0, got {lambda_}.")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")
    if n_iterations <= 0:
        raise ValueError(f"n_iterations must be > 0, got {n_iterations}.")

    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0

    loss_history = []
    print_every = max(n_iterations // 10, 1)

    for iteration in range(n_iterations):
        y_hat = compute_hypothesis(X, w, b)

        # Ridge loss = MSE + lambda * sum(w^2)   (the "L2 penalty")
        mse = mean_squared_error(y, y_hat)
        ridge_loss = mse + lambda_ * np.sum(w ** 2)
        loss_history.append(ridge_loss)

        # Plain MSE gradients, reused from multiple_linear_regression.py
        dw_mse, db = compute_gradients(X, y, y_hat)

        # Add the L2 penalty gradient (derivative of lambda*sum(w^2) is
        # 2*lambda*w) ONLY to the weight gradient - the bias gradient
        # db is left exactly as the plain MSE gradient, i.e. the
        # intercept is never regularized.
        dw = dw_mse + 2 * lambda_ * w

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if verbose and (iteration % print_every == 0 or iteration == n_iterations - 1):
            print(f"Iteration {iteration:5d} | Ridge loss: {ridge_loss:.4f}")

    return w, b, loss_history


def evaluate(y_train, y_train_pred, y_test, y_test_pred):
    """Compute MSE/RMSE/R^2 for train and test sets using existing metrics.py."""
    return {
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "train_rmse": root_mean_squared_error(y_train, y_train_pred),
        "test_rmse": root_mean_squared_error(y_test, y_test_pred),
        "train_r2": r_squared(y_train, y_train_pred),
        "test_r2": r_squared(y_test, y_test_pred),
    }


def print_metrics_block(title, metrics, lambda_=None):
    print(f"\n--- {title} ---")
    if lambda_ is not None:
        print(f"Lambda: {lambda_}")
    print(f"Training MSE: {metrics['train_mse']:.4f}")
    print(f"Test MSE: {metrics['test_mse']:.4f}")
    print(f"Training RMSE: {metrics['train_rmse']:.4f}")
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Training R\u00b2: {metrics['train_r2']:.4f}")
    print(f"Test R\u00b2: {metrics['test_r2']:.4f}")


def plot_coefficient_shrinkage(
    lambdas, coefficients_by_lambda, save_path="plots/ridge_coefficient_shrinkage.png",
    excluded_lambdas=None
):
    """
    Plot coefficient value vs. lambda, one line per feature.

    lambda=0 is included on the x-axis using a "symlog" scale, which
    behaves linearly near 0 and logarithmically for larger values - this
    displays lambda=0 correctly without the undefined log(0) problem a
    plain log-scale axis would have.
    """
    plots_dir = os.path.dirname(save_path)
    if plots_dir and not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    coefficients_by_lambda = np.array(coefficients_by_lambda)  # shape (n_lambdas, n_features)

    plt.figure(figsize=(9, 6))
    for feature_idx, name in enumerate(FEATURE_NAMES):
        plt.plot(
            lambdas,
            coefficients_by_lambda[:, feature_idx],
            marker="o",
            linewidth=2,
            label=name,
        )

    plt.xscale("symlog", linthresh=0.01)
    plt.xlabel("Lambda (symlog scale)")
    plt.ylabel("Coefficient value")
    title = "Ridge Regression: Coefficient Shrinkage vs Lambda"
    if excluded_lambdas:
        title += f"\n(lambda={excluded_lambdas} excluded - still unstable, see console)"
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.savefig(save_path)
    plt.close()
    print(f"\nPlot saved to: {save_path}")


def plot_performance_vs_lambda(
    lambdas, train_mse_list, test_mse_list,
    save_path="plots/ridge_performance_vs_lambda.png", excluded_lambdas=None
):
    """
    Plot Training MSE and Test MSE vs lambda (symlog x-axis, handles
    lambda=0 correctly - see plot_coefficient_shrinkage for why).
    """
    plots_dir = os.path.dirname(save_path)
    if plots_dir and not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plt.figure(figsize=(9, 6))
    plt.plot(lambdas, train_mse_list, marker="o", linewidth=2,
              color="steelblue", label="Training MSE")
    plt.plot(lambdas, test_mse_list, marker="o", linewidth=2,
              color="crimson", label="Test MSE")

    plt.xscale("symlog", linthresh=0.01)
    plt.xlabel("Lambda (symlog scale)")
    plt.ylabel("MSE")
    title = "Ridge Regression: Performance vs Lambda"
    if excluded_lambdas:
        title += f"\n(lambda={excluded_lambdas} excluded - still unstable, see console)"
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: {save_path}")


def plot_ridge_loss_curves(
    loss_curves, save_path="plots/ridge_loss_curves.png"
):
    """
    Compare training loss curves for a few lambda values.

    Parameters
    ----------
    loss_curves : dict
        Mapping of label -> loss_history list. Each loss_history is the
        RIDGE loss (MSE + lambda*sum(w^2)), not plain MSE - for
        lambda=0 these two are identical, but for lambda>0 the plotted
        curve includes the regularization term's contribution.
    """
    plots_dir = os.path.dirname(save_path)
    if plots_dir and not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plt.figure(figsize=(9, 6))
    colors = ["green", "darkorange", "purple"]
    for (label, loss_history), color in zip(loss_curves.items(), colors):
        plt.plot(
            np.arange(len(loss_history)), loss_history,
            linewidth=2, color=color, label=label
        )

    plt.xlabel("Iteration")
    plt.ylabel("Loss (MSE + \u03bb\u00b7\u03a3w\u00b2)")
    plt.title("Ridge Regression: Training Loss Curves")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: {save_path}")


def run_ridge_regression_pipeline():
    # ------------------------------------------------------------
    # Load the EXISTING Stage 11 dataset, split, and standardize -
    # exactly as Stage 12 does, so this comparison is fair (Part 9):
    # same dataset, same split, same scaling for every model below.
    # ------------------------------------------------------------
    X, y = load_multiple_dataset("data/multiple_regression_dataset.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

    # mean/std computed from X_train ONLY (see scaling.py) - no data leakage
    X_train_scaled, X_test_scaled, feature_mean, feature_std = standardize_features(
        X_train, X_test
    )

    learning_rate = 0.01
    n_iterations = 2000

    # ==============================================================
    # PART 4: Baseline comparison - plain linear regression vs Ridge
    # (lambda=0.1), same scaled data, same learning_rate/n_iterations
    # ==============================================================
    w_plain, b_plain, _ = plain_gradient_descent(
        X_train_scaled, y_train, learning_rate=learning_rate, n_iterations=n_iterations
    )
    y_train_pred_plain = compute_hypothesis(X_train_scaled, w_plain, b_plain)
    y_test_pred_plain = compute_hypothesis(X_test_scaled, w_plain, b_plain)
    metrics_plain = evaluate(y_train, y_train_pred_plain, y_test, y_test_pred_plain)

    w_ridge01, b_ridge01, _ = ridge_gradient_descent(
        X_train_scaled, y_train, learning_rate=learning_rate,
        n_iterations=n_iterations, lambda_=0.1
    )
    y_train_pred_r01 = compute_hypothesis(X_train_scaled, w_ridge01, b_ridge01)
    y_test_pred_r01 = compute_hypothesis(X_test_scaled, w_ridge01, b_ridge01)
    metrics_ridge01 = evaluate(y_train, y_train_pred_r01, y_test, y_test_pred_r01)

    print("=" * 50)
    print("Stage 13: Ridge Regression")
    print("=" * 50)
    print_metrics_block("Standard Linear Regression", metrics_plain)
    print_metrics_block("Ridge Regression", metrics_ridge01, lambda_=0.1)

    # ==============================================================
    # PART 5: Lambda sweep
    # ==============================================================
    lambdas = [0, 0.01, 0.1, 1, 10, 100]
    sweep_results = []       # list of dicts: lambda, learning_rate, metrics, w, b
    loss_histories_by_lambda = {}  # lambda -> loss_history (for Part 8)

    # ------------------------------------------------------------
    # Per-lambda learning rate selection.
    #
    # The weight-decay part of the Ridge update is:
    #     w <- w * (1 - 2*lambda*learning_rate) - learning_rate * grad_MSE
    # This is only numerically stable when 2*lambda*learning_rate < 1.
    # At the base learning_rate=0.01, this holds for every lambda up to
    # (but not including) 50 - so lambda=0, 0.01, 0.1, 1, 10 all keep
    # the SAME learning_rate=0.01 as before (their results are therefore
    # UNCHANGED from the previous run). Only lambda=100 exceeds this
    # bound (2*100*0.01 = 2 >= 1) and needs a smaller, stable learning
    # rate - computed here with a small safety margin (0.9 instead of
    # 1.0) rather than hand-picked, so the fix generalizes to any lambda.
    # This directly demonstrates that the earlier lambda=100 instability
    # was a property of the OPTIMIZER STEP SIZE, not of Ridge
    # regularization itself: the same lambda=100 penalty, trained with
    # an appropriately smaller learning rate, converges normally.
    # ------------------------------------------------------------
    STABILITY_SAFETY_MARGIN = 0.9

    for lam in lambdas:
        if lam == 0:
            lr_used = learning_rate
        else:
            max_stable_lr = STABILITY_SAFETY_MARGIN / (2 * lam)
            lr_used = min(learning_rate, max_stable_lr)

        print(f"Lambda={lam}: using learning_rate={lr_used}")

        w_lam, b_lam, loss_hist = ridge_gradient_descent(
            X_train_scaled, y_train, learning_rate=lr_used,
            n_iterations=n_iterations, lambda_=lam
        )
        y_train_pred_lam = compute_hypothesis(X_train_scaled, w_lam, b_lam)
        y_test_pred_lam = compute_hypothesis(X_test_scaled, w_lam, b_lam)
        metrics_lam = evaluate(y_train, y_train_pred_lam, y_test, y_test_pred_lam)

        sweep_results.append({
            "lambda": lam, "learning_rate": lr_used,
            "w": w_lam, "b": b_lam, "metrics": metrics_lam
        })
        loss_histories_by_lambda[lam] = loss_hist

    print("\n" + "=" * 50)
    print("Lambda sweep results")
    print("=" * 50)
    header = (f"{'Lambda':>8} | {'LR':>8} | {'Train MSE':>14} | {'Test MSE':>14} "
              f"| {'Train R2':>10} | {'Test R2':>10}")
    print(header)
    print("-" * len(header))
    for res in sweep_results:
        m = res["metrics"]
        print(
            f"{res['lambda']:>8} | {res['learning_rate']:>8} | {m['train_mse']:>14.4f} "
            f"| {m['test_mse']:>14.4f} | {m['train_r2']:>10.4f} | {m['test_r2']:>10.4f}"
        )

    print("\nCoefficients by lambda:")
    coef_header = f"{'Lambda':>8} | " + " | ".join(f"{name:>22}" for name in FEATURE_NAMES)
    print(coef_header)
    print("-" * len(coef_header))
    for res in sweep_results:
        coef_str = " | ".join(f"{val:>22.4f}" for val in res["w"])
        print(f"{res['lambda']:>8} | {coef_str}")

    # Identify (not "declare universally optimal") the lambda with the
    # lowest test MSE FOR THIS DATASET AND SPLIT, considering only
    # numerically STABLE results - comparing against an exploded,
    # diverged value would be meaningless. Stability is now checked
    # directly against each run's actual output (finite w, b, metrics),
    # since each lambda may use its own (already-adjusted) learning
    # rate rather than one fixed rate for all.
    def is_finite_result(res):
        return (
            np.all(np.isfinite(res["w"]))
            and np.isfinite(res["b"])
            and np.isfinite(res["metrics"]["test_mse"])
        )

    stable_results = [r for r in sweep_results if is_finite_result(r)]
    unstable_results = [r for r in sweep_results if not is_finite_result(r)]

    if unstable_results:
        unstable_lambdas = [r["lambda"] for r in unstable_results]
        print(
            f"\nNOTE: lambda value(s) {unstable_lambdas} were still numerically "
            f"unstable even after the per-lambda learning rate adjustment above. "
            f"These are EXCLUDED from the plots below (but kept in the printed "
            f"table above) so the stable trend remains readable."
        )
    else:
        print(
            "\nAll lambda values (including lambda=100) converged to finite, "
            "stable results once given an appropriately smaller learning rate - "
            "confirming the earlier lambda=100 instability was caused by the "
            "optimizer's step size, not by Ridge regularization itself."
        )

    best = min(stable_results, key=lambda r: r["metrics"]["test_mse"])
    print(
        f"\nObservation: for THIS dataset/split (among numerically stable "
        f"lambda values), lambda={best['lambda']} produced the lowest test MSE "
        f"({best['metrics']['test_mse']:.4f}) among the values tried. This is an "
        f"empirical observation for this experiment only, not a universal claim "
        f"about the best lambda in general."
    )

    # ==============================================================
    # PART 6: Coefficient shrinkage plot (stable lambdas only - see note above)
    # ==============================================================
    excluded_lambdas = [r["lambda"] for r in unstable_results] or None
    stable_lambdas = [r["lambda"] for r in stable_results]
    stable_coefficients = [r["w"] for r in stable_results]
    plot_coefficient_shrinkage(
        stable_lambdas, stable_coefficients, excluded_lambdas=excluded_lambdas
    )

    # ==============================================================
    # PART 7: Performance vs lambda plot (stable lambdas only - see note above)
    # ==============================================================
    stable_train_mse = [r["metrics"]["train_mse"] for r in stable_results]
    stable_test_mse = [r["metrics"]["test_mse"] for r in stable_results]
    plot_performance_vs_lambda(
        stable_lambdas, stable_train_mse, stable_test_mse, excluded_lambdas=excluded_lambdas
    )

    # ==============================================================
    # PART 8: Loss curve comparison - lambda=0, 0.1, 10
    # (reusing the loss_history already recorded during the sweep above,
    # not retraining)
    # ==============================================================
    loss_curves = {
        "Standard Linear Regression (\u03bb=0)": loss_histories_by_lambda[0],
        "Ridge \u03bb=0.1": loss_histories_by_lambda[0.1],
        "Ridge \u03bb=10": loss_histories_by_lambda[10],
    }
    plot_ridge_loss_curves(loss_curves)

    # ==============================================================
    # PART 10: Verify the intercept was not regularized
    # ==============================================================
    print("\nIntercept across lambda values (NOT regularized - see values below):")
    for res in sweep_results:
        print(f"lambda={res['lambda']:>6}: intercept = {res['b']:.4f}")
    print(
        "\nThe intercept is not regularized because the L2 penalty is applied "
        "only to feature weights (w), never to the bias term (b). The small "
        "differences in intercept across lambda values above come indirectly "
        "from the weights changing (which changes what bias best fits the "
        "data), not from any direct penalty on the intercept itself."
    )

    return sweep_results


if __name__ == "__main__":
    run_ridge_regression_pipeline()