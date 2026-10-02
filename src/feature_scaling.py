"""
feature_scaling.py

STAGE 12: Demonstrates the effect of feature scaling on gradient descent
convergence, using the existing Stage 11 multiple linear regression model
and dataset.

This script does NOT retrain or change Stage 11's own script - it reuses
Stage 11's compute_hypothesis() and gradient_descent() functions (not
reimplemented) on the SAME dataset and SAME train/test split, once
without scaling and once with standardization from scaling.py.
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
    gradient_descent,
    FEATURE_NAMES,
)


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


def print_metrics(label, metrics):
    print(f"\n{label}")
    print("-" * len(label))
    print(f"Training MSE: {metrics['train_mse']:.4f}")
    print(f"Test MSE: {metrics['test_mse']:.4f}")
    print(f"Training RMSE: {metrics['train_rmse']:.4f}")
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Training R\u00b2: {metrics['train_r2']:.4f}")
    print(f"Test R\u00b2: {metrics['test_r2']:.4f}")


def plot_comparison(
    loss_unscaled, loss_scaled, save_path="plots/feature_scaling_comparison.png"
):
    """
    Plot both loss curves on one chart for comparison.

    The unscaled curve typically explodes to inf/NaN within the first
    few hundred iterations (that is the whole point of this demo), so we
    only plot its FINITE portion and annotate where it diverged. A log
    scale on the y-axis is used since the two curves span wildly
    different magnitudes.
    """
    plots_dir = os.path.dirname(save_path)
    if plots_dir and not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    loss_unscaled = np.array(loss_unscaled, dtype=float)
    loss_scaled = np.array(loss_scaled, dtype=float)

    # The unscaled run doesn't just go non-finite - it blows up to
    # astronomical finite values (e.g. 1e300+) in the iterations just
    # before that. Values that extreme break matplotlib's log-scale
    # autoscaling entirely (the axis can end up showing nothing useful).
    # So we cap the DISPLAYED unscaled values at a multiple of the
    # scaled curve's own max loss - high enough to clearly show the
    # explosive runaway growth, low enough to keep the plot readable.
    display_cap = loss_scaled.max() * 50

    nonfinite_mask = ~np.isfinite(loss_unscaled)
    over_cap_mask = loss_unscaled > display_cap
    cutoff_mask = nonfinite_mask | over_cap_mask

    if cutoff_mask.any():
        diverge_at = int(np.argmax(cutoff_mask))  # first True index
        unscaled_plot_values = loss_unscaled[:diverge_at]
        unscaled_label = f"Unscaled (diverged by iteration {diverge_at})"
    else:
        unscaled_plot_values = loss_unscaled
        unscaled_label = "Unscaled"

    plt.figure(figsize=(9, 6))

    if len(unscaled_plot_values) > 0:
        # markers, since the unscaled curve may be only a handful of points
        plt.plot(
            np.arange(len(unscaled_plot_values)),
            unscaled_plot_values,
            color="crimson",
            linewidth=2,
            marker="o",
            markersize=4,
            label=unscaled_label,
        )
    plt.plot(
        np.arange(len(loss_scaled)),
        loss_scaled,
        color="green",
        linewidth=2,
        label="Standardized",
    )

    plt.yscale("log")
    plt.ylim(bottom=max(loss_scaled.min() * 0.5, 1e-3), top=display_cap * 2)
    plt.xlim(left=-5, right=len(loss_scaled) + 10)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss (log scale)")
    plt.title("Feature Scaling: Gradient Descent Convergence Comparison")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.savefig(save_path)
    plt.close()

    print(f"\nPlot saved to: {save_path}")


def run_feature_scaling_comparison():
    # ------------------------------------------------------------
    # 1-3. Load the EXISTING Stage 11 dataset and split it the SAME way
    # ------------------------------------------------------------
    X, y = load_multiple_dataset("data/multiple_regression_dataset.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

    # ==============================================================
    # EXPERIMENT 1: Unscaled features, using the existing Stage 11
    # gradient_descent() settings (learning_rate=0.001, n_iterations=2000)
    # ==============================================================
    lr_unscaled = 0.001
    iters_unscaled = 2000

    print("--- Unscaled Features ---")
    print(f"Learning rate: {lr_unscaled}")
    print(f"Iterations: {iters_unscaled}")

    w_unscaled, b_unscaled, loss_unscaled = gradient_descent(
        X_train, y_train, learning_rate=lr_unscaled, n_iterations=iters_unscaled
    )

    y_train_pred_u = compute_hypothesis(X_train, w_unscaled, b_unscaled)
    y_test_pred_u = compute_hypothesis(X_test, w_unscaled, b_unscaled)
    metrics_unscaled = evaluate(y_train, y_train_pred_u, y_test, y_test_pred_u)

    n_finite_unscaled = int(np.sum(np.isfinite(loss_unscaled)))
    if n_finite_unscaled < iters_unscaled:
        print(
            f"\nWARNING: unscaled loss became non-finite (inf/NaN) after "
            f"{n_finite_unscaled} of {iters_unscaled} iterations. This is "
            f"gradient descent DIVERGING, not a bug - but it is not that "
            f"unscaled features inherently cause divergence. Rather, "
            f"learning_rate={lr_unscaled} is unstable for the CURRENT "
            f"magnitudes of these particular unscaled features (since "
            f"'attendance'/'previous_score' range much higher than "
            f"'hours_studied'/'assignments_completed', the gradient is "
            f"poorly conditioned at this step size). Feature scaling "
            f"improves this conditioning by putting every feature on a "
            f"comparable scale, which is what lets the standardized run "
            f"below use a much larger, stable learning rate."
        )

    # ==============================================================
    # EXPERIMENT 2: Standardized features.
    #
    # DATA LEAKAGE CHECK: mean and std below are computed from X_train
    # ONLY, inside standardize_features(). X_test is transformed using
    # those same training statistics - at no point does any statistic
    # get computed from X_test. This mirrors a real deployment, where
    # the test/production data is unseen at scaling-parameter time.
    # ==============================================================
    X_train_scaled, X_test_scaled, feature_mean, feature_std = standardize_features(
        X_train, X_test
    )

    # A learning rate chosen to be stable AND fast now that all features
    # are on a comparable (mean 0, std 1) scale
    lr_scaled = 0.1
    iters_scaled = 2000

    print("\n--- Standardized Features ---")
    print(f"Learning rate: {lr_scaled}")
    print(f"Iterations: {iters_scaled}")

    w_scaled, b_scaled, loss_scaled = gradient_descent(
        X_train_scaled, y_train, learning_rate=lr_scaled, n_iterations=iters_scaled
    )

    y_train_pred_s = compute_hypothesis(X_train_scaled, w_scaled, b_scaled)
    y_test_pred_s = compute_hypothesis(X_test_scaled, w_scaled, b_scaled)
    metrics_scaled = evaluate(y_train, y_train_pred_s, y_test, y_test_pred_s)

    # ------------------------------------------------------------
    # Part 7: convert scaled coefficients back to original feature units
    #   w_original = w_scaled / std
    #   b_original = b_scaled - sum(w_scaled * mean / std)
    # ------------------------------------------------------------
    w_original = w_scaled / feature_std
    b_original = b_scaled - np.sum(w_scaled * feature_mean / feature_std)

    # ==============================================================
    # Print full comparison
    # ==============================================================
    print("\n--- Stage 12: Feature Scaling Comparison ---")
    print_metrics("Unscaled Features", metrics_unscaled)
    print_metrics("Standardized Features", metrics_scaled)

    print("\nTraining feature means:")
    for name, m in zip(FEATURE_NAMES, feature_mean):
        print(f"{name}: {m:.4f}")

    print("\nTraining feature standard deviations:")
    for name, s in zip(FEATURE_NAMES, feature_std):
        print(f"{name}: {s:.4f}")

    print("\nEquivalent coefficients in original feature units")
    print("(derived from the standardized model - should be close to the")
    print("coefficients Stage 11 learned directly in original units):")
    for name, coef in zip(FEATURE_NAMES, w_original):
        print(f"{name}: {coef:.4f}")
    print(f"Intercept: {b_original:.4f}")

    # ==============================================================
    # Loss curve comparison plot
    # ==============================================================
    plot_comparison(loss_unscaled, loss_scaled)

    return metrics_unscaled, metrics_scaled


if __name__ == "__main__":
    run_feature_scaling_comparison()