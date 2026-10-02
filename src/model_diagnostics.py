"""
model_diagnostics.py

STAGE 14: Residual / diagnostic analysis of the Stage 11 multiple linear
regression model, trained on standardized features.

Reuses (does not reimplement):
    - load_multiple_dataset(), compute_hypothesis(), gradient_descent(),
      FEATURE_NAMES    from multiple_linear_regression.py
    - train_test_split()                               from utils.py
    - mean_squared_error(), root_mean_squared_error(),
      r_squared()                                        from metrics.py
    - standardize_features()                             from scaling.py
    - calculate_residuals(), residual_statistics(),
      calculate_predictions_and_residuals(),
      print_residual_statistics()                        from diagnostics.py
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
from diagnostics import (
    calculate_residuals,
    residual_statistics,
    calculate_predictions_and_residuals,
    print_residual_statistics,
)


PLOTS_DIR = "plots"


def ensure_plots_dir():
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)


def plot_actual_vs_predicted(y_test, y_test_pred,
                              save_path="plots/diagnostic_actual_vs_predicted.png"):
    """
    Scatter plot of actual vs predicted test values, with a y=x
    reference line. Points closer to the line indicate smaller errors,
    but closeness alone does not prove the model is free of problems.
    """
    ensure_plots_dir()

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_test_pred, alpha=0.6, color="steelblue", label="Test predictions")

    lims = [min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())]
    plt.plot(lims, lims, color="red", linewidth=2, linestyle="--", label="y = x (perfect prediction)")

    plt.xlabel("Actual exam score")
    plt.ylabel("Predicted exam score")
    plt.title("Actual vs Predicted (Test Set)")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: {save_path}")


def plot_residuals_vs_predicted(y_test_pred, residuals,
                                 save_path="plots/residuals_vs_predicted.png"):
    """
    Scatter plot of residuals vs predicted values, with a horizontal
    reference line at residual=0. Ideally points scatter randomly
    around zero with no obvious pattern (curve, funnel shape, etc.).
    """
    ensure_plots_dir()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_pred, residuals, alpha=0.6, color="darkorange", label="Test residuals")
    plt.axhline(0, color="black", linewidth=2, linestyle="--", label="Residual = 0")

    plt.xlabel("Predicted exam score")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("Residuals vs Predicted (Test Set)")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: {save_path}")


def plot_residual_distribution(residuals,
                                save_path="plots/residual_distribution.png"):
    """
    Histogram of test residuals, with a vertical reference line at 0,
    to visualize whether residuals are roughly centered around zero and
    whether there are unusually large errors in either direction.
    """
    ensure_plots_dir()

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=20, color="mediumseagreen", edgecolor="black", alpha=0.8)
    plt.axvline(0, color="black", linewidth=2, linestyle="--", label="Residual = 0")

    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Distribution of Test Residuals")
    plt.legend()
    plt.grid(True, axis="y")

    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: {save_path}")


def plot_residuals_vs_features(X_test_raw, residuals,
                                save_path="plots/residuals_vs_features.png"):
    """
    One subplot per feature: residual vs that feature's ORIGINAL
    (unstandardized) value, each with a horizontal zero-residual line.
    Original values are used (not standardized) purely for readability.
    """
    ensure_plots_dir()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, FEATURE_NAMES)):
        ax.scatter(X_test_raw[:, i], residuals, alpha=0.6, color="slateblue")
        ax.axhline(0, color="black", linewidth=2, linestyle="--")
        ax.set_xlabel(name)
        ax.set_ylabel("Residual")
        ax.set_title(f"Residuals vs {name}")
        ax.grid(True)

    fig.suptitle("Residuals vs Each Feature (Test Set, Original Units)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: {save_path}")


def pearson_correlation(x, y):
    """
    Pearson correlation coefficient between two 1D arrays, implemented
    manually with NumPy (no scipy/sklearn):

        r = cov(x, y) / (std(x) * std(y))
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered ** 2)) * np.sqrt(np.sum(y_centered ** 2))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def run_model_diagnostics():
    # ------------------------------------------------------------
    # 1-3. Load, split (SAME split as Stages 11-13), standardize
    # (training-set statistics only - no data leakage)
    # ------------------------------------------------------------
    X, y = load_multiple_dataset("data/multiple_regression_dataset.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

    X_train_scaled, X_test_scaled, feature_mean, feature_std = standardize_features(
        X_train, X_test
    )

    # ------------------------------------------------------------
    # 4. Train the multiple linear regression model (Stage 11's
    #    gradient_descent, reused as-is) on the TRAINING set only
    # ------------------------------------------------------------
    w, b, _ = gradient_descent(
        X_train_scaled, y_train, learning_rate=0.1, n_iterations=2000
    )

    # ------------------------------------------------------------
    # 5. Predictions for train and test sets
    # ------------------------------------------------------------
    y_train_pred = compute_hypothesis(X_train_scaled, w, b)
    y_test_pred = compute_hypothesis(X_test_scaled, w, b)

    # ------------------------------------------------------------
    # 6-7. Residuals and diagnostic statistics
    # ------------------------------------------------------------
    train_residuals = calculate_residuals(y_train, y_train_pred)
    test_residuals = calculate_residuals(y_test, y_test_pred)

    train_stats = residual_statistics(train_residuals)
    test_stats = residual_statistics(test_residuals)

    print_residual_statistics("Training Residual Statistics", train_stats)
    print_residual_statistics("Test Residual Statistics", test_stats)

    # Existing metrics, for context alongside the residual statistics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    train_r2 = r_squared(y_train, y_train_pred)
    test_r2 = r_squared(y_test, y_test_pred)

    print("\n--- Model Performance Metrics ---")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Training R\u00b2: {train_r2:.4f}")
    print(f"Test R\u00b2: {test_r2:.4f}")

    # ------------------------------------------------------------
    # Part 4-6: diagnostic plots
    # ------------------------------------------------------------
    plot_actual_vs_predicted(y_test, y_test_pred)
    plot_residuals_vs_predicted(y_test_pred, test_residuals)
    plot_residual_distribution(test_residuals)

    print(
        "\nFor the residuals-vs-predicted plot: ideally, residuals scatter "
        "around zero without an obvious systematic pattern (a curve, a "
        "funnel/fan shape, or distinct clusters would suggest the model is "
        "missing some structure in the data). A handful of points not being "
        "perfectly random does not by itself prove a problem."
    )
    print(
        "\nFor the residual distribution histogram: this helps show whether "
        "residuals are roughly centered around zero and whether any residuals "
        "are unusually large in magnitude. The residuals are not expected to "
        "be perfectly normally distributed for this to be a reasonable model."
    )

    # ------------------------------------------------------------
    # Part 7: residuals vs each ORIGINAL (unstandardized) feature
    # ------------------------------------------------------------
    plot_residuals_vs_features(X_test, test_residuals)

    # ------------------------------------------------------------
    # Part 8: top 5 largest absolute test residuals
    # ------------------------------------------------------------
    bundle = calculate_predictions_and_residuals(y_test, y_test_pred)
    abs_residuals = bundle["absolute_residuals"]
    top5_idx = np.argsort(abs_residuals)[::-1][:5]

    print("\n--- Top 5 Largest Test Residuals ---")
    for rank, idx in enumerate(top5_idx, start=1):
        print(f"\n#{rank}")
        for name, val in zip(FEATURE_NAMES, X_test[idx]):
            print(f"  {name}: {val:.4f}")
        print(f"  Actual exam score: {y_test[idx]:.4f}")
        print(f"  Predicted exam score: {y_test_pred[idx]:.4f}")
        print(f"  Residual: {test_residuals[idx]:.4f}")
        print(f"  Absolute residual: {abs_residuals[idx]:.4f}")
    print(
        "\nThese are the observations with the largest prediction errors in "
        "the test set. They are not automatically labeled 'outliers' - they "
        "may simply reflect noise in the data, or may warrant further "
        "investigation."
    )

    # ------------------------------------------------------------
    # Part 9: Pearson correlation between test residuals and each
    # ORIGINAL feature (manual NumPy implementation, no scipy/sklearn)
    # ------------------------------------------------------------
    print("\n--- Residual Correlation with Original Features ---")
    correlations = {}
    for i, name in enumerate(FEATURE_NAMES):
        corr = pearson_correlation(X_test[:, i], test_residuals)
        correlations[name] = corr
        print(f"Residual correlation with {name}: {corr:.4f}")

    print(
        "\nA correlation near zero suggests little linear association between "
        "that feature and the residuals, while a stronger correlation may "
        "indicate that the model's errors still vary systematically with that "
        "feature. Correlation alone is not treated as proof of a modeling "
        "problem."
    )

    # ------------------------------------------------------------
    # Part 12: interpretation, based only on the actual results above
    # ------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Interpretation")
    print("=" * 50)

    mean_close_to_zero = abs(test_stats["mean"]) < 0.1 * test_stats["std"]
    print(
        f"1. The test mean residual is {test_stats['mean']:.4f}, which is "
        f"{'close to' if mean_close_to_zero else 'somewhat far from'} zero "
        f"relative to the residual standard deviation ({test_stats['std']:.4f})."
    )
    print(
        f"2. The residuals appear approximately centered around zero based on "
        f"the mean ({test_stats['mean']:.4f}) and median ({test_stats['median']:.4f}) "
        f"being close to each other and near 0."
    )
    print(
        "3. Whether the residual-vs-predicted plot shows an obvious systematic "
        "pattern (curve, funnel shape, or clustering) should be judged visually "
        "from plots/residuals_vs_predicted.png - the printed statistics alone "
        "cannot confirm or rule this out."
    )
    print(
        f"4. The residual distribution ranges from {test_stats['min']:.4f} to "
        f"{test_stats['max']:.4f}, with a mean absolute residual of "
        f"{test_stats['mean_absolute']:.4f}; see plots/residual_distribution.png "
        f"to judge whether any errors stand out as unusually large."
    )

    strongest_feature = max(correlations, key=lambda k: abs(correlations[k]))
    print(
        f"5. Among the four features, '{strongest_feature}' shows the strongest "
        f"(not necessarily strong) residual correlation at "
        f"{correlations[strongest_feature]:.4f}. All other features show "
        f"correlations of similar or smaller magnitude - see the full list above."
    )
    print(
        f"6. The single largest test prediction error was "
        f"{abs_residuals[top5_idx[0]]:.4f} (see '--- Top 5 Largest Test "
        f"Residuals ---' above for full details of that observation)."
    )

    print(
        "\nNote: these are descriptive observations about THIS model's errors "
        "on THIS test split, not a claim that the model satisfies all linear "
        "regression assumptions."
    )


if __name__ == "__main__":
    run_model_diagnostics()