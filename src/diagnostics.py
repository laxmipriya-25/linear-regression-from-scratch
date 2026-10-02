"""
diagnostics.py

STAGE 14: Reusable model-diagnostic / residual-analysis functions.

A residual is the difference between what actually happened and what
the model predicted:

    residual = y_true - y_pred

Residual analysis complements metrics like MSE/R^2: a model can have a
decent R^2 while still showing systematic patterns in its errors (e.g.
being consistently too high for certain inputs, or having errors that
grow with the predicted value). These functions compute the raw
residuals and summary statistics; src/model_diagnostics.py uses them to
actually analyze the Stage 11 multiple-regression model's errors.

No sklearn or scipy - NumPy only.
"""

import numpy as np


def calculate_residuals(y_true, y_pred):
    """
    Compute residuals: residual = y_true - y_pred

    A positive residual means the model under-predicted (actual was
    higher than predicted); a negative residual means it over-predicted.

    Parameters
    ----------
    y_true : numpy.ndarray, shape (n_samples,)
        Actual (ground-truth) values.
    y_pred : numpy.ndarray, shape (n_samples,)
        Predicted values.

    Returns
    -------
    residuals : numpy.ndarray, shape (n_samples,)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_pred has shape {y_pred.shape}."
        )

    return y_true - y_pred


def residual_statistics(residuals):
    """
    Compute summary statistics describing a set of residuals.

    Parameters
    ----------
    residuals : numpy.ndarray, shape (n_samples,)

    Returns
    -------
    stats : dict with keys:
        mean, median, std, min, max, mean_absolute, sum
    """
    residuals = np.asarray(residuals)

    if residuals.size == 0:
        raise ValueError("Cannot compute statistics on empty residuals array.")

    return {
        "mean": np.mean(residuals),
        "median": np.median(residuals),
        "std": np.std(residuals),
        "min": np.min(residuals),
        "max": np.max(residuals),
        "mean_absolute": np.mean(np.abs(residuals)),
        "sum": np.sum(residuals),
    }


def calculate_predictions_and_residuals(y_true, y_pred):
    """
    Convenience wrapper bundling predictions, residuals, and absolute
    residuals together for a single (y_true, y_pred) pair.

    Parameters
    ----------
    y_true : numpy.ndarray, shape (n_samples,)
    y_pred : numpy.ndarray, shape (n_samples,)

    Returns
    -------
    result : dict with keys:
        predictions : numpy.ndarray, shape (n_samples,) - same as y_pred
        residuals : numpy.ndarray, shape (n_samples,)
        absolute_residuals : numpy.ndarray, shape (n_samples,)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_pred has shape {y_pred.shape}."
        )
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have matching lengths.")

    residuals = calculate_residuals(y_true, y_pred)
    absolute_residuals = np.abs(residuals)

    return {
        "predictions": y_pred,
        "residuals": residuals,
        "absolute_residuals": absolute_residuals,
    }


def print_residual_statistics(label, stats):
    """Pretty-print a residual_statistics() dict under a given label."""
    print(f"\n--- {label} ---")
    print(f"Mean residual: {stats['mean']:.4f}")
    print(f"Median residual: {stats['median']:.4f}")
    print(f"Std residual: {stats['std']:.4f}")
    print(f"Min residual: {stats['min']:.4f}")
    print(f"Max residual: {stats['max']:.4f}")
    print(f"Mean absolute residual: {stats['mean_absolute']:.4f}")
    print(f"Residual sum: {stats['sum']:.4f}")


if __name__ == "__main__":
    print("--- Testing Stage 14: diagnostics.py ---")

    y_true_demo = np.array([55.0, 60.0, 65.0, 70.0, 75.0])
    y_pred_demo = np.array([57.0, 58.0, 65.0, 74.0, 70.0])

    residuals = calculate_residuals(y_true_demo, y_pred_demo)
    print(f"y_true: {y_true_demo}")
    print(f"y_pred: {y_pred_demo}")
    print(f"residuals: {residuals}")
    # Expected: [-2, 2, 0, -4, 5]

    stats = residual_statistics(residuals)
    print_residual_statistics("Demo Residual Statistics", stats)

    bundle = calculate_predictions_and_residuals(y_true_demo, y_pred_demo)
    print(f"\npredictions: {bundle['predictions']}")
    print(f"residuals: {bundle['residuals']}")
    print(f"absolute_residuals: {bundle['absolute_residuals']}")