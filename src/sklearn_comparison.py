"""
sklearn_comparison.py

STAGE 10: Compare the from-scratch Linear Regression implementation
against scikit-learn's LinearRegression, trained on the SAME data split.

IMPORTANT:
Scikit-learn is used ONLY here, for verification/comparison purposes.
The actual from-scratch model (src/linear_regression.py) is not modified
or replaced - it is simply imported and trained as-is.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as sk_mse
from sklearn.metrics import r2_score

try:
    from sklearn.metrics import root_mean_squared_error as sk_rmse_func
    HAS_SK_RMSE = True
except ImportError:
    # Older sklearn versions don't have root_mean_squared_error;
    # fall back to computing it manually from MSE.
    HAS_SK_RMSE = False

# Reuse our own existing implementations - nothing here is reimplemented
from utils import load_dataset, train_test_split
from linear_regression import gradient_descent, compute_hypothesis
from metrics import mean_squared_error, root_mean_squared_error, r_squared


def run_sklearn_comparison():
    """
    Train both the from-scratch model and sklearn's LinearRegression on
    the same train/test split, then print a side-by-side comparison.
    """
    # ------------------------------------------------------------
    # Load the SAME dataset and SAME train/test split used elsewhere
    # ------------------------------------------------------------
    X, y = load_dataset("data/dataset.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, seed=42
    )

    # ------------------------------------------------------------
    # From-scratch model (Stage 5/6): train with the same hyperparameters
    # used throughout the project
    # ------------------------------------------------------------
    w_scratch, b_scratch, _ = gradient_descent(
        X_train, y_train, learning_rate=0.01, n_iterations=1000, verbose=False
    )

    y_train_pred_scratch = compute_hypothesis(X_train, w_scratch, b_scratch)
    y_test_pred_scratch = compute_hypothesis(X_test, w_scratch, b_scratch)

    scratch_train_mse = mean_squared_error(y_train, y_train_pred_scratch)
    scratch_test_mse = mean_squared_error(y_test, y_test_pred_scratch)
    scratch_train_rmse = root_mean_squared_error(y_train, y_train_pred_scratch)
    scratch_test_rmse = root_mean_squared_error(y_test, y_test_pred_scratch)
    scratch_train_r2 = r_squared(y_train, y_train_pred_scratch)
    scratch_test_r2 = r_squared(y_test, y_test_pred_scratch)

    # ------------------------------------------------------------
    # Scikit-learn model: trained on the exact same X_train/y_train
    # ------------------------------------------------------------
    sk_model = LinearRegression()
    sk_model.fit(X_train, y_train)

    y_train_pred_sklearn = sk_model.predict(X_train)
    y_test_pred_sklearn = sk_model.predict(X_test)

    sklearn_train_mse = sk_mse(y_train, y_train_pred_sklearn)
    sklearn_test_mse = sk_mse(y_test, y_test_pred_sklearn)

    if HAS_SK_RMSE:
        sklearn_train_rmse = sk_rmse_func(y_train, y_train_pred_sklearn)
        sklearn_test_rmse = sk_rmse_func(y_test, y_test_pred_sklearn)
    else:
        sklearn_train_rmse = np.sqrt(sklearn_train_mse)
        sklearn_test_rmse = np.sqrt(sklearn_test_mse)

    sklearn_train_r2 = r2_score(y_train, y_train_pred_sklearn)
    sklearn_test_r2 = r2_score(y_test, y_test_pred_sklearn)

    sklearn_coef = sk_model.coef_[0]
    sklearn_intercept = sk_model.intercept_

    # ------------------------------------------------------------
    # Print the comparison
    # ------------------------------------------------------------
    print("=" * 40)
    print("Stage 10: From-Scratch vs Scikit-Learn")
    print("=" * 40)

    print("\nFROM-SCRATCH MODEL")
    print(f"Coefficient: {w_scratch[0]:.4f}")
    print(f"Intercept: {b_scratch:.4f}")
    print(f"Training MSE: {scratch_train_mse:.4f}")
    print(f"Test MSE: {scratch_test_mse:.4f}")
    print(f"Training RMSE: {scratch_train_rmse:.4f}")
    print(f"Test RMSE: {scratch_test_rmse:.4f}")
    print(f"Training R\u00b2: {scratch_train_r2:.4f}")
    print(f"Test R\u00b2: {scratch_test_r2:.4f}")

    print("\nSCIKIT-LEARN MODEL")
    print(f"Coefficient: {sklearn_coef:.4f}")
    print(f"Intercept: {sklearn_intercept:.4f}")
    print(f"Training MSE: {sklearn_train_mse:.4f}")
    print(f"Test MSE: {sklearn_test_mse:.4f}")
    print(f"Training RMSE: {sklearn_train_rmse:.4f}")
    print(f"Test RMSE: {sklearn_test_rmse:.4f}")
    print(f"Training R\u00b2: {sklearn_train_r2:.4f}")
    print(f"Test R\u00b2: {sklearn_test_r2:.4f}")

    print("\n" + "=" * 40)
    print("COMPARISON")
    print("=" * 40)

    coef_diff = abs(w_scratch[0] - sklearn_coef)
    intercept_diff = abs(b_scratch - sklearn_intercept)
    test_mse_diff = abs(scratch_test_mse - sklearn_test_mse)
    test_rmse_diff = abs(scratch_test_rmse - sklearn_test_rmse)
    test_r2_diff = abs(scratch_test_r2 - sklearn_test_r2)

    print(f"\nCoefficient difference: {coef_diff:.4f}")
    print(f"Intercept difference: {intercept_diff:.4f}")
    print(f"Test MSE difference: {test_mse_diff:.4f}")
    print(f"Test RMSE difference: {test_rmse_diff:.4f}")
    print(f"Test R\u00b2 difference: {test_r2_diff:.4f}")

    print(
        "\nThe scikit-learn implementation provides a reference solution. "
        "The from-scratch implementation should produce similar predictions "
        "and evaluation metrics, with small differences possible because the "
        "from-scratch model uses gradient descent with a fixed learning rate "
        "and number of iterations, while scikit-learn solves for the optimal "
        "parameters directly (via least squares)."
    )


if __name__ == "__main__":
    run_sklearn_comparison()