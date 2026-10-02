"""
linear_regression.py

This will contain the LinearRegression class, built from scratch
using NumPy (no sklearn).

STAGE 3: Hypothesis function (implemented below)
STAGE 5: Gradient Descent (implemented below)

Implemented in later stages:
    Stage 4 -> LinearRegression class skeleton (__init__, fit, predict)
      (Note: MSE itself lives in metrics.py, Stage 4)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from metrics import mean_squared_error, root_mean_squared_error, r_squared
from utils import load_dataset, train_test_split


def compute_hypothesis(X, w, b):
    """
    Compute the linear regression hypothesis (predictions): y_hat = Xw + b

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        Feature matrix.
    w : numpy.ndarray, shape (n_features,)
        Weight vector.
    b : float
        Bias (intercept) term.

    Returns
    -------
    y_hat : numpy.ndarray, shape (n_samples,)
        Predicted values for each sample.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}.")
    if X.shape[1] != w.shape[0]:
        raise ValueError(
            f"Feature/weight mismatch: X has {X.shape[1]} features, "
            f"w has {w.shape[0]} weights."
        )

    # Vectorized: dot product of X and w, then broadcast-add the bias
    y_hat = np.dot(X, w) + b
    return y_hat


def compute_gradients(X, y, y_hat):
    """
    Compute the gradients of MSE with respect to w and b.

        dw = -(2/n) * X^T (y - y_hat)
        db = -(2/n) * sum(y - y_hat)

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : numpy.ndarray, shape (n_samples,)
        Actual target values.
    y_hat : numpy.ndarray, shape (n_samples,)
        Predicted values.

    Returns
    -------
    dw : numpy.ndarray, shape (n_features,)
        Gradient of the loss with respect to w.
    db : float
        Gradient of the loss with respect to b.
    """
    n = X.shape[0]
    error = y - y_hat  # shape (n_samples,)

    dw = -(2 / n) * np.dot(X.T, error)   # shape (n_features,)
    db = -(2 / n) * np.sum(error)        # scalar

    return dw, db


def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000, verbose=False):
    """
    Train linear regression parameters (w, b) using batch gradient descent.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : numpy.ndarray, shape (n_samples,)
        Target values.
    learning_rate : float
        Step size for each parameter update.
    n_iterations : int
        Number of gradient descent iterations.
    verbose : bool
        If True, print loss at sensible intervals.

    Returns
    -------
    w : numpy.ndarray, shape (n_features,)
        Learned weights.
    b : float
        Learned bias.
    loss_history : list of float
        MSE recorded at every iteration (for later convergence plotting).
    """
    n_features = X.shape[1]

    # Start with w = 0 and b = 0, as specified
    w = np.zeros(n_features)
    b = 0.0

    loss_history = []
    print_every = max(n_iterations // 10, 1)  # print ~10 times total

    for iteration in range(n_iterations):
        y_hat = compute_hypothesis(X, w, b)
        loss = mean_squared_error(y, y_hat)
        loss_history.append(loss)

        dw, db = compute_gradients(X, y, y_hat)

        # Move w and b opposite to the gradient direction
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if verbose and (iteration % print_every == 0 or iteration == n_iterations - 1):
            print(f"Iteration {iteration:5d} | Loss: {loss:.4f} | w: {w} | b: {b:.4f}")

    return w, b, loss_history


def plot_regression_line(X, y, w, b, save_path="plots/linear_regression_fit.png"):
    """
    Plot actual data points and the learned regression line, and save to disk.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, 1)
        Feature matrix (single feature: hours_studied).
    y : numpy.ndarray, shape (n_samples,)
        Actual target values (exam_score).
    w : numpy.ndarray, shape (1,)
        Learned weight(s).
    b : float
        Learned bias.
    save_path : str
        Path (relative to project root) where the figure is saved.
    """
    # Create the plots/ directory automatically if it doesn't exist
    plots_dir = os.path.dirname(save_path)
    if plots_dir and not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Build a smooth line across the full range of X, rather than only
    # using the (possibly few/unevenly spaced) training points
    x_min, x_max = X.min(), X.max()
    x_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_line = compute_hypothesis(x_line, w, b)

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color="steelblue", alpha=0.6, label="Actual data")
    plt.plot(x_line, y_line, color="red", linewidth=2, label="Regression line")

    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.title("Linear Regression From Scratch")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()

    print(f"Plot saved to: {save_path}")


def plot_loss_curve(loss_history, save_path="plots/loss_curve.png"):
    """
    Plot MSE loss vs. iteration to visualize gradient descent convergence.

    Parameters
    ----------
    loss_history : list of float
        MSE recorded at every iteration during training (from gradient_descent()).
    save_path : str
        Path (relative to project root) where the figure is saved.
    """
    # Create the plots/ directory automatically if it doesn't exist
    plots_dir = os.path.dirname(save_path)
    if plots_dir and not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    iterations = np.arange(len(loss_history))

    plt.figure(figsize=(8, 6))
    plt.plot(iterations, loss_history, color="darkorange", linewidth=2)

    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Gradient Descent Convergence")
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()

    print(f"Plot saved to: {save_path}")


def run_stage6_pipeline():
    """
    Stage 6: End-to-end train/test pipeline on the real dataset.

    Loads data -> splits train/test -> trains via gradient descent ->
    predicts on both sets -> evaluates MSE on both sets -> prints summary.

    Uses only our own from-scratch implementations (no sklearn).
    """
    # a. Load the full dataset (path relative to project root, since this
    #    script is run as `python src/linear_regression.py` from project root)
    X, y = load_dataset("data/dataset.csv")

    # b. Split into train/test sets using our own manual split (Stage 2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, seed=42
    )

    # c. Train our from-scratch model using gradient descent (Stage 5),
    #    using ONLY X_train and y_train
    w_final, b_final, loss_history = gradient_descent(
        X_train, y_train, learning_rate=0.01, n_iterations=1000, verbose=False
    )

    # d. Predict on both training and test sets using our hypothesis (Stage 3)
    y_train_pred = compute_hypothesis(X_train, w_final, b_final)
    y_test_pred = compute_hypothesis(X_test, w_final, b_final)

    # e. Evaluate using our own MSE implementation (Stage 4)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    # f. Print summary
    print("\n--- Stage 6: Train/Test Pipeline ---")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Learned w: {w_final[0]:.4f}")
    print(f"Learned b: {b_final:.4f}")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

    # Stage 8: additional evaluation metrics (RMSE, R^2), reusing the
    # predictions already computed above - no retraining
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    train_r2 = r_squared(y_train, y_train_pred)
    test_r2 = r_squared(y_test, y_test_pred)

    print("\n--- Stage 8: Model Evaluation ---")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Training R\u00b2: {train_r2:.4f}")
    print(f"Test R\u00b2: {test_r2:.4f}")

    # Stage 7: visualize the learned regression line against the full dataset
    plot_regression_line(X, y, w_final, b_final)

    # Stage 9: visualize gradient descent convergence using the SAME
    # loss_history produced during training above (no retraining)
    plot_loss_curve(loss_history)


if __name__ == "__main__":
    print("--- Testing Stage 3: hypothesis function ---")

    # Tiny sanity-check example: 3 samples, 1 feature
    X_sample = np.array([[1.0], [2.0], [3.0]])
    w_sample = np.array([5.0])
    b_sample = 50.0

    y_hat = compute_hypothesis(X_sample, w_sample, b_sample)
    print(f"X:\n{X_sample}")
    print(f"w: {w_sample}, b: {b_sample}")
    print(f"y_hat: {y_hat}")
    # Expected: w*x + b for each row -> [55, 60, 65]

    print("\n--- Testing Stage 5: gradient descent ---")

    X_toy = np.array([[1.0], [2.0], [3.0]])
    y_toy = np.array([55.0, 60.0, 65.0])

    w_learned, b_learned, loss_history = gradient_descent(
        X_toy, y_toy, learning_rate=0.01, n_iterations=1000, verbose=True
    )

    final_y_hat = compute_hypothesis(X_toy, w_learned, b_learned)
    final_mse = mean_squared_error(y_toy, final_y_hat)

    print(f"\nLearned w: {w_learned}")
    print(f"Learned b: {b_learned:.4f}")
    print(f"Final MSE: {final_mse:.6f}")
    print(f"Loss at iteration 0:    {loss_history[0]:.4f}")
    print(f"Loss at final iteration: {loss_history[-1]:.4f}")

    # Stage 6: run the end-to-end pipeline on the real dataset
    run_stage6_pipeline()