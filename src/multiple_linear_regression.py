"""
multiple_linear_regression.py

STAGE 11: Multiple Linear Regression From Scratch.

Extends the project from single-feature to multi-feature linear
regression, as a SEPARATE, self-contained module. This does NOT replace
or modify src/linear_regression.py, which remains the original
single-feature implementation.

Model:
    y_hat = X @ w + b

where X has multiple columns (features) instead of just one.

Reused from elsewhere in the project (not reimplemented):
    - train_test_split()  from src/utils.py   (already feature-count agnostic)
    - mean_squared_error(), root_mean_squared_error(), r_squared()
      from src/metrics.py

NOT reused (implemented fresh here, as this stage's own self-contained
hypothesis/gradient/training logic, per the Stage 11 requirements):
    - compute_hypothesis()
    - compute_gradients()
    - gradient_descent()
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import train_test_split
from metrics import mean_squared_error, root_mean_squared_error, r_squared


FEATURE_NAMES = [
    "hours_studied",
    "attendance",
    "assignments_completed",
    "previous_score",
]


def generate_multiple_dataset(
    n_samples=300,
    noise_std=6.0,
    seed=42,
    save_path="data/multiple_regression_dataset.csv",
):
    """
    Generate a synthetic multi-feature student-performance dataset and
    save it to CSV.

    Simulates:
        exam_score = 3.5*hours_studied + 0.3*attendance
                     + 2.0*assignments_completed + 0.4*previous_score
                     + noise

    Parameters
    ----------
    n_samples : int
        Number of student records to generate.
    noise_std : float
        Standard deviation of Gaussian noise added to the target.
    seed : int
        Random seed for reproducibility.
    save_path : str
        Where to save the generated CSV.

    Returns
    -------
    df : pandas.DataFrame
        The generated dataset.
    """
    rng = np.random.default_rng(seed)

    hours_studied = rng.uniform(0, 10, n_samples)
    attendance = rng.uniform(50, 100, n_samples)
    assignments_completed = rng.uniform(0, 10, n_samples)
    previous_score = rng.uniform(30, 100, n_samples)

    noise = rng.normal(0, noise_std, n_samples)

    exam_score = (
        3.5 * hours_studied
        + 0.3 * attendance
        + 2.0 * assignments_completed
        + 0.4 * previous_score
        + noise
    )
    exam_score = np.clip(exam_score, 0, 100)

    df = pd.DataFrame({
        "hours_studied": hours_studied,
        "attendance": attendance,
        "assignments_completed": assignments_completed,
        "previous_score": previous_score,
        "exam_score": exam_score,
    })

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_csv(save_path, index=False)

    return df


def load_multiple_dataset(path="data/multiple_regression_dataset.csv"):
    """
    Load the multi-feature dataset and split into X (4 columns) and y.

    Parameters
    ----------
    path : str
        Path to the dataset CSV file.

    Returns
    -------
    X : numpy.ndarray, shape (n_samples, 4)
        Feature matrix: hours_studied, attendance, assignments_completed,
        previous_score.
    y : numpy.ndarray, shape (n_samples,)
        Target vector: exam_score.
    """
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    X = df[FEATURE_NAMES].to_numpy()
    y = df["exam_score"].to_numpy()

    return X, y


def compute_hypothesis(X, w, b):
    """
    Compute predictions: y_hat = X @ w + b

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
    w : numpy.ndarray, shape (n_features,)
    b : float

    Returns
    -------
    y_hat : numpy.ndarray, shape (n_samples,)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}.")
    if X.shape[1] != w.shape[0]:
        raise ValueError(
            f"Feature/weight mismatch: X has {X.shape[1]} features, "
            f"w has {w.shape[0]} weights."
        )

    y_hat = X @ w + b
    return y_hat


def compute_gradients(X, y, y_hat):
    """
    Compute the gradients of MSE with respect to w and b.

        dw = -(2/n) * X.T @ error
        db = -(2/n) * sum(error)

    where error = y - y_hat.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
    y : numpy.ndarray, shape (n_samples,)
    y_hat : numpy.ndarray, shape (n_samples,)

    Returns
    -------
    dw : numpy.ndarray, shape (n_features,)
    db : float
    """
    n = X.shape[0]
    error = y - y_hat

    dw = -(2 / n) * (X.T @ error)
    db = -(2 / n) * np.sum(error)

    return dw, db


def gradient_descent(X, y, learning_rate=0.001, n_iterations=2000, verbose=False):
    """
    Train multiple linear regression parameters (w, b) using batch
    gradient descent. Works for ANY number of features.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
    y : numpy.ndarray, shape (n_samples,)
    learning_rate : float
    n_iterations : int
    verbose : bool
        If True, print loss at sensible intervals.

    Returns
    -------
    w : numpy.ndarray, shape (n_features,)
    b : float
    loss_history : list of float
    """
    n_features = X.shape[1]

    w = np.zeros(n_features)
    b = 0.0

    loss_history = []
    print_every = max(n_iterations // 10, 1)

    for iteration in range(n_iterations):
        y_hat = compute_hypothesis(X, w, b)
        loss = mean_squared_error(y, y_hat)
        loss_history.append(loss)

        dw, db = compute_gradients(X, y, y_hat)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if verbose and (iteration % print_every == 0 or iteration == n_iterations - 1):
            print(f"Iteration {iteration:5d} | Loss: {loss:.4f}")

    return w, b, loss_history


def plot_multiple_loss_curve(
    loss_history, save_path="plots/multiple_regression_loss_curve.png"
):
    """
    Plot MSE loss vs. iteration for the multiple regression training run.

    Parameters
    ----------
    loss_history : list of float
    save_path : str
    """
    plots_dir = os.path.dirname(save_path)
    if plots_dir and not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    iterations = np.arange(len(loss_history))

    plt.figure(figsize=(8, 6))
    plt.plot(iterations, loss_history, color="green", linewidth=2)

    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Multiple Linear Regression - Gradient Descent Convergence")
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()

    print(f"Plot saved to: {save_path}")


def run_multiple_regression_pipeline():
    """
    Full Stage 11 pipeline: generate data -> load -> split -> train ->
    evaluate -> plot -> example prediction.
    """
    # 1. Generate the dataset (deterministic via seed=42)
    generate_multiple_dataset(n_samples=300, seed=42)

    # 2. Load it back from disk
    X, y = load_multiple_dataset("data/multiple_regression_dataset.csv")

    n_samples, n_features = X.shape
    print(f"Number of samples: {n_samples}")
    print(f"Number of features: {n_features}")

    # 3. Split using the existing (feature-count agnostic) utility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # 4. Standardize features before training.
    #
    #    WHY: our 4 features have very different scales (hours_studied is
    #    0-10, attendance is 50-100). With raw, unscaled features, the
    #    gradient for large-scale features dominates the update step and
    #    gradient descent overshoots and diverges (loss explodes to NaN)
    #    at learning_rate=0.001. Standardizing each feature to mean 0,
    #    std 1 puts all features on a comparable scale so a single
    #    learning rate works for all of them - this is standard practice
    #    for multi-feature gradient descent, not a change to the
    #    hypothesis/gradient/training math itself.
    #
    #    Statistics are computed from X_train ONLY (never from X_test),
    #    to avoid leaking test-set information into training.
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)

    X_train_scaled = (X_train - feature_mean) / feature_std
    X_test_scaled = (X_test - feature_mean) / feature_std

    # 5. Train via gradient descent, on the SCALED training features
    w_scaled, b_scaled, loss_history = gradient_descent(
        X_train_scaled, y_train, learning_rate=0.001, n_iterations=2000, verbose=True
    )

    # 6. Convert learned weights back to the ORIGINAL (unscaled) feature
    #    units, so coefficients are directly interpretable and so we can
    #    predict on raw feature values without re-scaling them everywhere.
    #    Derivation: since x_scaled = (x - mean) / std,
    #        y_hat = w_scaled . x_scaled + b_scaled
    #              = (w_scaled / std) . x + (b_scaled - sum(w_scaled * mean / std))
    #    so: w_original = w_scaled / std
    #        b_original = b_scaled - sum(w_original * mean)
    w = w_scaled / feature_std
    b = b_scaled - np.sum(w * feature_mean)

    print("\nLearned coefficients:")
    for name, weight in zip(FEATURE_NAMES, w):
        print(f"{name}: {weight:.4f}")
    print(f"\nLearned intercept:\n{b:.4f}")

    # 7. Evaluate (reusing existing metrics.py functions, not reimplemented)
    #    Predictions use the UNSCALED w, b on the RAW (unscaled) X_train/X_test
    y_train_pred = compute_hypothesis(X_train, w, b)
    y_test_pred = compute_hypothesis(X_test, w, b)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    train_r2 = r_squared(y_train, y_train_pred)
    test_r2 = r_squared(y_test, y_test_pred)

    print("\n--- Stage 11: Model Evaluation ---")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Training R\u00b2: {train_r2:.4f}")
    print(f"Test R\u00b2: {test_r2:.4f}")

    # 8. Convergence plot, using the REAL loss_history from training above
    plot_multiple_loss_curve(loss_history)

    # 9. Example prediction (raw feature values, using unscaled w, b)
    example = np.array([[6.0, 85.0, 7.0, 75.0]])  # one sample, 4 features
    predicted_score = compute_hypothesis(example, w, b)[0]

    print("\nExample Prediction")
    print("------------------")
    print(f"Hours studied: {example[0, 0]}")
    print(f"Attendance: {example[0, 1]}")
    print(f"Assignments completed: {example[0, 2]}")
    print(f"Previous score: {example[0, 3]}")
    print(f"Predicted exam score: {predicted_score:.4f}")


if __name__ == "__main__":
    run_multiple_regression_pipeline()