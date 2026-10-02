"""
utils.py

Utility functions for the Linear Regression From Scratch project.

STAGE 1: Dataset generation (implemented below)
Future stages will add: train/test split helpers, and other
preprocessing utilities as needed.
"""

import numpy as np
import pandas as pd


def generate_dataset(n_samples=200, w_true=5.0, b_true=50.0, noise_std=8.0, seed=42):
    """
    Generate a synthetic single-feature linear regression dataset.

    Simulates: exam_score = w_true * hours_studied + b_true + noise

    Parameters
    ----------
    n_samples : int
        Number of data points to generate.
    w_true : float
        The true slope (points per hour studied) used to generate the data.
    b_true : float
        The true intercept (baseline score with 0 hours studied).
    noise_std : float
        Standard deviation of the Gaussian noise added to y.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns 'hours_studied' and 'exam_score'.
    """
    rng = np.random.default_rng(seed)

    # Hours studied: uniformly spread between 0 and 10
    hours_studied = rng.uniform(0, 10, n_samples)

    # Gaussian noise to simulate real-world variability
    noise = rng.normal(0, noise_std, n_samples)

    # True linear relationship + noise
    exam_score = w_true * hours_studied + b_true + noise

    # Clip scores to a realistic 0-100 range
    exam_score = np.clip(exam_score, 0, 100)

    df = pd.DataFrame({
        "hours_studied": hours_studied,
        "exam_score": exam_score
    })

    return df


def load_dataset(path="../data/dataset.csv"):
    """
    Load the dataset from a CSV file and split into features (X) and target (y).

    Parameters
    ----------
    path : str
        Path to the dataset CSV file.

    Returns
    -------
    X : numpy.ndarray, shape (n_samples, 1)
        Feature matrix (hours_studied).
    y : numpy.ndarray, shape (n_samples,)
        Target vector (exam_score).
    """
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    # Features as a 2D column vector (n_samples, 1) - standard shape for X
    X = df[["hours_studied"]].to_numpy()
    # Target as a 1D vector (n_samples,)
    y = df["exam_score"].to_numpy()

    return X, y


def train_test_split(X, y, test_size=0.2, seed=42):
    """
    Manually split X and y into training and test sets.

    Implemented from scratch (no sklearn) using a random permutation
    of indices, so the split is shuffled rather than sequential.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : numpy.ndarray, shape (n_samples,)
        Target vector.
    test_size : float
        Fraction of samples to allocate to the test set (0 < test_size < 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy.ndarray
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have incompatible shapes: {X.shape[0]} vs {y.shape[0]} samples."
        )
    if X.shape[0] == 0:
        raise ValueError("Cannot split an empty dataset.")

    n_samples = X.shape[0]
    n_test = round(test_size * n_samples)

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(n_samples)

    test_indices = shuffled_indices[:n_test]
    train_indices = shuffled_indices[n_test:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("--- Testing Stage 2: preprocessing ---")

    X, y = load_dataset("data/dataset.csv")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")