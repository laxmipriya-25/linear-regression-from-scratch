"""
scaling.py

STAGE 12: Feature scaling (standardization), implemented from scratch
with NumPy only - no sklearn.StandardScaler.

Standardization formula:
    z = (x - mean) / std

This rescales each feature to have mean 0 and standard deviation 1,
which matters a great deal for gradient descent (see feature_scaling.py
for the demonstration).
"""

import numpy as np


def standardize_features(X_train, X_test):
    """
    Standardize X_train and X_test using statistics computed ONLY from
    X_train.

    WHY training-set-only statistics:
    If we computed mean/std from the full dataset (train + test combined),
    information about the test set's distribution would leak into the
    scaling parameters used during training. This is called "data
    leakage" - it makes the model's test performance look artificially
    better than it would on truly unseen data. By computing mean/std
    from X_train alone and reusing those same numbers to transform
    X_test, we simulate the real-world situation: at prediction time,
    we only ever know the training distribution, never the test set.

    Parameters
    ----------
    X_train : numpy.ndarray, shape (n_train_samples, n_features)
        Training feature matrix.
    X_test : numpy.ndarray, shape (n_test_samples, n_features)
        Test feature matrix.

    Returns
    -------
    X_train_scaled : numpy.ndarray, shape (n_train_samples, n_features)
    X_test_scaled : numpy.ndarray, shape (n_test_samples, n_features)
    mean : numpy.ndarray, shape (n_features,)
        Per-feature mean, computed from X_train only.
    std : numpy.ndarray, shape (n_features,)
        Per-feature standard deviation, computed from X_train only.
    """
    if not isinstance(X_train, np.ndarray) or X_train.ndim != 2:
        raise ValueError(f"X_train must be a 2D numpy array, got {type(X_train)} "
                          f"with shape {getattr(X_train, 'shape', None)}.")
    if not isinstance(X_test, np.ndarray) or X_test.ndim != 2:
        raise ValueError(f"X_test must be a 2D numpy array, got {type(X_test)} "
                          f"with shape {getattr(X_test, 'shape', None)}.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"X_train and X_test must have the same number of features: "
            f"{X_train.shape[1]} vs {X_test.shape[1]}."
        )

    # Statistics computed from X_train ONLY - never from X_test
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # A feature with std == 0 is constant across all training samples;
    # dividing by zero would produce inf/NaN, so we fail loudly instead.
    if np.any(std == 0):
        constant_feature_indices = np.where(std == 0)[0]
        raise ValueError(
            f"Feature(s) at index/indices {constant_feature_indices.tolist()} "
            f"have zero standard deviation in X_train (constant values) and "
            f"cannot be standardized (division by zero)."
        )

    # Apply the SAME training statistics to both sets
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled, mean, std


if __name__ == "__main__":
    print("--- Testing Stage 12: standardize_features ---")

    # Small demo: 2 features on very different scales
    X_train_demo = np.array([
        [1.0, 100.0],
        [2.0, 200.0],
        [3.0, 300.0],
        [4.0, 400.0],
    ])
    X_test_demo = np.array([
        [2.5, 250.0],
        [5.0, 500.0],
    ])

    X_train_scaled, X_test_scaled, mean, std = standardize_features(
        X_train_demo, X_test_demo
    )

    print(f"X_train (raw):\n{X_train_demo}")
    print(f"Training mean: {mean}")
    print(f"Training std:  {std}")
    print(f"\nX_train_scaled:\n{X_train_scaled}")
    print(f"X_test_scaled:\n{X_test_scaled}")

    # Sanity check: scaled training data should have mean ~0, std ~1
    print(f"\nX_train_scaled mean (should be ~0): {X_train_scaled.mean(axis=0)}")
    print(f"X_train_scaled std (should be ~1):  {X_train_scaled.std(axis=0)}")

    # Demonstrate the zero-std error case
    print("\n--- Testing zero-std error handling ---")
    try:
        X_constant = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        standardize_features(X_constant, X_constant)
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")