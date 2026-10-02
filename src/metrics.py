"""
metrics.py

Manually implemented evaluation metrics (no sklearn).

STAGE 4: Mean Squared Error (MSE) - implemented below
STAGE 8: Root Mean Squared Error (RMSE), R-squared (R^2) - implemented below
"""

import numpy as np


def mean_squared_error(y, y_hat):
    """
    Compute Mean Squared Error: MSE = (1/n) * sum((y - y_hat)^2)

    Parameters
    ----------
    y : numpy.ndarray, shape (n_samples,)
        Actual (ground-truth) values.
    y_hat : numpy.ndarray, shape (n_samples,)
        Predicted values.

    Returns
    -------
    mse : float
        Mean squared error between y and y_hat.
    """
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)

    if y.shape != y_hat.shape:
        raise ValueError(
            f"Shape mismatch: y has shape {y.shape}, y_hat has shape {y_hat.shape}."
        )
    if y.size == 0:
        raise ValueError("Cannot compute MSE on empty arrays.")

    errors = y - y_hat
    squared_errors = errors ** 2
    mse = np.mean(squared_errors)
    return mse


def root_mean_squared_error(y_true, y_pred):
    """
    Compute Root Mean Squared Error: RMSE = sqrt(MSE)

    RMSE is in the same units as the target variable (unlike MSE, which
    is in squared units), making it easier to interpret directly.

    Parameters
    ----------
    y_true : numpy.ndarray, shape (n_samples,)
        Actual (ground-truth) values.
    y_pred : numpy.ndarray, shape (n_samples,)
        Predicted values.

    Returns
    -------
    rmse : float
        Root mean squared error between y_true and y_pred.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_pred has shape {y_pred.shape}."
        )

    # Reuse the existing MSE function, then take the square root
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse


def r_squared(y_true, y_pred):
    """
    Compute the R-squared (coefficient of determination):

        R^2 = 1 - (SS_res / SS_tot)

    where:
        SS_res = sum((y_true - y_pred)^2)   <- residual sum of squares
        SS_tot = sum((y_true - mean(y_true))^2)  <- total sum of squares

    R^2 measures the proportion of variance in y_true that is explained
    by the model. R^2 = 1 means a perfect fit; R^2 = 0 means the model
    does no better than always predicting the mean of y_true.

    Parameters
    ----------
    y_true : numpy.ndarray, shape (n_samples,)
        Actual (ground-truth) values.
    y_pred : numpy.ndarray, shape (n_samples,)
        Predicted values.

    Returns
    -------
    r2 : float
        R-squared score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_pred has shape {y_pred.shape}."
        )
    if y_true.size == 0:
        raise ValueError("Cannot compute R^2 on empty arrays.")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        raise ValueError(
            "SS_tot is 0 (all y_true values are identical), so R^2 is "
            "undefined - there is no variance in y_true to explain."
        )

    r2 = 1 - (ss_res / ss_tot)
    return r2


if __name__ == "__main__":
    print("--- Testing Stage 4: MSE ---")

    y = np.array([55, 60, 65])
    y_hat = np.array([55, 62, 63])

    mse = mean_squared_error(y, y_hat)
    print(f"y:      {y}")
    print(f"y_hat:  {y_hat}")
    print(f"MSE:    {mse}")

    print("\n--- Testing Stage 8: RMSE and R^2 ---")

    y_true = np.array([55.0, 60.0, 65.0])
    y_pred = np.array([55.0, 62.0, 63.0])

    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r_squared(y_true, y_pred)

    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"MSE:    {mean_squared_error(y_true, y_pred)}")
    print(f"RMSE:   {rmse}")
    print(f"R^2:    {r2}")