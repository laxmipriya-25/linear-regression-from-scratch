import sys
from pathlib import Path

# Add the src folder to Python's import path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from utils import load_dataset, train_test_split
from linear_regression import (
    gradient_descent,
    compute_hypothesis,
    plot_regression_line,
    plot_loss_curve,
)
from metrics import (
    mean_squared_error,
    root_mean_squared_error,
    r_squared,
)
from sklearn_comparison import run_sklearn_comparison
def main():
    print("=" * 60)
    print("LINEAR REGRESSION FROM SCRATCH")
    print("=" * 60)

    print("\n[1] Loading dataset...")
    X, y = load_dataset("data/dataset.csv")
    print(f"Dataset shape: {X.shape}")
    print(f"Target shape : {y.shape}")

    print("\n[2] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, seed=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    print("\n[3] Training Linear Regression from scratch...")
    w, b, loss_history = gradient_descent(
        X_train,
        y_train,
        learning_rate=0.01,
        n_iterations=1000,
        verbose=False,
    )

    print(f"Learned weight    : {w[0]:.4f}")
    print(f"Learned intercept : {b:.4f}")

    print("\n[4] Generating predictions...")
    y_train_pred = compute_hypothesis(X_train, w, b)
    y_test_pred = compute_hypothesis(X_test, w, b)

    print("\n[5] Model Evaluation")
    print("-" * 40)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)

    train_r2 = r_squared(y_train, y_train_pred)
    test_r2 = r_squared(y_test, y_test_pred)

    print(f"Training MSE  : {train_mse:.4f}")
    print(f"Testing MSE   : {test_mse:.4f}")
    print(f"Training RMSE : {train_rmse:.4f}")
    print(f"Testing RMSE  : {test_rmse:.4f}")
    print(f"Training R²   : {train_r2:.4f}")
    print(f"Testing R²    : {test_r2:.4f}")

    print("\n[6] Generating plots...")
    plot_regression_line(X_test, y_test, w, b)
    plot_loss_curve(loss_history)

    print("\n[7] Comparing with Scikit-Learn...")
    run_sklearn_comparison()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nGenerated plots are available in the 'plots/' folder.")


if __name__ == "__main__":
    main()