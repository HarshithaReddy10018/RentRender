import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data.load_data import load_data


def prepare_data(df):

    target_col = "price"

    numerical_features = [
        "bathrooms",
        "bedrooms",
        "square_feet",
        "latitude",
        "longitude"
    ]

    numerical_features = [
        col
        for col in numerical_features
        if col in df.columns
    ]

    data = df[
        numerical_features + [target_col]
    ].copy()

    # Convert values to numeric
    for col in numerical_features + [target_col]:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # Remove missing values
    data = data.dropna()

    # Remove invalid prices
    data = data[
        data[target_col] > 0
    ]

    X = data[
        numerical_features
    ]

    y = data[
        target_col
    ]

    return X, y


def run_gradient_boosting():

    print("=" * 50)
    print("RENT RADAR - GRADIENT BOOSTING REGRESSION")
    print("=" * 50)

    # Load dataset
    df = load_data()

    print("\nOriginal Dataset Shape:")
    print(df.shape)

    # Prepare data
    X, y = prepare_data(df)

    print("\nPrepared Dataset Shape:")
    print(X.shape)

    print("\nFeatures:")
    print(X.columns.tolist())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTraining Samples:")
    print(len(X_train))

    print("\nTesting Samples:")
    print(len(X_test))

    # Create Gradient Boosting model
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    print("\nTraining Gradient Boosting...")

    # Train
    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # Prediction
    y_pred = model.predict(
        X_test
    )

    # Metrics
    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    # Feature importance
    feature_importance = {}

    for feature, importance in zip(
        X.columns,
        model.feature_importances_
    ):

        feature_importance[feature] = round(
            float(importance),
            4
        )

    # Results
    results = {
        "model": "Gradient Boosting Regression",
        "mae": round(float(mae), 2),
        "mse": round(float(mse), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
        "feature_importance": feature_importance
    }

    return results


if __name__ == "__main__":

    results = run_gradient_boosting()

    print("\n" + "=" * 50)
    print("GRADIENT BOOSTING RESULTS")
    print("=" * 50)

    print("\nMAE:")
    print(results["mae"])

    print("\nMSE:")
    print(results["mse"])

    print("\nRMSE:")
    print(results["rmse"])

    print("\nR2 Score:")
    print(results["r2"])

    print("\nFeature Importance:")
    print("-" * 40)

    for feature, importance in results[
        "feature_importance"
    ].items():

        print(
            f"{feature}: {importance}"
        )

    print(
        "\nGradient Boosting completed successfully."
    )