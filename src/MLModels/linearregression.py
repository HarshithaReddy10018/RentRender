import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data.load_data import load_data


def prepare_data(df):

    target_col = "price"

    # Remove unnecessary columns
    drop_cols = [
        col for col in [
            "id",
            "title",
            "body",
            "amenities",
            "price_display",
            "address"
        ]
        if col in df.columns
    ]

    # Keep useful numerical features
    numerical_features = [
        "bathrooms",
        "bedrooms",
        "square_feet",
        "latitude",
        "longitude"
    ]

    numerical_features = [
        col for col in numerical_features
        if col in df.columns
    ]

    data = df[numerical_features + [target_col]].copy()

    # Convert values to numeric
    for col in numerical_features + [target_col]:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # Remove missing values
    data = data.dropna()

    # Remove invalid prices
    data = data[data[target_col] > 0]

    X = data[numerical_features]

    y = data[target_col]

    return X, y


def run_linear_regression():

    # Load dataset
    df = load_data()

    print("=" * 50)
    print("RENT RADAR - LINEAR REGRESSION")
    print("=" * 50)

    print("\nOriginal Dataset Shape:")
    print(df.shape)

    # Prepare data
    X, y = prepare_data(df)

    print("\nPrepared Dataset Shape:")
    print(X.shape)

    print("\nFeatures:")
    print(X.columns.tolist())

    # Split dataset
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

    # Create model
    model = LinearRegression()

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
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

    # Coefficients
    coefficients = {}

    for feature, coefficient in zip(
        X_train.columns,
        model.coef_
    ):
        coefficients[feature] = round(
            float(coefficient),
            2
        )

    results = {
        "model": "Linear Regression",
        "intercept": round(
            float(model.intercept_),
            2
        ),
        "coefficients": coefficients,
        "mae": round(
            float(mae),
            2
        ),
        "mse": round(
            float(mse),
            2
        ),
        "rmse": round(
            float(rmse),
            2
        ),
        "r2": round(
            float(r2),
            4
        )
    }

    return results


if __name__ == "__main__":

    results = run_linear_regression()

    print("\n" + "=" * 50)
    print("LINEAR REGRESSION RESULTS")
    print("=" * 50)

    print("\nIntercept:")
    print(results["intercept"])

    print("\nCoefficients: ")

    for feature, value in results["coefficients"].items():
        print(
            feature,
            ":",
            value
        )

    print("\nMAE:")
    print(results["mae"])

    print("\nMSE:")
    print(results["mse"])

    print("\nRMSE:")
    print(results["rmse"])

    print("\nR2 Score:")
    print(results["r2"])

    print("\nLinear Regression completed successfully.")