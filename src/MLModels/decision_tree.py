import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data.load_data import load_data


def prepare_data(df):

    target_col = "price"

    # Numerical features
    numerical_features = [
        "bathrooms",
        "bedrooms",
        "square_feet",
        "latitude",
        "longitude"
    ]

    # Keep only columns that exist
    numerical_features = [
        col for col in numerical_features
        if col in df.columns
    ]

    # Select required columns
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

    X = data[numerical_features]

    y = data[target_col]

    return X, y


def run_decision_tree():

    print("=" * 50)
    print("RENT RADAR - DECISION TREE REGRESSION")
    print("=" * 50)

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    df = load_data()

    print("\nOriginal Dataset Shape:")
    print(df.shape)

    # --------------------------------------------------
    # Prepare Data
    # --------------------------------------------------

    X, y = prepare_data(df)

    print("\nPrepared Dataset Shape:")
    print(X.shape)

    print("\nFeatures:")
    print(X.columns.tolist())

    # --------------------------------------------------
    # Train Test Split
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Decision Tree Model
    # --------------------------------------------------

    model = DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    # --------------------------------------------------
    # Train Model
    # --------------------------------------------------

    print("\nTraining Decision Tree...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        y_pred
    )

    # --------------------------------------------------
    # Feature Importance
    # --------------------------------------------------

    feature_importance = {}

    for feature, importance in zip(
        X.columns,
        model.feature_importances_
    ):

        feature_importance[feature] = round(
            float(importance),
            4
        )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    results = {

        "model": "Decision Tree Regression",

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
        ),

        "feature_importance":
            feature_importance
    }

    return results


if __name__ == "__main__":

    results = run_decision_tree()

    print("\n" + "=" * 50)
    print("DECISION TREE RESULTS")
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
        "\nDecision Tree completed successfully."
    )