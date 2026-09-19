import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.data.load_data import load_data


def prepare_data(df):

    target_col = "price"

    # --------------------------------------------------
    # Numerical Features
    # --------------------------------------------------

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


    # --------------------------------------------------
    # Categorical Features
    # --------------------------------------------------

    categorical_features = [
        "category",
        "has_photo",
        "pets_allowed",
        "price_type",
        "cityname",
        "state",
        "source"
    ]

    categorical_features = [
        col for col in categorical_features
        if col in df.columns
    ]


    # --------------------------------------------------
    # Select Columns
    # --------------------------------------------------

    data = df[
        numerical_features +
        categorical_features +
        [target_col]
    ].copy()


    # --------------------------------------------------
    # Convert Numerical Columns
    # --------------------------------------------------

    for col in numerical_features + [target_col]:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )


    # --------------------------------------------------
    # Remove Missing Target Values
    # --------------------------------------------------

    data = data.dropna(
        subset=[target_col]
    )


    # --------------------------------------------------
    # Remove Invalid Prices
    # --------------------------------------------------

    data = data[
        data[target_col] > 0
    ]


    # --------------------------------------------------
    # Fill Missing Values
    # --------------------------------------------------

    for col in numerical_features:

        data[col] = data[col].fillna(
            data[col].median()
        )


    for col in categorical_features:

        data[col] = data[col].fillna(
            "Unknown"
        )


    # --------------------------------------------------
    # Create Price Category
    # --------------------------------------------------

    median_price = data[target_col].median()

    data["price_category"] = (
        data[target_col] > median_price
    ).astype(int)


    # 0 = Low Price
    # 1 = High Price

    X = data[
        numerical_features +
        categorical_features
    ]

    y = data["price_category"]


    return (
        X,
        y,
        numerical_features,
        categorical_features,
        median_price
    )


def run_logistic_regression():

    print("=" * 50)
    print("RENT RADAR - LOGISTIC REGRESSION")
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

    (
        X,
        y,
        numerical_features,
        categorical_features,
        median_price
    ) = prepare_data(df)


    print("\nPrepared Dataset Shape:")
    print(X.shape)


    print("\nNumerical Features:")
    print(numerical_features)


    print("\nCategorical Features:")
    print(categorical_features)


    print("\nMedian Rental Price:")
    print(median_price)


    # --------------------------------------------------
    # Split Dataset
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    print("\nTraining Samples:")
    print(len(X_train))


    print("\nTesting Samples:")
    print(len(X_test))


    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numerical",
                StandardScaler(),
                numerical_features
            ),

            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )


    # --------------------------------------------------
    # Logistic Regression Model
    # --------------------------------------------------

    model = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )


    # --------------------------------------------------
    # Train Model
    # --------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Low Price",
            "High Price"
        ],
        zero_division=0
    )


    results = {

        "model": "Logistic Regression",

        "median_price": round(
            float(median_price),
            2
        ),

        "accuracy": round(
            float(accuracy),
            4
        ),

        "precision": round(
            float(precision),
            4
        ),

        "recall": round(
            float(recall),
            4
        ),

        "f1_score": round(
            float(f1),
            4
        ),

        "confusion_matrix": cm.tolist(),

        "classification_report": report
    }


    return results


if __name__ == "__main__":

    results = run_logistic_regression()


    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 50)


    print("\nMedian Price:")
    print(
        results["median_price"]
    )


    print("\nAccuracy: ")
    print(
        results["accuracy"]
    )


    print("\nPrecision:")
    print(
        results["precision"]
    )


    print("\nRecall:")
    print(
        results["recall"]
    )


    print("\nF1 Score:")
    print(
        results["f1_score"]
    )


    print("\nConfusion Matrix:")

    print(
        results["confusion_matrix"]
    )


    print("\nClassification Report:")

    print(
        results["classification_report"]
    )


    print(
        "\nLogistic Regression completed successfully."
    )