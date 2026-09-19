import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_data


def prepare_data(df):

    features = [
        "price",
        "bedrooms",
        "bathrooms",
        "square_feet"
    ]

    features = [
        col for col in features
        if col in df.columns
    ]

    data = df[features].copy()

    for col in features:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna()

    data = data[
        data["price"] > 0
    ]

    return data, features


def run_kmeans():

    print("=" * 50)
    print("RENT RADAR - K-MEANS CLUSTERING")
    print("=" * 50)

    df = load_data()

    print("\nOriginal Dataset Shape:")
    print(df.shape)

    data, features = prepare_data(df)

    print("\nPrepared Dataset Shape:")
    print(data.shape)

    print("\nFeatures:")
    print(features)

    X = data[features]

    # Standardization
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # K-Means
    model = KMeans(
        n_clusters=3,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    print("\nTraining K-Means...")

    model.fit(X_scaled)

    print("Training completed.")

    # Add cluster number
    data["cluster"] = model.labels_

    print("\nCluster Counts:")
    print(data["cluster"].value_counts().sort_index())

    print("\nCluster Centers:")

    centers = scaler.inverse_transform(
        model.cluster_centers_
    )

    centers_df = pd.DataFrame(
        centers,
        columns=features
    )

    centers_df.index.name = "Cluster"

    print(centers_df)

    # Save result
    output_path = (
        r"C:\2nd year odd\ML\PythonProject\data"
        r"\kmeans_clusters.csv"
    )

    data.to_csv(
        output_path,
        index=False
    )

    print("\nClustered data saved to:")
    print(output_path)

    return data, model


if __name__ == "__main__":

    data, model = run_kmeans()

    print("\n" + "=" * 50)
    print("K-MEANS COMPLETED SUCCESSFULLY")
    print("=" * 50)