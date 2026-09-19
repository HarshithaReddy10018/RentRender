import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_data


def find_optimal_k(X):

    wcss = []

    for k in range(1, 11):

        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42
        )

        model.fit(X)

        wcss.append(
            model.inertia_
        )

    print("\nWCSS Value:")

    for k, value in zip(range(1, 11), wcss):

        print(
            "K =",
            k,
            "WCSS =",
            value
        )

    plt.figure(figsize=(8, 6))

    plt.plot(
        range(1, 11),
        wcss,
        marker="o"
    )

    plt.xlabel("Number of clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method For Optimal K")

    plt.xticks(range(1, 11))

    plt.grid(True)

    plt.show()

    return wcss


def create_model(k):

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42
    )

    return model


def train_model(model, X):

    labels = model.fit_predict(X)

    print("\nK-Means trained successfully!")

    return model, labels


def evaluate_model(model, X, labels):

    print("\nInertia (WCSS):")
    print(model.inertia_)

    print("\nIterations to Converge:")
    print(model.n_iter_)

    silhouette = silhouette_score(
        X,
        labels
    )

    print("\nSilhouette score:")
    print(silhouette)

    return silhouette


def display_clusters(X_df, labels, model):

    plt.figure(figsize=(8, 6))

    X_vals = (
        X_df.values
        if isinstance(X_df, pd.DataFrame)
        else X_df
    )

    plt.scatter(
        X_vals[:, 0],
        X_vals[:, 1],
        c=labels,
        cmap="viridis",
        s=30
    )

    plt.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        marker="X",
        color="red",
        s=200,
        label="Centroids"
    )

    plt.title("K-Means Clustering")

    if isinstance(X_df, pd.DataFrame):

        plt.xlabel(
            X_df.columns[0]
        )

        plt.ylabel(
            X_df.columns[1]
        )

    else:

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")

    plt.legend()

    plt.show()


def main():

    # 1. Load Data

    df = load_data()

    print("Original Dataset Shape:")
    print(df.shape)


    # 2. Select features for clustering

    clustering_features = [
        "price",
        "bedrooms",
        "bathrooms",
        "square_feet"
    ]

    clustering_features = [
        col
        for col in clustering_features
        if col in df.columns
    ]

    features_df = df[
        clustering_features
    ].copy()


    # 3. Convert features to numeric

    for col in clustering_features:

        features_df[col] = pd.to_numeric(
            features_df[col],
            errors="coerce"
        )


    # 4. Remove missing values

    features_df = features_df.dropna()


    # 5. Remove invalid prices

    features_df = features_df[
        features_df["price"] > 0
    ]


    print("\nK-Means Dataset Shape:")
    print(features_df.shape)

    print("\nK-Means Features:")
    print(features_df.columns.tolist())


    # 6. Standardize features

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        features_df
    )


    # 7. Find Optimal K

    find_optimal_k(
        X_scaled
    )


    # 8. Select K

    # Change this after checking
    # the elbow graph

    optimal_k = 3


    # 9. Create K-Means model

    model = create_model(
        optimal_k
    )


    # 10. Train model

    model, labels = train_model(
        model,
        X_scaled
    )


    # 11. Evaluate model

    evaluate_model(
        model,
        X_scaled,
        labels
    )


    # 12. Display clusters

    display_clusters(
        features_df,
        labels,
        model
    )


if __name__ == "__main__":

    main()