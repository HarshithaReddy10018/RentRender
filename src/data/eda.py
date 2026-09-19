import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD DATASET
# ============================================================

file_path = r"C:\2nd year odd\ML\PythonProject\data\apartments_for_rent_classified_10K.csv"

df = pd.read_csv(
    file_path,
    sep=";",
    encoding="cp1252"
)


print("========================================")
print("RENTRENDER EXPLORATORY DATA ANALYSIS")
print("========================================")


# ============================================================
# 1. DATASET SHAPE
# ============================================================

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# 2. COLUMN NAMES
# ============================================================

print("\nDataset Columns:")

for i, column in enumerate(df.columns):
    print(i, ":", column)


# ============================================================
# 3. FIRST 5 ROWS
# ============================================================

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 4. DATA TYPES
# ============================================================

print("\nData Types:")
print(df.dtypes)


# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 6. DUPLICATE VALUES
# ============================================================

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 7. CONVERT NUMERICAL COLUMNS
# ============================================================

df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

df["square_feet"] = pd.to_numeric(
    df["square_feet"],
    errors="coerce"
)

df["bedrooms"] = pd.to_numeric(
    df["bedrooms"],
    errors="coerce"
)

df["bathrooms"] = pd.to_numeric(
    df["bathrooms"],
    errors="coerce"
)


# ============================================================
# 8. STATISTICAL SUMMARY
# ============================================================

print("\nStatistical Summary:")

print(
    df[
        [
            "price",
            "square_feet",
            "bedrooms",
            "bathrooms"
        ]
    ].describe()
)


# ============================================================
# 9. PRICE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["price"].dropna(),
    bins=30
)

plt.xlabel("Rent Price")
plt.ylabel("Number of Apartments")
plt.title("Rent Price Distribution")

plt.tight_layout()
plt.savefig(r"C:\2nd year odd\ML\PythonProject\app\static\charts\price_distrubution.png")

plt.show()


# ============================================================
# 10. SQUARE FEET DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["square_feet"].dropna(),
    bins=30
)

plt.xlabel("Square Feet")
plt.ylabel("Number of Apartments")
plt.title("Square Feet Distribution")

plt.tight_layout()

plt.savefig(r"C:\2nd year odd\ML\PythonProject\app\static\charts\Square_feet_distribution.png")
plt.show()


# ============================================================
# 11. BEDROOM DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

df["bedrooms"].value_counts().sort_index().plot(
    kind="bar"
)

plt.xlabel("Number of Bedrooms")
plt.ylabel("Number of Apartments")
plt.title("Apartments by Number of Bedrooms")

plt.tight_layout()
plt.savefig(r"C:\2nd year odd\ML\PythonProject\app\static\charts\bedrooms_distribution.png")
plt.show()


# ============================================================
# 12. BATHROOM DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

df["bathrooms"].value_counts().sort_index().plot(
    kind="bar"
)

plt.xlabel("Number of Bathrooms")
plt.ylabel("Number of Apartments")
plt.title("Apartments by Number of Bathrooms")

plt.tight_layout()
plt.savefig(r"C:\2nd year odd\ML\PythonProject\app\static\charts\bathrooms_distribution.")
plt.show()


# ============================================================
# 13. PRICE VS SQUARE FEET
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["square_feet"],
    df["price"],
    alpha=0.5
)

plt.xlabel("Square Feet")
plt.ylabel("Rent Price")
plt.title("Rent Price vs Square Feet")


plt.tight_layout()
plt.savefig(r"C:\2nd year odd\ML\PythonProject\app\static\charts\price_vs_square_feet_distribution.png")
plt.show()


# ============================================================
# 14. PRICE VS BEDROOMS
# ============================================================

plt.figure(figsize=(10, 6))

df.boxplot(
    column="price",
    by="bedrooms"
)

plt.xlabel("Bedrooms")
plt.ylabel("Rent Price")
plt.title("Rent Price vs Bedrooms")

plt.suptitle("")

plt.tight_layout()
plt.savefig(r"C:\2nd year odd\ML\PythonProject\app\static\charts\Rent_price_vs_bedrooms_distribution.png")
plt.show()


# ============================================================
# 15. CORRELATION MATRIX
# ============================================================

print("\n========================================")
print("CORRELATION MATRIX")
print("========================================")

correlation = df[
    [
        "price",
        "square_feet",
        "bedrooms",
        "bathrooms"
    ]
].corr()

print(correlation)


# ============================================================
# 16. TOP CITIES
# ============================================================

if "cityname" in df.columns:

    print("\n========================================")
    print("TOP 10 CITIES")
    print("========================================")

    print(
        df["cityname"]
        .value_counts()
        .head(10)
    )


# ============================================================
# 17. TOP STATES
# ============================================================

if "state" in df.columns:

    print("\n========================================")
    print("TOP 10 STATES")
    print("========================================")

    print(
        df["state"]
        .value_counts()
        .head(10)
    )


# ============================================================
# 18. AVERAGE RENT
# ============================================================

print("\n========================================")
print("AVERAGE RENT")
print("========================================")

print(
    "Average Rent:",
    df["price"].mean()
)


# ============================================================
# 19. MINIMUM AND MAXIMUM RENT
# ============================================================

print("\nMinimum Rent:")
print(df["price"].min())

print("\nMaximum Rent:")
print(df["price"].max())


# ============================================================
# FINISHED
# ============================================================

print("\n=========================================")
print("EDA COMPLETED SUCCESSFULLY")
print("==========================================")