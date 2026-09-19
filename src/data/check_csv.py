import pandas as pd


file_path = r"C:\2nd year odd\ML\PythonProject\data\apartments_for_rent_classified_10K.csv"

with open(file_path, "r", encoding="cp1252") as f:
    for i in range(3):
        print(f.readline())


try:

    df = pd.read_csv(
        file_path,
        encoding="cp1252",
        engine="python",
        on_bad_lines="warn"
    )

    print("========================================")
    print("CSV CHECK")
    print("========================================")

    print("Shape:", df.shape)

    print("\nColumns:")
    for i, column in enumerate(df.columns):
        print(i + 1, ":", column)

except Exception as e:

    print("ERROR:")
    print(e)