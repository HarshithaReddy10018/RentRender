import pandas as pd


def load_data():

    file_path = r"C:\2nd year odd\ML\PythonProject\data\apartments_for_rent_classified_10K.csv"

    df = pd.read_csv(
        file_path,
        sep=";",
        encoding="cp1252",
        engine="python"
    )

    return df


def get_summary(df):

    return {
        "rows": df.shape[0],
        "columns": df.shape[1]
    }