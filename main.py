from src.data.preprocess import preprocess_dataset
from src.data.eda import run_eda

from src.MLmodels.logistic_regression import train as train_logistic
from src.MLmodels.linearregression import train_salary_model


def main():

    print("=" * 60)
    print("PLACEMENT PREDICTION SYSTEM")
    print("=" * 60)

    print("\n1. PREPROCESSING")
    preprocess_dataset()

    print("\n2. EDA")
    run_eda()

    print("\n3. LOGISTIC REGRESSION")
    train_logistic()

    print("\n4. LINEAR REGRESSION")
    train_salary_model()

    print("\n" + "=" * 60)
    print("PROJECT EXECUTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()