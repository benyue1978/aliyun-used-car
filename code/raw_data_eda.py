import pandas as pd
import numpy as np
import io

def log(msg):
    print(f"[LOG] {msg}")

def read_csv_correctly(path):
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(' ')
        data_content = f.read()
    csv_content = ' '.join(header) + '\n' + data_content
    csv_content = csv_content.replace(' ', ',')
    data_io = io.StringIO(csv_content)
    df = pd.read_csv(data_io, sep=',', na_values=['-'])
    df.columns = header
    df = df.iloc[1:].reset_index(drop=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    return df

def raw_data_eda():
    log("--- Loading Raw Data ---")
    train_path = 'data/used_car_train_20200313.csv'
    testB_path = 'data/used_car_testB_20200421.csv'

    df_train = read_csv_correctly(train_path)
    df_testB = read_csv_correctly(testB_path)

    log(f'Raw Train shape: {df_train.shape}')
    log(f'Raw TestB shape: {df_testB.shape}')

    # Categorical features to analyze
    categorical_cols = ['seller', 'offerType', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
    log("\n--- Categorical Feature Analysis (Raw Data) ---")
    for col in categorical_cols:
        log(f"\nFeature: {col}")
        log(f"  Train Unique Values and Counts (including NaN):\n{df_train[col].value_counts(dropna=False)}")
        log(f"  TestB Unique Values and Counts (including NaN):\n{df_testB[col].value_counts(dropna=False)}")

    # Numerical features to analyze
    numerical_cols = ['power', 'kilometer'] + [f'v_{i}' for i in range(15)]
    log("\n--- Numerical Feature Analysis (Raw Data) ---")
    for col in numerical_cols:
        log(f"\nFeature: {col}")
        log(f"  Train Describe:\n{df_train[col].describe()}")
        log(f"  TestB Describe:\n{df_testB[col].describe()}")

if __name__ == '__main__':
    raw_data_eda()
