import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import time
import gc
import io
from feature.generation import FeatureGenerator
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from xgboost import DMatrix

def log(msg):
    print(f"[LOG][{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def timeit(msg):
    class Timer:
        def __enter__(self):
            self.start = time.time()
            log(f"[START] {msg}")
        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self.start
            log(f"[END] {msg} | Time elapsed: {elapsed:.2f}s")
    return Timer()

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

if __name__ == '__main__':
    with timeit('Loading data'):
        train_path = 'data/used_car_train_20200313.csv'
        testB_path = 'data/used_car_testB_20200421.csv'

        df = read_csv_correctly(train_path)
        testB_df = read_csv_correctly(testB_path)
        log(f'Original Train shape: {df.shape}, TestB shape: {testB_df.shape}')

        # Drop useless columns early
        df.drop(['seller', 'offerType'], axis=1, inplace=True)
        testB_df.drop(['seller', 'offerType'], axis=1, inplace=True)

    

    df = df[df['price'].notna() & (df['price'] >= 0)].reset_index(drop=True)

    # Split train data into 90% for training and 10% for validation
    df_train_subset, df_val_subset = train_test_split(df, test_size=0.1, random_state=42)

    fg = FeatureGenerator()
    with timeit('Fitting feature generator'):
        fg.fit(df_train_subset.copy())
    with timeit('Transforming train subsets and testB'):
        df_train_subset = fg.transform(df_train_subset.copy())
        df_val_subset = fg.transform(df_val_subset.copy())
        testB_df = fg.transform(testB_df.copy())
    log('Feature engineering done.')
    gc.collect()

    # Define feature columns to use for training
    feature_cols = [col for col in df_train_subset.columns if col not in 
                    ['SaleID', 'price', 'name', 'regDate', 'creatDate', 
                     'notRepairedDamage', 'bodyType', 'fuelType', 'gearbox']]
    
    log(f'\nUsing {len(feature_cols)} feature columns for training.')

    X_train = df_train_subset[feature_cols]
    y_train = np.log1p(df_train_subset['price'].values)
    X_val = df_val_subset[feature_cols]
    y_val = np.log1p(df_val_subset['price'].values)
    X_testB = testB_df[feature_cols]

    dtrain = DMatrix(X_train, label=y_train)
    dval = DMatrix(X_val, label=y_val)
    dtestB = DMatrix(X_testB)

    # XGBoost parameters (can be tuned later)
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'n_estimators': 50000, # Increased estimators for manual early stopping
        'learning_rate': 0.01,
        'max_depth': 10,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'seed': 42,
        'n_jobs': -1
    }

    log('\n--- Training XGBoost Model with Early Stopping ---')

    bst = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=xgb_params['n_estimators'],
        evals=[(dval, 'validation_0')],
        early_stopping_rounds=100,
        verbose_eval=True
    )

    best_iteration = bst.best_iteration
    best_mae_log_scale = bst.best_score
    log(f'Final XGBoost Validation MAE (Log Scale, Best Iteration): {best_mae_log_scale}')

    # Calculate MAE on original scale
    val_preds_log_scale = bst.predict(dval, iteration_range=(0, best_iteration))
    val_preds_original_scale = np.expm1(val_preds_log_scale)
    y_val_original_scale = np.expm1(y_val)
    mae_original_scale = mean_absolute_error(y_val_original_scale, val_preds_original_scale)
    log(f'Final XGBoost Validation MAE (Original Scale): {mae_original_scale:.2f}')

    # Create submission file if needed
    testB_preds_xgb = bst.predict(dtestB, iteration_range=(0, best_iteration))
    predB = np.expm1(testB_preds_xgb)
    predB = np.clip(predB, 0, None)
    outB = pd.DataFrame({'SaleID': testB_df['SaleID'], 'price': predB})
    outB.to_csv('prediction_result/predictions_xgb.csv', index=False)
    log('All predictions finished!')