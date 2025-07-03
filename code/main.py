import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import time
import gc
from feature.generation import FeatureGenerator
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set device (for future deep learning use, not used here)
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def batch_predict(model, X, batch_size=5000):
    preds = []
    for i in range(0, X.shape[0], batch_size):
        preds.append(model.predict(X[i:i+batch_size]))
    return np.concatenate(preds)

if __name__ == '__main__':
    with timeit('Loading data'):
        train_path = '../data/used_car_train_20200313.csv' if not os.path.exists('data/used_car_train_20200313.csv') else 'data/used_car_train_20200313.csv'
        testA_path = '../data/used_car_testA_20200313.csv' if not os.path.exists('data/used_car_testA_20200313.csv') else 'data/used_car_testA_20200313.csv'
        testB_path = '../data/used_car_testB_20200421.csv' if not os.path.exists('data/used_car_testB_20200421.csv') else 'data/used_car_testB_20200421.csv'
        df = pd.read_csv(train_path, delim_whitespace=True, na_values=['-'])
        testA_df = pd.read_csv(testA_path, delim_whitespace=True, na_values=['-'])
        testB_df = pd.read_csv(testB_path, delim_whitespace=True, na_values=['-'])
        log(f'Train shape: {df.shape}, TestA shape: {testA_df.shape}, TestB shape: {testB_df.shape}')

    with timeit('Splitting train/val set'):
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        log(f'Train split (80%): {train_df.shape}, Test split (20%): {val_df.shape}')
    train_df = train_df[train_df['price'].notna() & (train_df['price'] >= 0)].reset_index(drop=True)
    val_df = val_df[val_df['price'].notna() & (val_df['price'] >= 0)].reset_index(drop=True)
    del df; gc.collect()

    fg = FeatureGenerator()
    with timeit('Fitting feature generator'):
        fg.fit(train_df)
    with timeit('Transforming train/val/testA/testB'):
        train_df = fg.transform(train_df)
        val_df = fg.transform(val_df)
        testA_df = fg.transform(testA_df)
        testB_df = fg.transform(testB_df)
    log('Feature engineering done.')
    gc.collect()

    # Use all features except price and SaleID
    exclude_cols = ['price', 'SaleID']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    X_train = train_df[feature_cols].values
    y_train = np.log1p(train_df['price'].values)
    X_val = val_df[feature_cols].values
    y_val = np.log1p(val_df['price'].values)
    del train_df; del val_df; gc.collect()

    with timeit('Training XGBoost'):
        xgb_model = xgb.XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=0.5,
            max_depth=8,
            tree_method='hist',
            eval_metric='mae'
        )
        xgb_model.fit(X_train, y_train)
    os.makedirs('model', exist_ok=True)
    xgb_model.save_model('model/xgb_model.json')
    log('XGBoost model saved to model/xgb_model.json')

    with timeit('Evaluating on validation set'):
        val_pred_xgb = xgb_model.predict(X_val)
        val_pred_xgb = np.expm1(val_pred_xgb)
        val_pred_xgb = np.clip(val_pred_xgb, 0, None)
        y_val_true = np.expm1(y_val)
        mse_xgb = mean_squared_error(y_val_true, val_pred_xgb)
        mae_xgb = mean_absolute_error(y_val_true, val_pred_xgb)
        log(f'XGBoost val MSE: {mse_xgb:.4f}')
        log(f'XGBoost val MAE: {mae_xgb:.4f}')
        log('Sample validation predictions (XGBoost):')
        print(pd.DataFrame({'true': y_val_true[:10], 'pred': val_pred_xgb[:10]}))
    del X_train; del y_train; del X_val; del y_val; gc.collect()

    with timeit('Predicting TestA and TestB'):
        X_testA = testA_df[feature_cols].values
        X_testB = testB_df[feature_cols].values
        predA = batch_predict(xgb_model, X_testA)
        predB = batch_predict(xgb_model, X_testB)
        predA = np.expm1(predA)
        predB = np.expm1(predB)
        predA = np.clip(predA, 0, None)
        predB = np.clip(predB, 0, None)
    outA = pd.DataFrame({'SaleID': testA_df['SaleID'], 'price': predA})
    outB = pd.DataFrame({'SaleID': testB_df['SaleID'], 'price': predB})
    outA.to_csv('prediction_result/predictions_A.csv', index=False)
    outB.to_csv('prediction_result/predictions.csv', index=False)
    log('All predictions finished!')
