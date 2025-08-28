import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import time
import gc
import io
import joblib # Import joblib
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
    # Removed df.iloc[1:] to ensure all rows are read
    df = df.reset_index(drop=True)
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
                     'notRepairedDamage', 'bodyType', 'fuelType', 'gearbox',
                     'power_x_kilometer', 'car_age_x_kilometer', 'power_x_car_age']]
    
    log(f'\nUsing {len(feature_cols)} feature columns for training.')

    X_train = df_train_subset[feature_cols]
    y_train = np.log1p(df_train_subset['price'].values)
    X_val = df_val_subset[feature_cols]
    y_val = np.log1p(df_val_subset['price'].values)
    X_testB = testB_df[feature_cols]

    dtrain = DMatrix(X_train, label=y_train)
    dval = DMatrix(X_val, label=y_val)
    dtestB = DMatrix(X_testB)

    # GPU detection and configuration
    def detect_gpu():
        """Detect available GPU devices and return configuration"""
        gpu_info = {'has_gpu': False, 'device': 'cpu', 'gpu_count': 0, 'gpu_type': 'none'}
        
        # Try CUDA detection first (for XGBoost and PyTorch)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else 'Unknown'
                log(f'CUDA GPU detected: {gpu_count} device(s) - {gpu_name}')
                gpu_info.update({
                    'has_gpu': True, 
                    'gpu_count': gpu_count, 
                    'device': 'gpu',
                    'gpu_type': 'cuda'
                })
                return gpu_info
        except ImportError:
            log('PyTorch not available for CUDA detection')
        
        # Try XGBoost CUDA detection as fallback
        try:
            import xgboost as xgb
            # Check if XGBoost was compiled with CUDA support
            if hasattr(xgb, 'get_config') and 'use_cuda' in xgb.get_config():
                log('XGBoost CUDA support detected')
                gpu_info.update({
                    'has_gpu': True,
                    'gpu_count': 1,
                    'device': 'gpu',
                    'gpu_type': 'cuda'
                })
                return gpu_info
        except Exception:
            pass
        
        # Try OpenCL detection
        try:
            import pyopencl as cl
            try:
                platforms = cl.get_platforms()
                if platforms:
                    for platform in platforms:
                        devices = platform.get_devices(cl.device_type.GPU)
                        if devices:
                            log(f'OpenCL GPU detected: {len(devices)} device(s) on platform {platform.name}')
                            gpu_info.update({
                                'has_gpu': True,
                                'gpu_count': len(devices),
                                'device': 'gpu',
                                'gpu_type': 'opencl'
                            })
                            return gpu_info
            except Exception as e:
                log(f'OpenCL detection failed: {e} (this is normal on some systems)')
        except ImportError:
            log('PyOpenCL not available for OpenCL detection')
        
        log('No GPU detected, using CPU mode')
        return gpu_info
    
    gpu_config = detect_gpu()
    
    # XGBoost parameters with GPU support
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'n_estimators': 100000, # Further increased estimators
        'learning_rate': 0.01, # Reduced learning rate
        'max_depth': 8, # Increased max_depth
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0, # L2 regularization
        'seed': 42,
        'n_jobs': -1
    }
    
    # Enable GPU acceleration for XGBoost if available
    if gpu_config['has_gpu']:
        xgb_params.update({
            'tree_method': 'hist',  # Use hist method with device parameter
            'device': 'cuda',       # New XGBoost 2.0+ GPU parameter
            'predictor': 'gpu_predictor',
            'max_bin': 256,         # Optimized for NVIDIA A10 memory
            'grow_policy': 'lossguide'  # Better for GPU training
        })
        log('XGBoost GPU acceleration enabled (CUDA) - Optimized for NVIDIA A10')
    else:
        xgb_params.update({
            'tree_method': 'hist',
            'device': 'cpu'
        })
        log('XGBoost using CPU mode')

    log('\n--- Training XGBoost Model with Early Stopping ---')

    bst = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=xgb_params['n_estimators'],
        evals=[(dval, 'validation_0')],
        early_stopping_rounds=100,
        verbose_eval=False
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

    import lightgbm as lgb

    # LightGBM parameters with GPU support
    lgb_params = {
        'objective': 'regression_l1', # MAE objective
        'metric': 'mae',
        'n_estimators': 100000, # Increased estimators for manual early stopping
        'learning_rate': 0.01,
        'num_leaves': 64,
        'max_depth': 8,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0, # L2 regularization
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1, # Suppress verbose output
    }
    
    # Enable GPU acceleration for LightGBM if available
    if gpu_config['has_gpu']:
        lgb_params.update({
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'force_row_wise': True,  # Better performance on GPU
            'max_bin': 255,          # Optimized for NVIDIA A10 memory
        })
        log('LightGBM GPU acceleration enabled - Optimized for NVIDIA A10')
    else:
        lgb_params.update({
            'device': 'cpu',
            'force_col_wise': True,  # Better performance on CPU
        })
        log('LightGBM using CPU mode')

    log('\n--- Training LightGBM Model with Early Stopping ---')

    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='mae',
                  callbacks=[lgb.early_stopping(100, verbose=False)])

    val_preds_lgb_log_scale = lgb_model.predict(X_val)
    testB_preds_lgb_log_scale = lgb_model.predict(X_testB)

    # Blending predictions on validation set
    val_preds_blended_log_scale = 0.5 * val_preds_log_scale + 0.5 * val_preds_lgb_log_scale
    val_preds_blended_original_scale = np.expm1(val_preds_blended_log_scale)
    mae_blended_original_scale = mean_absolute_error(y_val_original_scale, val_preds_blended_original_scale)
    log(f'Final Blended Validation MAE (Original Scale): {mae_blended_original_scale:.2f}')

    # Blending predictions for testB
    blend_preds = 0.5 * np.expm1(testB_preds_xgb) + 0.5 * np.expm1(testB_preds_lgb_log_scale)
    blend_preds = np.clip(blend_preds, 0, None)

    outB = pd.DataFrame({'SaleID': testB_df['SaleID'], 'price': blend_preds})
    outB.to_csv('prediction_result/predictions_blended.csv', index=False)
    log('All predictions finished!')

    # Save models
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(bst, os.path.join(model_dir, 'xgboost_model.pkl'))
    joblib.dump(lgb_model, os.path.join(model_dir, 'lightgbm_model.pkl'))
    log(f'Models saved to {model_dir} directory.')