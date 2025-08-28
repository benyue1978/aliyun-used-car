#!/usr/bin/env python3
"""
Simple LightGBM GPU Test
Tests if LightGBM can use GPU with the correct parameters
"""

import numpy as np
import lightgbm as lgb

def test_lightgbm_gpu():
    """Test LightGBM GPU functionality"""
    print("=== LightGBM GPU Test ===")
    
    # Create simple test data
    np.random.seed(42)
    X = np.random.rand(1000, 10)
    y = np.random.rand(1000)
    
    print(f"Data shape: {X.shape}")
    
    # Test GPU parameters
    gpu_params = {
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'n_estimators': 10,
        'verbose': -1
    }
    
    print("GPU parameters:", gpu_params)
    
    try:
        # Create and train model
        print("\nTraining LightGBM model with GPU...")
        model = lgb.LGBMRegressor(**gpu_params)
        model.fit(X, y)
        
        print("✅ LightGBM GPU training successful!")
        
        # Make prediction
        pred = model.predict(X[:10])
        print(f"✅ Prediction successful! Shape: {pred.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ LightGBM GPU training failed: {e}")
        return False

if __name__ == "__main__":
    success = test_lightgbm_gpu()
    
    if success:
        print("\n🎉 LightGBM GPU is working correctly!")
        print("Your main script should now work with GPU acceleration.")
    else:
        print("\n⚠️ LightGBM GPU is not working.")
        print("We need to investigate further.")
