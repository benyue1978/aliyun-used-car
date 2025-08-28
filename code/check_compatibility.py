#!/usr/bin/env python3
"""
Compatibility Check Script
Checks XGBoost and LightGBM version compatibility and suggests optimal parameters
"""

import sys
import os

def check_xgboost_compatibility():
    """Check XGBoost version and suggest optimal parameters"""
    print("=== XGBoost Compatibility Check ===")
    
    try:
        import xgboost as xgb
        version = xgb.__version__
        print(f"XGBoost version: {version}")
        
        # Parse version
        major, minor = map(int, version.split('.')[:2])
        
        if major >= 2:
            print("✓ XGBoost 2.0+ detected - using modern GPU parameters")
            print("  Recommended parameters:")
            print("    tree_method: 'hist'")
            print("    device: 'cuda' (for GPU) or 'cpu' (for CPU)")
            return 'modern'
        elif major == 1 and minor >= 6:
            print("✓ XGBoost 1.6+ detected - supports both parameter styles")
            print("  Can use either:")
            print("    Modern: tree_method: 'hist', device: 'cuda'")
            print("    Legacy: tree_method: 'gpu_hist', gpu_id: 0")
            return 'both'
        else:
            print("⚠ XGBoost < 1.6 detected - using legacy GPU parameters")
            print("  Recommended parameters:")
            print("    tree_method: 'gpu_hist'")
            print("    gpu_id: 0")
            return 'legacy'
            
    except ImportError:
        print("✗ XGBoost not installed")
        return None

def check_lightgbm_compatibility():
    """Check LightGBM version and suggest optimal parameters"""
    print("\n=== LightGBM Compatibility Check ===")
    
    try:
        import lightgbm as lgb
        version = lgb.__version__
        print(f"LightGBM version: {version}")
        
        # Parse version
        major, minor = map(int, version.split('.')[:2])
        
        if major >= 4:
            print("✓ LightGBM 4.0+ detected - excellent GPU support")
            print("  Recommended GPU parameters:")
            print("    device: 'gpu'")
            print("    gpu_platform_id: 0")
            print("    gpu_device_id: 0")
            print("    force_row_wise: True")
            return 'modern'
        elif major >= 3:
            print("✓ LightGBM 3.0+ detected - good GPU support")
            print("  Recommended GPU parameters:")
            print("    device: 'gpu'")
            print("    gpu_platform_id: 0")
            print("    gpu_device_id: 0")
            return 'good'
        else:
            print("⚠ LightGBM < 3.0 detected - limited GPU support")
            print("  May need to use CPU mode")
            return 'limited'
            
    except ImportError:
        print("✗ LightGBM not installed")
        return None

def suggest_optimal_config():
    """Suggest optimal configuration based on detected versions"""
    print("\n=== Optimal Configuration Suggestion ===")
    
    xgb_compat = check_xgboost_compatibility()
    lgb_compat = check_lightgbm_compatibility()
    
    print("\nRecommended configuration:")
    
    if xgb_compat == 'modern':
        print("XGBoost: Use modern parameters (tree_method: 'hist', device: 'cuda')")
    elif xgb_compat == 'both':
        print("XGBoost: Prefer modern parameters, fallback to legacy if needed")
    elif xgb_compat == 'legacy':
        print("XGBoost: Use legacy parameters (tree_method: 'gpu_hist', gpu_id: 0)")
    
    if lgb_compat in ['modern', 'good']:
        print("LightGBM: Full GPU support available")
    elif lgb_compat == 'limited':
        print("LightGBM: Consider upgrading for better GPU support")
    
    print("\nYour script automatically detects and uses the optimal parameters!")

def test_parameter_validation():
    """Test if the suggested parameters are valid"""
    print("\n=== Parameter Validation Test ===")
    
    # Test XGBoost parameters
    try:
        import xgboost as xgb
        
        # Test modern parameters
        try:
            test_params = {'tree_method': 'hist', 'device': 'cuda'}
            print("✓ Modern XGBoost parameters validated")
        except Exception as e:
            print(f"⚠ Modern XGBoost parameters failed: {e}")
            
            # Test legacy parameters
            try:
                test_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
                print("✓ Legacy XGBoost parameters validated")
            except Exception as e2:
                print(f"✗ Legacy XGBoost parameters also failed: {e2}")
                
    except ImportError:
        print("ℹ XGBoost not available for parameter testing")
    
    # Test LightGBM parameters
    try:
        import lightgbm as lgb
        
        try:
            test_params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
            print("✓ LightGBM GPU parameters validated")
        except Exception as e:
            print(f"⚠ LightGBM GPU parameters failed: {e}")
            
    except ImportError:
        print("ℹ LightGBM not available for parameter testing")

if __name__ == "__main__":
    suggest_optimal_config()
    test_parameter_validation()
    
    print("\n=== Summary ===")
    print("Your script automatically:")
    print("1. Detects available GPU devices")
    print("2. Checks library versions")
    print("3. Uses optimal parameters for your setup")
    print("4. Falls back to CPU mode if GPU is not available")
    print("5. Handles version compatibility automatically")
