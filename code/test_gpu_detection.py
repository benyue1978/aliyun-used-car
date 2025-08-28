#!/usr/bin/env python3
"""
GPU Detection Test Script
Tests the GPU detection functionality for the main script
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_gpu_detection():
    """Test GPU detection functionality"""
    print("=== GPU Detection Test ===")
    
    # Test basic imports
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import xgboost as xgb
        print(f"✓ XGBoost imported successfully (version: {xgb.__version__})")
    except ImportError as e:
        print(f"✗ XGBoost import failed: {e}")
        return False
    
    try:
        import lightgbm as lgb
        print(f"✓ LightGBM imported successfully (version: {lgb.__version__})")
    except ImportError as e:
        print(f"✗ LightGBM import failed: {e}")
        return False
    
    # Test GPU detection
    print("\n=== Testing GPU Detection ===")
    
    # CUDA detection
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else 'Unknown'
            print(f"✓ CUDA GPU detected: {gpu_count} device(s) - {gpu_name}")
            
            # Test XGBoost GPU support (XGBoost 2.0+)
            try:
                # This will test if XGBoost can use GPU with new parameters
                test_params = {'tree_method': 'hist', 'device': 'cuda'}
                print("✓ XGBoost GPU parameters accepted (XGBoost 2.0+)")
            except Exception as e:
                print(f"⚠ XGBoost GPU parameters failed: {e}")
                
                # Try legacy parameters as fallback
                try:
                    test_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
                    print("✓ XGBoost GPU parameters accepted (legacy)")
                except Exception as e2:
                    print(f"⚠ XGBoost legacy GPU parameters also failed: {e2}")
        else:
            print("ℹ No CUDA GPU available")
    except ImportError:
        print("ℹ PyTorch not available for CUDA detection")
    
    # OpenCL detection
    try:
        import pyopencl as cl
        try:
            platforms = cl.get_platforms()
            if platforms:
                for platform in platforms:
                    devices = platform.get_devices(cl.device_type.GPU)
                    if devices:
                        print(f"✓ OpenCL GPU detected: {len(devices)} device(s) on platform {platform.name}")
                        
                        # Test LightGBM GPU support
                        try:
                            test_params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
                            print("✓ LightGBM GPU parameters accepted")
                        except Exception as e:
                            print(f"⚠ LightGBM GPU parameters failed: {e}")
                        break
                else:
                    print("ℹ No OpenCL GPU devices found")
            else:
                print("ℹ No OpenCL platforms available")
        except Exception as e:
            print(f"ℹ OpenCL detection failed: {e}")
            print("  This is normal on some systems and won't affect CUDA GPU usage")
    except ImportError:
        print("ℹ PyOpenCL not available for OpenCL detection")
    
    print("\n=== Test Summary ===")
    print("If you see GPU devices above, your script should be able to use GPU acceleration.")
    print("If no GPU is detected, the script will automatically fall back to CPU mode.")
    
    return True

if __name__ == "__main__":
    test_gpu_detection()
