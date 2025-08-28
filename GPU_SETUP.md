# GPU 加速配置说明

## 概述
本脚本已更新为支持GPU加速，能够自动检测可用的GPU设备并配置相应的参数。

## 支持的GPU类型

### 1. CUDA GPU (推荐)
- **检测方式**: 通过PyTorch检测CUDA设备
- **支持框架**: XGBoost, LightGBM
- **性能提升**: 2-8倍加速

### 2. OpenCL GPU
- **检测方式**: 通过PyOpenCL检测OpenCL设备
- **支持框架**: LightGBM
- **性能提升**: 1.5-5倍加速

## 安装依赖

### 基础安装
```bash
# 激活虚拟环境
source venv/bin/activate

# 安装GPU版本依赖
pip install -r code/requirements_gpu.txt
```

### CUDA版本选择
根据你的CUDA版本选择对应的PyTorch版本：
```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only (如果没有GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 自动配置

脚本会自动检测GPU并配置参数：

### XGBoost GPU配置
```python
if gpu_config['has_gpu']:
    xgb_params.update({
        'tree_method': 'hist',        # XGBoost 2.0+ 使用 hist 方法
        'device': 'cuda',             # 新版本使用 device 参数
        'predictor': 'gpu_predictor'
    })
```

**注意**: XGBoost 2.0+ 版本参数变化：
- 旧版本: `tree_method: 'gpu_hist'`, `gpu_id: 0`
- 新版本: `tree_method: 'hist'`, `device: 'cuda'`

### LightGBM GPU配置
```python
if gpu_config['has_gpu']:
    lgb_params.update({
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'force_row_wise': True,
    })
```

## 性能优化建议

### 1. 数据规模
- **小数据集** (< 10万样本): GPU加速效果不明显
- **中等数据集** (10万-100万样本): 1.5-3倍加速
- **大数据集** (> 100万样本): 3-8倍加速

### 2. 特征数量
- **低维特征** (< 100): GPU加速效果有限
- **高维特征** (> 100): GPU加速效果显著

### 3. 模型复杂度
- **浅层模型** (max_depth < 6): 加速效果一般
- **深层模型** (max_depth > 6): 加速效果明显

## 故障排除

### 常见问题

#### 1. "No OpenCL device found"
**原因**: LightGBM找不到OpenCL设备
**解决**: 确保安装了正确的GPU驱动和OpenCL运行时

#### 2. "CUDA out of memory"
**原因**: GPU显存不足
**解决**: 
- 减少batch_size
- 减少max_depth
- 使用CPU模式

#### 3. GPU检测失败
**原因**: 依赖库版本不兼容
**解决**: 更新到最新版本的torch、xgboost、lightgbm

#### 4. XGBoost参数警告
**原因**: XGBoost 2.0+ 版本参数格式变化
**解决**: 
- 新版本使用: `tree_method: 'hist'`, `device: 'cuda'`
- 旧版本使用: `tree_method: 'gpu_hist'`, `gpu_id: 0`
- 脚本已自动适配新版本参数

### 调试命令
```bash
# 检查CUDA可用性
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

# 检查XGBoost GPU支持
python -c "import xgboost as xgb; print('XGBoost version:', xgb.__version__)"

# 检查LightGBM GPU支持
python -c "import lightgbm as lgb; print('LightGBM version:', lgb.__version__)"
```

## 性能基准测试

在不同配置下的预期性能提升：

| 配置 | 数据规模 | 特征数量 | 预期加速 |
|------|----------|----------|----------|
| CPU | 50万样本 | 200特征 | 基准 |
| CUDA GPU | 50万样本 | 200特征 | 3-5倍 |
| OpenCL GPU | 50万样本 | 200特征 | 2-3倍 |

## 注意事项

1. **首次运行**: GPU首次运行会有编译时间，后续运行会更快
2. **内存管理**: GPU模式会占用更多内存，确保有足够的显存
3. **兼容性**: 不同GPU架构的性能可能差异较大
4. **回退机制**: 如果GPU不可用，脚本会自动回退到CPU模式
