# NVIDIA A10 GPU 专用配置指南

## 🎯 概述
你的机器配备了**NVIDIA A10 GPU**，这是一款高性能的数据中心GPU，非常适合机器学习任务。本指南将帮助你最大化利用GPU性能。

## 🚀 GPU 规格
- **GPU**: NVIDIA A10
- **内存**: 24GB GDDR6
- **CUDA核心**: 9,216
- **张量核心**: 288 (第2代)
- **内存带宽**: 600 GB/s
- **CUDA版本**: 支持最新版本

## 📊 性能预期
基于你的GPU配置，预期性能提升：

| 数据规模 | 特征数量 | 预期加速 |
|----------|----------|----------|
| < 10万样本 | < 100特征 | 1.5-2倍 |
| 10万-100万样本 | 100-500特征 | 2-4倍 |
| > 100万样本 | > 500特征 | 4-8倍 |

## 🛠️ 安装步骤

### 1. 激活虚拟环境
```bash
source venv/bin/activate
```

### 2. 安装GPU依赖
```bash
# 安装CUDA版本的PyTorch (推荐CUDA 11.8或12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r code/requirements_gpu.txt
```

### 3. 验证安装
```bash
# 检查CUDA可用性
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0))"

# 检查XGBoost GPU支持
python -c "import xgboost as xgb; print('XGBoost version:', xgb.__version__)"

# 检查LightGBM GPU支持
python -c "import lightgbm as lgb; print('LightGBM version:', lgb.__version__)"
```

## 🎮 运行脚本

### 1. GPU检测测试
```bash
python code/test_gpu_detection.py
```
**预期输出**: 应该显示 "✓ CUDA GPU detected: 1 device(s) - NVIDIA A10"

### 2. 兼容性检查
```bash
python code/check_compatibility.py
```
**预期输出**: 显示XGBoost 2.0+和LightGBM 4.0+的兼容性

### 3. NVIDIA A10优化配置
```bash
python code/nvidia_a10_optimization.py
```
**功能**: 生成针对A10优化的配置文件

### 4. 运行主脚本
```bash
python code/main.py
```
**预期结果**: 自动检测GPU并使用优化参数，无警告信息

## ⚙️ 优化参数说明

### XGBoost GPU优化
```python
xgb_params = {
    'tree_method': 'hist',        # 使用hist方法配合device参数
    'device': 'cuda',             # 启用CUDA GPU加速
    'predictor': 'gpu_predictor', # GPU预测器
    'max_bin': 256,               # 针对A10内存优化
    'grow_policy': 'lossguide',   # GPU训练优化策略
    'max_leaves': 255,            # 最大叶子数
    'reg_alpha': 0.1,             # L1正则化
    'reg_lambda': 1.0,            # L2正则化
}
```

### LightGBM GPU优化
```python
lgb_params = {
    'device': 'gpu',              # 启用GPU
    'gpu_platform_id': 0,         # GPU平台ID
    'gpu_device_id': 0,           # GPU设备ID
    'force_row_wise': True,       # GPU性能优化
    'max_bin': 255,               # 针对A10内存优化
    'reg_alpha': 0.1,             # L1正则化
    'reg_lambda': 1.0,            # L2正则化
}
```

## 📈 性能监控

### GPU内存使用
```bash
# 实时监控GPU使用情况
watch -n 1 nvidia-smi

# 或者使用
nvidia-smi -l 1
```

### 预期内存使用
- **XGBoost**: 2-8GB GPU内存
- **LightGBM**: 1-6GB GPU内存
- **系统预留**: 10GB+ (确保稳定性)

## 🔧 故障排除

### 常见问题

#### 1. "CUDA out of memory"
**原因**: GPU内存不足
**解决**: 
- 减少`max_bin`参数 (256 → 128)
- 减少`max_leaves`参数 (255 → 127)
- 减少`max_depth`参数 (8 → 6)

#### 2. "No OpenCL device found"
**原因**: OpenCL检测失败
**影响**: 不影响CUDA GPU使用，脚本会自动使用CUDA
**解决**: 忽略此警告，CUDA GPU正常工作

#### 3. XGBoost参数警告
**原因**: 版本兼容性
**状态**: ✅ 已修复，使用新版本参数格式

### 调试命令
```bash
# 检查CUDA环境
nvcc --version
nvidia-smi

# 检查Python环境
python -c "import torch; print(torch.version.cuda)"
python -c "import xgboost as xgb; print(xgb.__version__)"
```

## 🎉 成功标志

当一切配置正确时，你应该看到：

1. **GPU检测成功**: "CUDA GPU detected: 1 device(s) - NVIDIA A10"
2. **无警告信息**: XGBoost和LightGBM参数验证通过
3. **性能提升**: 训练速度明显快于CPU模式
4. **内存稳定**: GPU内存使用在合理范围内

## 📞 技术支持

如果遇到问题：

1. 检查CUDA驱动是否正确安装
2. 确认PyTorch、XGBoost、LightGBM版本兼容性
3. 运行诊断脚本获取详细信息
4. 检查GPU内存是否充足

---

**🎯 目标**: 在你的NVIDIA A10 GPU上获得2-8倍的性能提升！
**🚀 开始**: 运行 `python code/main.py` 享受GPU加速！
