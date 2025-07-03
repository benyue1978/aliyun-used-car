# 解决方案及算法介绍

## 项目结构 Project Structure

```dir
project
|-- README.md
|-- data
|-- user_data
|-- feature
|-- model
|-- prediction_result
|   |-- predictions.csv
|-- code
    |-- main.py
    |-- requirements.txt
```

- `data/`：存放原始数据（无需提交数据文件，仅结构）。
- `user_data/`：存放预测过程中生成的中间数据。
- `feature/`：特征工程相关代码。
- `model/`：模型训练相关代码。
- `prediction_result/`：预测结果输出文件夹，最终结果为 `predictions.csv`。
- `code/`：主程序及依赖说明。

## PyTorch方案说明

本项目采用PyTorch实现MLP回归模型，支持A榜和B榜预测：

- 训练集：data/used_car_train_20200313.csv
- 测试集A：data/used_car_testA_20200313.csv
- 测试集B：data/used_car_testB_20200421.csv
- 预测结果分别输出到 prediction_result/predictions.csv（A榜）和 prediction_result/predictions_B.csv（B榜）
- 训练好的模型参数保存在 model/mlp.pth

### 运行方式

1. 安装依赖：

```bash
pip install -r code/requirements.txt
```

1. 运行主程序：

```bash
python code/main.py
```

---

### 其他说明 Others

- 读入文件路径均为相对路径，如 `../data/XX`。
- 如需额外参数，请在 `main.py` 或 `main.sh` 中说明。
- 若有特殊注意事项，请在此补充。
