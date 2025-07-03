import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
train_path = 'data/used_car_train_20200313.csv'
df = pd.read_csv(train_path, delim_whitespace=True, na_values=['-'])

print('【数据集基本信息】')
print('样本数:', df.shape[0], '字段数:', df.shape[1])
print('字段名:', df.columns.tolist())

print('\n【字段缺失率】')
print(df.isnull().mean().sort_values(ascending=False))

print('\n【目标变量 price 分布】')
print(df['price'].describe())
plt.figure(figsize=(8,4))
sns.histplot(df['price'], bins=100, kde=True)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 数值特征
num_cols = ['power', 'kilometer'] + [f'v_{i}' for i in range(15)]
print('\n【数值特征分布】')
for col in num_cols:
    print(f'\n{col} 描述:')
    print(df[col].describe())
    if df[col].nunique() < 50:
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'{col} Distribution')
        plt.tight_layout()
        plt.show()

# 类别特征
cat_cols = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode', 'seller', 'offerType']
print('\n【类别特征分布】')
for col in cat_cols:
    print(f'\n{col} 唯一值数:', df[col].nunique())
    print(df[col].value_counts(dropna=False).head(10))
    if df[col].nunique() < 20:
        sns.countplot(x=col, data=df)
        plt.title(f'{col} Count')
        plt.tight_layout()
        plt.show()

# 目标与主要特征关系
print('\n【目标与主要特征关系】')
for col in ['brand', 'bodyType', 'fuelType', 'gearbox']:
    if col in df.columns:
        plt.figure(figsize=(10,4))
        sns.boxplot(x=col, y='price', data=df)
        plt.title(f'Price vs {col}')
        plt.tight_layout()
        plt.show()

print('\n【相关性分析】')
corr = df[num_cols + ['price']].corr()
print(corr['price'].sort_values(ascending=False))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show() 