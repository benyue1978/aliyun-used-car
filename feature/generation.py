import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold

# List of categorical and numerical features
CATEGORICAL_FEATURES = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode', 'seller', 'offerType']
NUMERICAL_FEATURES = ['power', 'kilometer'] + [f'v_{i}' for i in range(15)]
LOW_CARDINALITY = ['bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'seller', 'offerType']
HIGH_CARDINALITY = ['model', 'brand', 'regionCode']

def extract_year(series):
    # Extract 4-digit year, fallback to most common year if not found
    years = series.astype(str).str.extract(r'(\d{4})')[0]
    # Fill missing with mode year or 2000
    mode_year = years.dropna().mode()
    fill_year = mode_year.iloc[0] if not mode_year.empty else '2000'
    years = years.fillna(fill_year)
    return years.astype(int)

# Feature generation and transformation
class FeatureGenerator:
    def __init__(self):
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.scaler = None
        self.fitted = False
        self.stat_cols = []
        self.cross_cols = []
        self.onehot_cols = []
        self.brand_stat_map = {}
        self.model_target_mean = None
        self.cross_le = {}
        self.v_bin_ohe = {}
        self.missing_cols = []
        self.global_price_mean = None

    def fit(self, df):
        # Fillna and type conversion
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].fillna('-1').astype(str)
        for col in NUMERICAL_FEATURES:
            df[col] = df[col].fillna(0)
        # 缺失指示器
        for col in CATEGORICAL_FEATURES + NUMERICAL_FEATURES:
            if df[col].isnull().any():
                df[f'{col}_missing'] = df[col].isnull().astype(int)
                self.missing_cols.append(f'{col}_missing')
        # Log1p for power
        df['power_log'] = np.log1p(df['power'])
        # car_age
        reg_year = extract_year(df['regDate'])
        creat_year = extract_year(df['creatDate'])
        df['car_age'] = (creat_year - reg_year).clip(lower=0)
        # 交叉特征
        df['brand_bodyType'] = df['brand'].astype(str) + '_' + df['bodyType'].astype(str)
        df['brand_fuelType'] = df['brand'].astype(str) + '_' + df['fuelType'].astype(str)
        df['model_gearbox'] = df['model'].astype(str) + '_' + df['gearbox'].astype(str)
        for cross in ['brand_bodyType','brand_fuelType','model_gearbox']:
            le = LabelEncoder()
            all_classes = list(df[cross].unique())
            if '-1' not in all_classes:
                all_classes.append('-1')
            le.fit(all_classes)
            self.cross_le[cross] = le
        # model目标均值编码（K折）
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        df['model_target_mean'] = 0
        for train_idx, val_idx in kf.split(df):
            m = df.iloc[train_idx].groupby('model')['price'].mean()
            df.loc[df.index[val_idx], 'model_target_mean'] = df.iloc[val_idx]['model'].map(m)
        self.model_target_mean = df.groupby('model')['price'].mean().to_dict()
        self.global_price_mean = df['price'].mean()
        # v_0/v_12分箱+OneHot
        for v in ['v_0','v_12']:
            df[f'{v}_bin'] = pd.qcut(df[v], q=5, duplicates='drop', labels=False)
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe.fit(df[[f'{v}_bin']])
            self.v_bin_ohe[v] = ohe
        # 其余原有特征工程
        df['kilometer_bin'] = pd.cut(df['kilometer'], bins=[-1,5,10,15,20], labels=[0,1,2,3])
        df['kilometer_bin'] = df['kilometer_bin'].cat.add_categories([-1]).fillna(-1).astype(int)
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit(df[['kilometer_bin']])
        self.onehot_encoders['kilometer_bin'] = ohe
        self.onehot_cols = list(ohe.get_feature_names_out(['kilometer_bin']))
        # brand统计特征
        stat_df = df.groupby('brand')['price'].agg(['mean','median','count']).reset_index()
        stat_df.columns = ['brand','brand_price_mean','brand_price_median','brand_count']
        self.stat_cols = ['brand_price_mean','brand_price_median','brand_count']
        self.brand_stat_map = stat_df.set_index('brand')[self.stat_cols].to_dict(orient='index')
        df = df.merge(stat_df, on='brand', how='left')
        # OneHot for low cardinality
        for col in LOW_CARDINALITY:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe.fit(df[[col]])
            self.onehot_encoders[col] = ohe
            self.onehot_cols += list(ohe.get_feature_names_out([col]))
        # LabelEncoder for high cardinality
        for col in HIGH_CARDINALITY:
            le = LabelEncoder()
            all_classes = list(df[col].unique())
            if '-1' not in all_classes:
                all_classes.append('-1')
            le.fit(all_classes)
            self.label_encoders[col] = le
        # Standardize numerical features
        num_cols = NUMERICAL_FEATURES + ['power_log','car_age'] + self.stat_cols + ['model_target_mean']
        self.scaler = StandardScaler()
        self.scaler.fit(df[num_cols])
        self.num_cols = num_cols
        self.fitted = True

    def transform(self, df):
        assert self.fitted, 'Call fit() first!'
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].fillna('-1').astype(str)
        for col in NUMERICAL_FEATURES:
            df[col] = df[col].fillna(0)
        # 缺失指示器
        for col in self.missing_cols:
            base = col.replace('_missing','')
            df[col] = df[base].isnull().astype(int)
        df['power_log'] = np.log1p(df['power'])
        reg_year = extract_year(df['regDate'])
        creat_year = extract_year(df['creatDate'])
        df['car_age'] = (creat_year - reg_year).clip(lower=0)
        # 交叉特征
        df['brand_bodyType'] = df['brand'].astype(str) + '_' + df['bodyType'].astype(str)
        df['brand_fuelType'] = df['brand'].astype(str) + '_' + df['fuelType'].astype(str)
        df['model_gearbox'] = df['model'].astype(str) + '_' + df['gearbox'].astype(str)
        for cross in ['brand_bodyType','brand_fuelType','model_gearbox']:
            le = self.cross_le[cross]
            df[cross] = df[cross].where(df[cross].isin(le.classes_), '-1')
            df[cross] = le.transform(df[cross])
        # model目标均值编码
        if 'price' in df.columns:
            df['model_target_mean'] = df['model'].map(self.model_target_mean).fillna(df['price'].mean())
        else:
            df['model_target_mean'] = df['model'].map(self.model_target_mean).fillna(self.global_price_mean)
        # v_0/v_12分箱+OneHot
        v_bin_df_list = []
        for v in ['v_0','v_12']:
            df[f'{v}_bin'] = pd.qcut(df[v], q=5, duplicates='drop', labels=False)
            ohe = self.v_bin_ohe[v]
            arr = ohe.transform(df[[f'{v}_bin']])
            names = ohe.get_feature_names_out([f'{v}_bin'])
            v_bin_df_list.append(pd.DataFrame(arr, columns=names, index=df.index))
        v_bin_df = pd.concat(v_bin_df_list, axis=1)
        # 其余原有特征工程
        df['kilometer_bin'] = pd.cut(df['kilometer'], bins=[-1,5,10,15,20], labels=[0,1,2,3])
        df['kilometer_bin'] = df['kilometer_bin'].cat.add_categories([-1]).fillna(-1).astype(int)
        ohe = self.onehot_encoders['kilometer_bin']
        kb_ohe = ohe.transform(df[['kilometer_bin']])
        kb_ohe_df = pd.DataFrame(kb_ohe, columns=ohe.get_feature_names_out(['kilometer_bin']), index=df.index)
        # brand统计特征
        if not all(col in df.columns for col in self.stat_cols):
            for stat in self.stat_cols:
                df[stat] = df['brand'].map(lambda x: self.brand_stat_map.get(x, {}).get(stat, np.nan))
        # OneHot for low cardinality
        onehot_df_list = [kb_ohe_df, v_bin_df]
        for col in LOW_CARDINALITY:
            ohe = self.onehot_encoders[col]
            arr = ohe.transform(df[[col]])
            names = ohe.get_feature_names_out([col])
            onehot_df_list.append(pd.DataFrame(arr, columns=names, index=df.index))
        onehot_df = pd.concat(onehot_df_list, axis=1)
        # LabelEncoder for high cardinality
        for col in HIGH_CARDINALITY:
            le = self.label_encoders[col]
            df[col] = df[col].where(df[col].isin(le.classes_), '-1')
            df[col] = le.transform(df[col])
        # Standardize numerical features
        scaled = self.scaler.transform(df[self.num_cols])
        scaled_df = pd.DataFrame(scaled, columns=[col+'_scaled' for col in self.num_cols], index=df.index)
        # 拼接所有特征
        df = pd.concat([df, onehot_df, scaled_df], axis=1)
        return df

# Usage example:
# fg = FeatureGenerator()
# fg.fit(train_df)
# train_df = fg.transform(train_df)
# test_df = fg.transform(test_df) 