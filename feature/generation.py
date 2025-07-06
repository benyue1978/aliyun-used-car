import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold

# Define feature groups for clarity
LOW_CARDINALITY = ['notRepairedDamage', 'bodyType', 'fuelType', 'gearbox']
HIGH_CARDINALITY = ['model', 'brand', 'regionCode']

def extract_year(series):
    """Extracts the year from a date series, filling NaNs with the mode."""
    years = series.astype(str).str.extract(r'(\d{4})')[0]
    mode_year = years.dropna().mode()
    fill_year = mode_year.iloc[0] if not mode_year.empty else '2000'
    years = years.astype(int)
    return years

def extract_month(series):
    """Extracts the month from a date series, filling NaNs with the mode."""
    series = pd.to_numeric(series, errors='coerce').fillna(0).astype(int)
    months = series.astype(str).str.slice(4, 6).replace('', np.nan)
    mode_month = months.dropna().mode()
    fill_month = mode_month.iloc[0] if not mode_month.empty else '01'
    months = months.fillna(fill_month)
    return months.astype(int)

def clean_not_repaired_damage(val):
    """Cleans and standardizes the 'notRepairedDamage' column."""
    try:
        num_val = float(val)
        if num_val == 0.0:
            return 'no_damage'
        elif num_val == 1.0:
            return 'has_damage'
        else:
            return 'unknown'
    except (ValueError, TypeError):
        return 'unknown'

class FeatureGenerator:
    def __init__(self):
        self.onehot_encoders = {}
        self.scaler = None
        self.fitted = False
        self.stat_cols = []
        self.stat_maps = {}
        self.model_target_mean = None
        self.global_price_mean = None
        self.num_cols = []
        self.name_counts = {}
        self.power_cap = None  # For dynamic power clipping

    def fit(self, df):
        """Fits the feature generator on the training data."""
        # === Part 1: Generate features and fit mappings ===
        df['notRepairedDamage'] = df['notRepairedDamage'].astype(str).apply(clean_not_repaired_damage)
        
        # Dynamically cap power based on 99th percentile
        self.power_cap = df['power'].quantile(0.99)
        df['power'] = df['power'].clip(0, self.power_cap)
        df['power_log'] = np.log1p(df['power'])

        reg_year = extract_year(df['regDate'])
        creat_year = extract_year(df['creatDate'])
        df['car_age'] = (creat_year - reg_year).clip(lower=0)
        df['creatDate_month'] = extract_month(df['creatDate'])
        
        self.name_counts = df['name'].value_counts().to_dict()
        df['name_count'] = df['name'].map(self.name_counts)

        # Interaction features
        df['power_x_kilometer'] = df['power'] * df['kilometer']
        df['car_age_x_kilometer'] = df['car_age'] * df['kilometer']
        df['power_x_car_age'] = df['power'] * df['car_age']

        # === Part 2: Fit encoders and statistical mappings ===
        self.global_price_mean = df['price'].mean()
        
        # Target encoding for 'model' with KFold to prevent leakage
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        df['model_target_mean'] = 0
        for train_idx, val_idx in kf.split(df):
            m = df.iloc[train_idx].groupby('model')['price'].mean()
            df.loc[df.index[val_idx], 'model_target_mean'] = df.iloc[val_idx]['model'].map(m)
        
        # Global target mean for 'model' for use in transform
        self.model_target_mean = df.groupby('model')['price'].agg(lambda x: (x.sum() + self.global_price_mean) / (len(x) + 1)).to_dict()

        # Statistical features for high cardinality columns
        for col in HIGH_CARDINALITY:
            stat_df = df.groupby(col)['price'].agg(
                mean=lambda x: (x.sum() + self.global_price_mean) / (len(x) + 1),
                median=lambda x: (x.median() + self.global_price_mean) / (len(x) + 1),
                count='count'
            ).reset_index()
            self.stat_maps[col] = stat_df.set_index(col)[['mean', 'median', 'count']].to_dict(orient='index')
            self.stat_cols.extend([f'{col}_price_mean', f'{col}_price_median', f'{col}_count'])

        # Fit OneHotEncoder for low cardinality columns
        for col in LOW_CARDINALITY:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe.fit(df[[col]].astype(str))
            self.onehot_encoders[col] = ohe

        # === Part 3: Fit scaler on all generated numerical columns ===
        df_transformed = self.transform(df.copy(), is_fitting=True)
        self.num_cols = [col for col in df_transformed.columns if df_transformed[col].dtype != 'object' and col not in ['SaleID', 'price']]
        
        self.scaler = StandardScaler()
        self.scaler.fit(df_transformed[self.num_cols])
        self.fitted = True

    def transform(self, df, is_fitting=False):
        """Transforms the data using the fitted generator."""
        if not is_fitting:
            assert self.fitted, 'Call fit() first!'

        # === Part 1: Basic transformations ===
        df['notRepairedDamage'] = df['notRepairedDamage'].astype(str).apply(clean_not_repaired_damage)
        df['power'] = df['power'].clip(0, self.power_cap)
        df['power_log'] = np.log1p(df['power'])
        
        reg_year = extract_year(df['regDate'])
        creat_year = extract_year(df['creatDate'])
        df['car_age'] = (creat_year - reg_year).clip(lower=0)
        df['creatDate_month'] = extract_month(df['creatDate'])
        df['name_count'] = df['name'].map(self.name_counts).fillna(0)

        # Interaction features
        df['power_x_kilometer'] = df['power'] * df['kilometer']
        df['car_age_x_kilometer'] = df['car_age'] * df['kilometer']
        df['power_x_car_age'] = df['power'] * df['car_age']

        # === Part 2: Apply pre-computed mappings ===
        fill_value = df['price'].mean() if 'price' in df.columns else self.global_price_mean
        df['model_target_mean'] = df['model'].map(self.model_target_mean).fillna(fill_value)

        for col, stat_map in self.stat_maps.items():
            df[f'{col}_price_mean'] = df[col].map(lambda x: stat_map.get(x, {}).get('mean')).fillna(self.global_price_mean)
            df[f'{col}_price_median'] = df[col].map(lambda x: stat_map.get(x, {}).get('median')).fillna(self.global_price_mean)
            df[f'{col}_count'] = df[col].map(lambda x: stat_map.get(x, {}).get('count')).fillna(0)

        # === Part 3: Apply encoders ===
        onehot_dfs = []
        for col in LOW_CARDINALITY:
            ohe = self.onehot_encoders[col]
            arr = ohe.transform(df[[col]].astype(str))
            names = ohe.get_feature_names_out([col])
            onehot_dfs.append(pd.DataFrame(arr, columns=names, index=df.index))
        onehot_df = pd.concat(onehot_dfs, axis=1)

        df = pd.concat([df, onehot_df], axis=1)

        # === Part 4: Apply scaler and finalize columns ===
        if self.scaler:
            num_cols_to_scale = [col for col in self.num_cols if col in df.columns]
            scaled_data = self.scaler.transform(df[num_cols_to_scale])
            df[num_cols_to_scale] = scaled_data

        return df