import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# List of categorical and numerical features
CATEGORICAL_FEATURES = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode', 'seller', 'offerType']
NUMERICAL_FEATURES = ['power', 'kilometer'] + [f'v_{i}' for i in range(15)]

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
        self.scaler = None
        self.fitted = False

    def fit(self, df):
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].fillna('-1').astype(str)
            le = LabelEncoder()
            # Ensure '-1' is always in classes
            le.fit(list(df[col].unique()) + ['-1'])
            self.label_encoders[col] = le
        for col in NUMERICAL_FEATURES:
            df[col] = df[col].fillna(0)
        reg_year = extract_year(df['regDate'])
        creat_year = extract_year(df['creatDate'])
        df['car_age'] = (creat_year - reg_year).clip(lower=0)
        df['kilometer_bin'] = pd.cut(df['kilometer'], bins=[-1,5,10,15,20], labels=[0,1,2,3])
        df['kilometer_bin'] = df['kilometer_bin'].cat.add_categories([-1]).fillna(-1).astype(int)
        self.scaler = StandardScaler()
        self.scaler.fit(df[NUMERICAL_FEATURES + ['car_age']])
        self.fitted = True

    def transform(self, df):
        assert self.fitted, 'Call fit() first!'
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].fillna('-1').astype(str)
            le = self.label_encoders[col]
            # If unseen label, set to '-1'
            df[col] = df[col].where(df[col].isin(le.classes_), '-1')
            df[col] = le.transform(df[col])
        for col in NUMERICAL_FEATURES:
            df[col] = df[col].fillna(0)
        reg_year = extract_year(df['regDate'])
        creat_year = extract_year(df['creatDate'])
        df['car_age'] = (creat_year - reg_year).clip(lower=0)
        df['kilometer_bin'] = pd.cut(df['kilometer'], bins=[-1,5,10,15,20], labels=[0,1,2,3])
        df['kilometer_bin'] = df['kilometer_bin'].cat.add_categories([-1]).fillna(-1).astype(int)
        scaled = self.scaler.transform(df[NUMERICAL_FEATURES + ['car_age']])
        for i, col in enumerate(NUMERICAL_FEATURES + ['car_age']):
            df[col+'_scaled'] = scaled[:,i]
        return df

# Usage example:
# fg = FeatureGenerator()
# fg.fit(train_df)
# train_df = fg.transform(train_df)
# test_df = fg.transform(test_df) 