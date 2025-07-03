import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
from feature.generation import FeatureGenerator

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data paths
train_path = os.path.join('data', 'used_car_train_20200313.csv')
testA_path = os.path.join('data', 'used_car_testA_20200313.csv')
testB_path = os.path.join('data', 'used_car_testB_20200421.csv')

# Features (will be generated automatically)
# CATEGORICAL_FEATURES, NUMERICAL_FEATURES are defined in feature/generation.py

# Data loading with auto separator and missing value handling
def load_data(path, is_train=True):
    try:
        df = pd.read_csv(path, na_values=['-'])
        if len(df.columns) == 1:
            raise ValueError('Single column detected, try whitespace separator')
    except Exception:
        try:
            df = pd.read_csv(path, delim_whitespace=True, na_values=['-'])
            if len(df.columns) == 1:
                raise ValueError('Single column detected, try tab separator')
        except Exception:
            df = pd.read_csv(path, sep='\t', na_values=['-'])
    df.columns = df.columns.str.strip()
    print(f"[DEBUG] Columns in {path}: {df.columns.tolist()}")
    return df

class CarDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y) if y is not None else None
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x).squeeze(1)

def train_model(model, train_loader, epochs=10, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader.dataset):.4f}")

def predict(model, test_loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            output = model(X_batch)
            preds.append(output.cpu().numpy())
    return np.concatenate(preds)

def save_submission(sale_ids, preds, out_path):
    df = pd.DataFrame({'SaleID': sale_ids, 'price': preds.astype(int)})
    df.to_csv(out_path, index=False)

if __name__ == '__main__':
    # Load raw data
    train_df = load_data(train_path, is_train=True)
    testA_df = load_data(testA_path, is_train=False)
    testB_df = load_data(testB_path, is_train=False)

    # Feature engineering
    fg = FeatureGenerator()
    fg.fit(train_df)
    train_df = fg.transform(train_df)
    testA_df = fg.transform(testA_df)
    testB_df = fg.transform(testB_df)

    # Select all scaled numerical features and new features for modeling
    feature_cols = [col for col in train_df.columns if col.endswith('_scaled') or col in ['kilometer_bin'] + ['car_age']]
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['price'].values.astype(np.float32)
    X_testA = testA_df[feature_cols].values.astype(np.float32)
    X_testB = testB_df[feature_cols].values.astype(np.float32)

    # Prepare dataset
    train_dataset = CarDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Define model
    model = MLP(input_dim=len(feature_cols)).to(DEVICE)

    # Train
    train_model(model, train_loader, epochs=10, lr=1e-3)

    # Save model
    torch.save(model.state_dict(), os.path.join('model', 'mlp.pth'))

    # Predict testA
    saleidA = testA_df['SaleID']
    testA_dataset = CarDataset(X_testA)
    testA_loader = DataLoader(testA_dataset, batch_size=256, shuffle=False)
    predsA = predict(model, testA_loader)
    save_submission(saleidA, predsA, os.path.join('prediction_result', 'predictions.csv'))

    # Calculate MAE for testA if ground truth exists
    testA_label_path = os.path.join('data', 'used_car_testA_20200313_label.csv')
    if os.path.exists(testA_label_path):
        testA_label_df = pd.read_csv(testA_label_path)
        y_true_A = testA_label_df['price'].values
        mae_A = mean_absolute_error(y_true_A, predsA)
        print(f"TestA MAE: {mae_A:.4f}")
    else:
        print("TestA ground truth not found, skip MAE calculation.")

    # Predict testB
    saleidB = testB_df['SaleID']
    testB_dataset = CarDataset(X_testB)
    testB_loader = DataLoader(testB_dataset, batch_size=256, shuffle=False)
    predsB = predict(model, testB_loader)
    save_submission(saleidB, predsB, os.path.join('prediction_result', 'predictions_B.csv'))

    # Calculate MAE for testB if ground truth exists
    testB_label_path = os.path.join('data', 'used_car_testB_20200421_label.csv')
    if os.path.exists(testB_label_path):
        testB_label_df = pd.read_csv(testB_label_path)
        y_true_B = testB_label_df['price'].values
        mae_B = mean_absolute_error(y_true_B, predsB)
        print(f"TestB MAE: {mae_B:.4f}")
    else:
        print("TestB ground truth not found, skip MAE calculation.")

    print('All predictions finished!')
