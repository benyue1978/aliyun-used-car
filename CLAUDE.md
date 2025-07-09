# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project for predicting used car prices from the Aliyun (Tianchi) competition. The project uses ensemble methods combining XGBoost and LightGBM models with feature engineering to predict car prices.

## Key Commands

### Running the Project

```bash
# Activate Python virtual environment
source venv/bin/activate

# Install dependencies (includes lightgbm which is imported in main.py)
pip install -r code/requirements.txt
pip install lightgbm

# Run main prediction pipeline (trains models and generates predictions)
python code/main.py

# Alternative: run via shell script
bash code/main.sh
```

### Data Analysis

```bash
# Run exploratory data analysis
python code/eda.py

# Generate insights on raw data
python code/raw_data_eda.py
```

## Code Architecture

### Core Components

1. **Feature Engineering** (`feature/generation.py`):
   - `FeatureGenerator` class: Handles all feature transformations
   - Supports both categorical (OneHot encoding) and numerical features
   - Target encoding for high-cardinality features (model, brand, regionCode)
   - Statistical features (mean, median, count) for categorical variables
   - Interaction features (power×kilometer, car_age×kilometer, power×car_age)

2. **Main Pipeline** (`code/main.py`):
   - Data loading with custom CSV parser (`read_csv_correctly`)
   - Feature generation and transformation using FeatureGenerator
   - Model training: XGBoost and LightGBM with early stopping
   - Model validation and evaluation (MAE on both log and original scales)
   - Ensemble prediction blending (0.7 XGBoost + 0.3 LightGBM)
   - Model persistence (saves trained models to `model/` directory)
   - Final prediction output to `prediction_result/predictions_blended.csv`

3. **Data Analysis** (`code/eda.py`):
   - Exploratory data analysis with visualizations
   - Feature distribution analysis
   - Correlation analysis
   - Target variable analysis

### Data Structure

- **Training Data**: `data/used_car_train_20200313.csv`
- **Test Data A**: `data/used_car_testA_20200313.csv`
- **Test Data B**: `data/used_car_testB_20200421.csv`
- **Trained Models**: `model/xgboost_model.pkl`, `model/lightgbm_model.pkl` (generated after training)
- **Predictions**: `prediction_result/predictions_blended.csv` (final ensemble output)

### Key Features

The dataset contains 22 features including:

- **Categorical**: model, brand, bodyType, fuelType, gearbox, notRepairedDamage, regionCode
- **Numerical**: power, kilometer, v_0 through v_14 (anonymous features)
- **Date**: regDate (registration date), creatDate (creation date)
- **Target**: price (log-transformed for training)

### Model Pipeline

1. **Data Preprocessing**: Custom CSV reader handles space-separated format
2. **Data Splitting**: 90% train, 10% validation split for model evaluation
3. **Feature Engineering**: FeatureGenerator creates ~100+ features from raw data
4. **Model Training**:
   - XGBoost with early stopping (100,000 max estimators, 0.01 learning rate)
   - LightGBM with early stopping (100,000 max estimators, 0.01 learning rate)
5. **Model Evaluation**: MAE calculated on both log-scale and original price scale
6. **Ensemble**: Fixed 0.7/0.3 weighted blend of XGBoost and LightGBM predictions
7. **Model Persistence**: Trained models saved to `model/` directory using joblib
8. **Output**: Single blended prediction file for test set B

### Dependencies

Core libraries: pandas, scikit-learn, torch, xgboost, lightgbm, joblib (see `code/requirements.txt`)

## Important Notes

- The project uses log-transformed prices for training and predictions
- Feature engineering includes K-fold target encoding to prevent overfitting
- All relative paths are structured from the project root
- Models are trained from scratch each run with early stopping (100 rounds patience)
- Training uses a 90/10 train/validation split for evaluation
- Final ensemble uses a fixed 0.7/0.3 blend ratio optimized for this dataset
- Training time is significant (~100,000 max estimators with 0.01 learning rate)
- LightGBM dependency must be installed separately (`pip install lightgbm`)
