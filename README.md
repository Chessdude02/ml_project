# Avocado Price Prediction

Machine learning models to predict avocado prices using time series analysis and feature engineering.

## Project Structure

```
avocado_src/
├── data/
│   ├── loader.py                  # Data loader and preprocessor
│   ├── raw_avocado.csv            # Preprocessed data
│   └── avocado_with_features.csv  # Data with engineered features
├── models/
│   └── trainer.py                 # Model training pipeline
├── utils/
│   └── graphs.py                  # Visualization utilities
├── saved_models/                  # Trained model files
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── prophet.pkl
│   └── arima.pkl
├── outputs/                       # Generated visualizations
│   ├── predictions_comparison_h.png
│   ├── metrics_comparison_h.png
│   ├── feature_importance_h.png
│   ├── time_series_h.png
│   ├── residuals_h.png
│   ├── model_ranking_table_h.png
│   ├── error_distribution_h.png
│   ├── correlation_heatmap_h.png
│   ├── rolling_rmse_h.png
│   └── interactive_predictions_h.html
└── main.py                        # Main execution script
```

## Dataset

The dataset uses real historical avocado prices from the Hass Avocado Board (2015-2018).
It contains weekly avocado price data for conventional avocados in the TotalUS region with the following features:
- **Date**: Week of observation (169 weeks, 139 after feature engineering)
- **Price**: Average avocado price ($0.71 - $2.02)
- **Total Volume**: Total volume sold
- **Time features**: day_of_week, month, quarter, is_weekend
- **Lag features**: price_lag_1, price_lag_7, price_lag_30
- **Rolling features**: rolling means and standard deviations
- **Volume features**: volume lags and rolling means

## Models

Four machine learning models are trained and compared:

1. **Random Forest Regressor**
2. **Gradient Boosting Regressor**
3. **Facebook Prophet**
4. **ARIMA**

## Results (with Hyperparameter Tuning)

| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|----|----|
| **Gradient Boosting** ⭐ | 0.092 | 0.117 | 0.715 | 8.07% |
| Random Forest | 0.093 | 0.116 | 0.717 | 7.70% |
| Prophet | 0.210 | 0.227 | -0.081 | 18.52% |
| ARIMA | 0.378 | 0.421 | -2.71 | 34.99% |

**Best Model**: Gradient Boosting with MAE of $0.09 and R² of 0.72

### Optimal Hyperparameters

**Random Forest:**
- n_estimators: 100
- max_depth: 15
- min_samples_split: 10
- min_samples_leaf: 1
- max_features: None

**Gradient Boosting:**
- n_estimators: 200
- max_depth: 5
- learning_rate: 0.2
- subsample: 0.6
- min_samples_split: 10

## Usage

### 1. Preprocess data:
```bash
python avocado_src/data/loader.py
```

### 2. Train models (without tuning):
```bash
python avocado_src/main.py
```

### 3. Train with hyperparameter tuning:
```bash
python avocado_src/main.py --tune
```

### 4. Train models only:
```bash
python avocado_src/models/trainer.py --tune
```

## Features

- ✅ Real avocado price data preprocessing and feature engineering
- ✅ Multiple ML models (Random Forest, Gradient Boosting, Prophet, ARIMA)
- ✅ Hyperparameter tuning with RandomizedSearchCV
- ✅ Comprehensive model evaluation (MAE, RMSE, R², MAPE)
- ✅ Model persistence (saved as .pkl files)
- ✅ 10 visualization types including interactive plots
- ✅ Feature importance analysis
- ✅ Time series decomposition

## Requirements

- pandas
- numpy
- scikit-learn
- prophet
- statsmodels
- matplotlib
- seaborn
- plotly

## Notes

- The dataset uses **real historical avocado prices** from C:\Users\vedan\PythonProject11\Avocado.csv
- Data is filtered for conventional avocados in TotalUS region
- Covers 169 weeks from 2015-01-04 to 2018-03-25
- All visualizations are saved with `_h` suffix when using hyperparameter tuning
- Models are saved in `saved_models/` directory
- Graphs are saved in `outputs/` directory
