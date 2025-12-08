import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

class ModelConfig:
    feature_cols = [
        'day_of_week', 'month', 'is_weekend', 'quarter',
        'total_volume', 'price_lag_1', 'price_lag_7', 'price_lag_30',
        'price_rolling_mean_7', 'price_rolling_std_7',
        'price_rolling_mean_30', 'volume_lag_1', 'volume_rolling_mean_7'
    ]

    rf_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1
    }

    gb_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': 42
    }

    # Hyperparameter tuning grids
    rf_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    gb_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'min_samples_split': [2, 5, 10]
    }

    ARIMA_ORDER = (2, 1, 2)


class ModelEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred, model_name):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }


class BaseModelTrainer:
    def __init__(self, test_size=0.2):
        self.test_size = test_size
        self.evaluator = ModelEvaluator()

    def split_data(self, df):
        split_idx = int(len(df) * (1 - self.test_size))
        train_df = df[:split_idx]
        test_df = df[split_idx:]
        return train_df, test_df


class RandomForestTrainer(BaseModelTrainer):
    def train(self, X_train, y_train, X_test, y_test, tune_hyperparams=False):
        if tune_hyperparams:
            print("  Tuning Random Forest hyperparameters...")
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            random_search = RandomizedSearchCV(
                base_model,
                param_distributions=ModelConfig.rf_param_grid,
                n_iter=20,
                cv=3,
                scoring='neg_mean_absolute_error',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            random_search.fit(X_train, y_train)
            model = random_search.best_estimator_
            print(f"  Best params: {random_search.best_params_}")
        else:
            model = RandomForestRegressor(**ModelConfig.rf_params)
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = self.evaluator.evaluate(y_test, y_pred, 'Random Forest')
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        return model, y_pred, metrics, feature_importance


class GradientBoostingTrainer(BaseModelTrainer):
    def train(self, X_train, y_train, X_test, y_test, tune_hyperparams=False):
        if tune_hyperparams:
            print("  Tuning Gradient Boosting hyperparameters...")
            base_model = GradientBoostingRegressor(random_state=42)
            random_search = RandomizedSearchCV(
                base_model,
                param_distributions=ModelConfig.gb_param_grid,
                n_iter=20,
                cv=3,
                scoring='neg_mean_absolute_error',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            random_search.fit(X_train, y_train)
            model = random_search.best_estimator_
            print(f"  Best params: {random_search.best_params_}")
        else:
            model = GradientBoostingRegressor(**ModelConfig.gb_params)
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = self.evaluator.evaluate(y_test, y_pred, 'Gradient Boosting')
        return model, y_pred, metrics


class ProphetTrainer(BaseModelTrainer):
    def train(self, df):
        train_df, test_df = self.split_data(df)
        prophet_train = pd.DataFrame({
            'ds': pd.to_datetime(train_df['date']),
            'y': train_df['price']
        })
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        model.fit(prophet_train)
        prophet_test = pd.DataFrame({'ds': pd.to_datetime(test_df['date'])})
        forecast = model.predict(prophet_test)
        y_pred = forecast['yhat'].values
        y_test = test_df['price'].values
        metrics = self.evaluator.evaluate(y_test, y_pred, 'Prophet')
        return model, y_pred, metrics


class ARIMATrainer(BaseModelTrainer):
    def train(self, df):
        train_df, test_df = self.split_data(df)
        model = ARIMA(train_df['price'], order=ModelConfig.ARIMA_ORDER)
        model_arima = model.fit()
        y_pred = model_arima.forecast(steps=len(test_df))
        y_test = test_df['price'].values
        metrics = self.evaluator.evaluate(y_test, y_pred, 'ARIMA')
        return model_arima, y_pred, metrics


class TrainingPipeline:
    def __init__(self, data_path, tune_hyperparams=False):
        self.data_path = data_path
        self.tune_hyperparams = tune_hyperparams
        self.results = []
        self.predictions = {}
        self.best_model = None
        self.feature_importance = None

    def load_data(self):
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def run(self):
        df = self.load_data()
        X = df[ModelConfig.feature_cols]
        y = df['price']
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_gradient_boosting(X_train, y_train, X_test, y_test)
        self.train_prophet(df)
        self.train_arima(df)
        results_df = pd.DataFrame(self.results)
        self.best_model = results_df.loc[results_df['MAE'].idxmin(), 'Model']
        return results_df

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        trainer = RandomForestTrainer()
        model, pred, metrics, importance = trainer.train(X_train, y_train, X_test, y_test, self.tune_hyperparams)
        self.results.append(metrics)
        self.predictions['Random Forest'] = pred
        self.feature_importance = importance

    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        trainer = GradientBoostingTrainer()
        model, pred, metrics = trainer.train(X_train, y_train, X_test, y_test, self.tune_hyperparams)
        self.results.append(metrics)
        self.predictions['Gradient Boosting'] = pred

    def train_prophet(self, df):
        trainer = ProphetTrainer()
        model, pred, metrics = trainer.train(df)
        self.results.append(metrics)
        self.predictions['Prophet'] = pred

    def train_arima(self, df):
        trainer = ARIMATrainer()
        model, pred, metrics = trainer.train(df)
        self.results.append(metrics)
        self.predictions['ARIMA'] = pred


if __name__ == "__main__":
    import sys
    tune = '--tune' in sys.argv or '-t' in sys.argv
    
    if tune:
        print("Running with hyperparameter tuning...")
    
    pipeline = TrainingPipeline("avocado_src/data/avocado_with_features.csv", tune_hyperparams=tune)
    results = pipeline.run()
    print("\n" + results.to_string(index=False))
