import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from models.trainer import TrainingPipeline, ModelConfig
from utils.graphs import ModelVisualizer

def save_model(model, filename, folder='saved_models'):
    Path(folder).mkdir(parents=True, exist_ok=True)
    filepath = Path(folder) / filename
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved: {filepath}")

def main():
    # Load data
    df = pd.read_csv('data/prices_with_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Prepare train/test split
    X = df[ModelConfig.feature_cols]
    y = df['price']
    split_idx = int(len(df) * 0.8)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    test_dates = df['date'][split_idx:].reset_index(drop=True)
    
    # Initialize pipeline
    pipeline = TrainingPipeline('data/prices_with_features.csv')
    
    # Train all models
    print("\n=== Training Models ===")
    from models.trainer import RandomForestTrainer, GradientBoostingTrainer, ProphetTrainer, ARIMATrainer
    
    # Random Forest
    print("\nTraining Random Forest...")
    rf_trainer = RandomForestTrainer()
    rf_model, rf_pred, rf_metrics, rf_importance = rf_trainer.train(X_train, y_train, X_test, y_test)
    save_model(rf_model, 'random_forest.pkl')
    
    # Gradient Boosting
    print("Training Gradient Boosting...")
    gb_trainer = GradientBoostingTrainer()
    gb_model, gb_pred, gb_metrics = gb_trainer.train(X_train, y_train, X_test, y_test)
    save_model(gb_model, 'gradient_boosting.pkl')
    
    # Prophet
    print("Training Prophet...")
    prophet_trainer = ProphetTrainer()
    prophet_model, prophet_pred, prophet_metrics = prophet_trainer.train(df)
    save_model(prophet_model, 'prophet.pkl')
    
    # ARIMA
    print("Training ARIMA...")
    arima_trainer = ARIMATrainer()
    arima_model, arima_pred, arima_metrics = arima_trainer.train(df)
    save_model(arima_model, 'arima.pkl')
    
    # Collect results and predictions
    results_df = pd.DataFrame([rf_metrics, gb_metrics, prophet_metrics, arima_metrics])
    predictions = {
        'Random Forest': rf_pred,
        'Gradient Boosting': gb_pred,
        'Prophet': prophet_pred,
        'ARIMA': arima_pred
    }
    
    print("\n=== Model Performance ===")
    print(results_df.to_string(index=False))
    print(f"\nBest Model: {results_df.loc[results_df['MAE'].idxmin(), 'Model']}")
    
    # Generate all visualizations
    print("\n=== Generating Visualizations ===")
    viz = ModelVisualizer(output_dir='outputs')
    
    print("Creating prediction comparison plot...")
    viz.plot_predictions_comparison(y_test.values, predictions, test_dates)
    
    print("Creating metrics comparison plot...")
    viz.plot_metrics_comparison(results_df)
    
    print("Creating feature importance plot...")
    viz.plot_feature_importance(rf_importance)
    
    print("Creating time series plot...")
    viz.plot_time_series(df)
    
    print("Creating residuals plot...")
    viz.plot_residuals(y_test.values, predictions)
    
    print("Creating model ranking table...")
    viz.plot_model_ranking_table(results_df)
    
    print("Creating error distribution plot...")
    viz.plot_error_distribution(y_test.values, predictions)
    
    print("Creating correlation heatmap...")
    viz.plot_correlation_heatmap(df)
    
    print("Creating rolling RMSE plot...")
    viz.plot_rolling_rmse(y_test.values, predictions)
    
    print("Creating interactive predictions plot...")
    viz.plot_interactive_predictions(y_test.values, predictions, test_dates)
    
    print("\n=== Complete ===")
    print("Models saved in: saved_models/")
    print("Graphs saved in: outputs/")

if __name__ == "__main__":
    main()

