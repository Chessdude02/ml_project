import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class AvocadoDataLoader:
    def __init__(self, data_url='https://gist.githubusercontent.com/gkroiz/81019af6ccdbd5e97bb4a0ef5eaa0c7d/raw/4f42e85e6b2e81b98f5e7edcd4d6f7d2e70c7c9c/avocado.csv'):
        self.data_url = data_url
        
    def load_data(self) -> pd.DataFrame:
        """Load avocado price data from URL or local file"""
        try:
            print("Loading avocado dataset...")
            df = pd.read_csv(self.data_url)
            print(f"Loaded {len(df)} records")
            return df
        except Exception as e:
            raise IOError(f"Failed to load data: {str(e)}")
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess avocado data"""
        df = df.copy()
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Rename columns for consistency
        df = df.rename(columns={
            'Date': 'date',
            'AveragePrice': 'price',
            'Total Volume': 'total_volume',
            'type': 'avocado_type',
            'year': 'year',
            'region': 'region'
        })
        
        # Focus on conventional avocados nationally (for simpler analysis)
        df_filtered = df[(df['avocado_type'] == 'conventional') & (df['region'] == 'TotalUS')].copy()
        
        # Add time-based features
        df_filtered['day_of_week'] = df_filtered['date'].dt.dayofweek
        df_filtered['month'] = df_filtered['date'].dt.month
        df_filtered['is_weekend'] = (df_filtered['date'].dt.dayofweek >= 5).astype(int)
        df_filtered['quarter'] = df_filtered['date'].dt.quarter
        
        # Select relevant columns
        cols = ['date', 'price', 'total_volume', 'day_of_week', 'month', 'is_weekend', 'quarter']
        df_filtered = df_filtered[cols].reset_index(drop=True)
        
        print(f"Preprocessed data: {len(df_filtered)} records")
        print(f"Date range: {df_filtered['date'].min()} to {df_filtered['date'].max()}")
        
        return df_filtered


class FeatureEngineer:
    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add lag and rolling features for time series prediction"""
        if 'price' not in df.columns:
            raise ValueError("DataFrame must contain price column")
        
        df = df.copy()
        
        # Lag features
        df['price_lag_1'] = df['price'].shift(1)
        df['price_lag_7'] = df['price'].shift(7)
        df['price_lag_30'] = df['price'].shift(30)
        
        # Rolling mean features
        df['price_rolling_mean_7'] = df['price'].rolling(window=7).mean()
        df['price_rolling_std_7'] = df['price'].rolling(window=7).std()
        df['price_rolling_mean_30'] = df['price'].rolling(window=30).mean()
        
        # Volume lag features
        df['volume_lag_1'] = df['total_volume'].shift(1)
        df['volume_rolling_mean_7'] = df['total_volume'].rolling(window=7).mean()
        
        # Drop rows with NaN values
        df = df.dropna()
        
        print(f"Features added. Final dataset: {len(df)} records")
        
        return df
    
    @staticmethod
    def export_data(df: pd.DataFrame, filepath: str):
        """Export data to CSV"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filepath, index=False)
            print(f"Exported to: {filepath}")
        except Exception as e:
            raise IOError(f"Cannot write to {filepath}: {str(e)}")


if __name__ == "__main__":
    # Load and preprocess data
    raw_data = pd.read_csv('C:\\Users\\vedan\\PythonProject11\\Avocado.csv')
    loader = AvocadoDataLoader()
    processed_data = loader.preprocess_data(raw_data)
    
    # Add features
    engineer = FeatureEngineer()
    featured_data = engineer.add_features(processed_data)
    
    # Export
    engineer.export_data(processed_data, 'avocado_src/data/raw_avocado.csv')
    engineer.export_data(featured_data, 'avocado_src/data/avocado_with_features.csv')
    
    print("\nData loading complete!")
