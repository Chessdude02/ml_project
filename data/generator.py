import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

class PriceDataGenerator:
    def __init__(self, n_days=180, base_price=299, seed=42):
        if n_days <= 0 or base_price <= 0:
            raise ValueError("Both values need to be positive")
        self.n_days = n_days
        self.base_price = base_price
        self.seed = seed

    def generate(self) -> pd.DataFrame:
        np.random.seed(self.seed)
        random.seed(self.seed)
        end_date = datetime.now()
        # generate list of dates oldest -> newest
        dates = [end_date - timedelta(days=x) for x in range(self.n_days)][::-1]
        trend = self.generate_trend()
        seasonality = self.generate_seasonality()
        noise = self.generate_noise()
        prices = self.base_price + trend + seasonality + noise
        prices = self.holidays(prices)
        prices = self.volatility_spikes(prices)
        prices = self.launch_jump(prices)
        prices = self.weekend_discount(prices, dates)

        df = self.build_dataframe(dates, prices)
        return df

    def generate_trend(self):
        return np.linspace(0, -40, self.n_days)  # donward trend to display longterm price decline

    def generate_seasonality(self):
        weekly = 10 * np.sin(2 * np.pi * np.arange(self.n_days) / 7)
        monthly = 15 * np.sin(2 * np.pi * np.arange(self.n_days) / 30)
        return weekly + monthly  # weekly and monthly rhythgmic price fluctuation

    def generate_noise(self):
        return np.random.normal(0, 8, self.n_days)  # random day to day variation

    def holidays(self, prices: np.ndarray) -> np.ndarray:
        # random 10 holiday indices within valid range
        holiday = [random.randint(0, self.n_days - 1) for _ in range(10)]

        p = prices.copy()  # avoid modifying original array in-place
        for idx in holiday:
            if idx < len(p):
                p[idx] = p[idx] * 0.8
        return p

    def volatility_spikes(self, prices: np.ndarray) -> np.ndarray:
        # 5 random days with +/-20% spikes
        p = prices.copy()
        spikes = [random.randint(0, self.n_days - 1) for _ in range(5)]
        for idx in spikes:
            factor = 1 + random.uniform(-0.2, 0.2)
            p[idx] = p[idx] * factor
        return p

    def launch_jump(self, prices: np.ndarray) -> np.ndarray:
        # simulate product launch price jump at day 30
        p = prices.copy()
        launch_day = min(30, self.n_days - 1)
        p[launch_day:] += 20  # jump +20 from launch day onwards
        return p

    def weekend_discount(self, prices: np.ndarray, dates: list) -> np.ndarray:
        # 10% discount on weekends
        p = prices.copy()
        for i, d in enumerate(dates):
            if d.weekday() >= 5:  # Saturday or Sunday
                p[i] *= 0.9
        return p

    def build_dataframe(self, dates, prices: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame({
            'date': dates,
            'price': prices,
            'day_of_week': [d.weekday() for d in dates],
            'month': [d.month for d in dates],
            'is_weekend': [1 if d.weekday() >= 5 else 0 for d in dates]
        })
        return df

class FeatureEngineer:
    @staticmethod
    def add_features(df:pd.DataFrame)->pd.DataFrame:
        if 'price' not in df.columns:
            raise ValueError("DataFrame must contain price column")
        df=df.copy()
        df['price_lag_1']=df['price'].shift(1)
        df['price_lag_7'] = df['price'].shift(7)
        df['price_lag_30'] = df['price'].shift(30)
        df['price_rolling_mean_7']=df['price'].rolling(window=7).mean()
        df['price_rolling_std_7'] = df['price'].rolling(window=7).std()
        df['price_rolling_mean_30'] = df['price'].rolling(window=30).mean()
        df=df.dropna()
        return df
    @staticmethod
    def export_data(df:pd.DataFrame,filepath:str):
        try:
            Path(filepath).parent.mkdir(parents=True,exist_ok=True)
            df.to_csv(filepath,index=False)
        except Exception as e:
            raise IOError(f"Cannot write to {filepath}: {str(e)}")

if __name__=="__main__":
    generator=PriceDataGenerator(n_days=180)
    raw_data=generator.generate()
    engineer=FeatureEngineer()
    processed_data=engineer.add_features(raw_data)

    engineer.export_data(raw_data,'data/raw_prices.csv')
    engineer.export_data(processed_data,'data/prices_with_features.csv')


