import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class ModelVisualizer:
    def __init__(self, output_dir: str = 'outputs'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_predictions_comparison(self,y_test: np.ndarray,predictions: Dict[str, np.ndarray],dates: Optional[pd.Series] = None) -> str:
        fig, ax = plt.subplots(figsize=(14, 7))
        x = dates if dates is not None else range(len(y_test))
        ax.plot(x, y_test, 'k-', linewidth=2, label='Actual', alpha=0.7)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, (model_name, preds) in enumerate(predictions.items()):
            ax.plot(x,preds,'-',linewidth=1.5,label=model_name,color=colors[i % len(colors)],alpha=0.8)
        ax.set_xlabel('Date' if dates is not None else 'Sample', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_title('Price Predictions Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        if dates is not None:
            plt.xticks(rotation=45)
        plt.tight_layout()
        output_path = self.output_dir / 'predictions_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)

    def plot_metrics_comparison(self, results_df: pd.DataFrame) -> str:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = ['MAE', 'RMSE', 'R2', 'MAPE']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for ax, metric, color in zip(axes.flat, metrics, colors):
            values = results_df[metric]
            models = results_df['Model']
            bars = ax.barh(models, values, color=color, alpha=0.7, edgecolor='black')
            ax.set_xlabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'Model Comparison: {metric}', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            for bar in bars:
                width = bar.get_width()
                ax.text(width,bar.get_y() + bar.get_height() / 2,f'{width:.2f}',ha='left',va='center',fontsize=9,fontweight='bold')
        plt.tight_layout()
        output_path = self.output_dir / 'metrics_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)

    def plot_feature_importance(self, importance_df, top_n=10):
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = importance_df.head(top_n)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        bars = ax.barh(top_features['feature'],top_features['importance'],color=colors,edgecolor='black',alpha=0.8)
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title('Top Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2,f'{width:.3f}', ha='left', va='center',fontsize=9, fontweight='bold')
        plt.tight_layout()
        output_path = self.output_dir / 'feature_importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)

    def plot_time_series(self, df):
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df['date'], df['price'], linewidth=2, alpha=0.8)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax.set_title('Price Time Series', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        output_path = self.output_dir / 'time_series.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)

    def plot_residuals(self, y_test, predictions: Dict[str, np.ndarray]):
        n_models = len(predictions)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models))
        if n_models == 1:
            axes = [axes]
        for ax, (model_name, preds) in zip(axes, predictions.items()):
            residuals = y_test - preds
            ax.scatter(range(len(residuals)), residuals, alpha=0.6, s=30)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax.set_xlabel('Sample', fontsize=11)
            ax.set_ylabel('Residual', fontsize=11)
            ax.set_title(f'Residuals: {model_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = self.output_dir / 'residuals.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)

    def plot_model_ranking_table(self, results_df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        table = ax.table(cellText=results_df.values,colLabels=results_df.columns,cellLoc='center',loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        output_path = self.output_dir / 'model_ranking_table.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)

    def plot_error_distribution(self, y_test, predictions: Dict[str, np.ndarray]) -> str:
        fig, ax = plt.subplots(figsize=(12, 6))
        for model_name, preds in predictions.items():
            errors = y_test - preds
            sns.histplot(errors, kde=True, bins=30, label=model_name, alpha=0.5, ax=ax)
        ax.set_xlabel('Prediction Error', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        output_path = self.output_dir / 'error_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> str:
        plt.figure(figsize=(10, 8))
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
        output_path = self.output_dir / 'correlation_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)

    def plot_rolling_rmse(self, y_test, predictions: Dict[str, np.ndarray], window: int = 7) -> str:
        import pandas as pd
        fig, ax = plt.subplots(figsize=(14, 6))
        for model_name, preds in predictions.items():
            errors = (y_test - preds) ** 2
            rolling_rmse = pd.Series(errors).rolling(window=window).mean().apply(np.sqrt)
            ax.plot(rolling_rmse, linewidth=2, label=model_name)
        ax.set_xlabel('Sample', fontsize=12)
        ax.set_ylabel('Rolling RMSE', fontsize=12)
        ax.set_title(f'Rolling RMSE (window={window})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        output_path = self.output_dir / 'rolling_rmse.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)

    def plot_interactive_predictions(self, y_test, predictions: Dict[str, np.ndarray], dates=None) -> str:
        import plotly.graph_objs as go
        import plotly.offline as pyo

        x = dates if dates is not None else list(range(len(y_test)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y_test, mode='lines', name='Actual'))
        for model_name, preds in predictions.items():
            fig.add_trace(go.Scatter(x=x, y=preds, mode='lines', name=model_name))
        fig.update_layout(
            title='Interactive Predictions Comparison',
            xaxis_title='Date' if dates is not None else 'Sample',
            yaxis_title='Price',
            template='plotly_white'
        )
        output_path = str(self.output_dir / 'interactive_predictions.html')
        pyo.plot(fig, filename=output_path, auto_open=False)
        return output_path


class DataAnalyzer:
    @staticmethod
    def generate_summary_stats(df):
        stats = df.describe()
        stats.loc['median'] = df.median()
        stats.loc['skewness'] = df.skew()
        stats.loc['kurtosis'] = df.kurtosis()
        return stats.round(3)

    @staticmethod
    def generate_outliers(series, threshold=3.0):
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
