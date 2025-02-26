# File: models.py
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
from collections import defaultdict
from utils import ModelCheckpoint, ModelVisualizer
import numpy as np
import pandas as pd
import os

warnings.filterwarnings('ignore')
class ModelVisualizer:
    def __init__(self, base_dir='model/forecasting/output'):
        self.base_dir = base_dir
        self.viz_dir = os.path.join(base_dir, 'model_visualizations')
        self._create_model_directories()
        
    def _create_model_directories(self):
        """Create directories for each model type"""
        model_types = [
            'Prophet',
            'Exponential_Smoothing',
            'Seasonal_Decompose',
            'Comparisons'
        ]
        for model_type in model_types:
            model_dir = os.path.join(self.viz_dir, model_type)
            os.makedirs(model_dir, exist_ok=True)
        
    def plot_forecasts(self, df, forecasts, model_name, link_name):
        """Plot forecasted values with actual data"""
        try:
            # Create a new figure for each plot
            plt.clf()
            plt.figure(figsize=(15, 8))
            
            # Plot data
            plt.plot(df['timestamp'], df['traffic_in'], label='Actual Traffic In', alpha=0.7)
            plt.plot(forecasts['timestamp'], forecasts['traffic_in'], label='Forecast Traffic In', alpha=0.7)
            
            plt.title(f'{model_name} - Traffic In Forecast for {link_name}')
            plt.legend()
            plt.xlabel('Timestamp')
            plt.ylabel('Traffic In')
            
            # Save and close
            model_dir = os.path.join(self.viz_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            plt.savefig(os.path.join(model_dir, f"{link_name}_{model_name}_forecast.png"))
            plt.close('all')  # Close all figures
        except Exception as e:
            print(f"Error plotting forecast for {model_name} on {link_name}: {str(e)}")
    
    def plot_model_comparison(self, results):
        """Plot model comparison in Comparisons directory"""
        try:
            # Clear any existing plots
            plt.clf()
            
            metrics = ['mse', 'mae']
            model_names = list(results.keys())
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Model Comparison Metrics')
            
            for idx, metric in enumerate(metrics):
                values = [results[model][metric] for model in model_names]
                sns.barplot(x=model_names, y=values, ax=axes[idx])
                axes[idx].set_title(metric.upper())
                axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Save and close
            comparison_dir = os.path.join(self.viz_dir, 'Comparisons')
            os.makedirs(comparison_dir, exist_ok=True)
            plt.savefig(os.path.join(comparison_dir, "model_comparison.png"))
            plt.close('all')  # Close all figures
        except Exception as e:
            print(f"Error plotting model comparison: {str(e)}")

class ModelComparison:
    def __init__(self):
        self.models = {
            'Prophet': self.prophet_forecasting,
            'Exponential_Smoothing': self.exp_smoothing_forecasting,
            'Seasonal_Decompose': self.seasonal_decompose_forecasting
        }
        self.results = {}
        self.detailed_results = defaultdict(dict)
        self.checkpointer = ModelCheckpoint()
        self.visualizer = ModelVisualizer()
        
        # Set matplotlib backend at initialization
        matplotlib.use('Agg')
    
    def prepare_data(self, df):
        df = df.copy()
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        return df
    
    def create_evaluation_metrics(self, actual, predicted):
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        return {'mse': mse, 'mae': mae}
    
    def prophet_forecasting(self, train_df, test_df):
        """Prophet-based forecasting"""
        predictions = {}
        
        for direction in ['traffic_in', 'traffic_out']:
            try:
                model_name = f'Prophet_{direction}'
                model = self.checkpointer.load_model(model_name, train_df.name)
                
                if model is None:
                    train_prophet = pd.DataFrame({
                        'ds': train_df['timestamp'],
                        'y': train_df[direction]
                    })
                    
                    model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
                    model.fit(train_prophet)
                    self.checkpointer.save_model(model, model_name, train_df.name)
                
                future = pd.DataFrame({
                    'ds': pd.date_range(
                        start=test_df['timestamp'].iloc[0],
                        periods=len(test_df) + 30,  # Forecasting next 30 days
                        freq='D'
                    )
                })
                forecast = model.predict(future)
                predictions[direction] = pd.DataFrame({
                    'timestamp': forecast['ds'],
                    direction: forecast['yhat']
                })
            except Exception as e:
                print(f"    Error in Prophet {direction}: {str(e)}")
                predictions[direction] = pd.DataFrame()
        
        return predictions
    
    def exp_smoothing_forecasting(self, train_df, test_df):
        """Exponential Smoothing-based forecasting"""
        predictions = {}
        for direction in ['traffic_in', 'traffic_out']:
            try:
                model_name = f'Exponential_Smoothing_{direction}'
                model = self.checkpointer.load_model(model_name, train_df.name)
                
                if model is None:
                    model = ExponentialSmoothing(
                        train_df[direction],
                        seasonal_periods=7,
                        seasonal='add'
                    ).fit()
                    self.checkpointer.save_model(model, model_name, train_df.name)
                
                forecast_index = pd.date_range(
                    start=test_df['timestamp'].iloc[0],
                    periods=len(test_df) + 30,  # Forecasting next 30 days
                    freq='D'
                )
                forecast_values = model.forecast(len(forecast_index))
                predictions[direction] = pd.DataFrame({
                    'timestamp': forecast_index,
                    direction: forecast_values
                })
            except Exception as e:
                print(f"    Error in Exponential Smoothing {direction}: {str(e)}")
                predictions[direction] = pd.DataFrame()
        
        return predictions
    
    def seasonal_decompose_forecasting(self, train_df, test_df):
        """Seasonal Decomposition-based forecasting"""
        predictions = {}
        for direction in ['traffic_in', 'traffic_out']:
            try:
                model_name = f'Seasonal_Decompose_{direction}'
                decomp_results = self.checkpointer.load_model(model_name, train_df.name)
                
                if decomp_results is None:
                    decomp_results = seasonal_decompose(
                        train_df[direction],
                        period=7
                    )
                    self.checkpointer.save_model(decomp_results, model_name, train_df.name)
                
                seasonal = decomp_results.seasonal[-7:]
                trend = np.mean(decomp_results.trend[~np.isnan(decomp_results.trend)])
                
                forecast_index = pd.date_range(
                    start=test_df['timestamp'].iloc[0],
                    periods=len(test_df) + 30,  # Forecasting next 30 days
                    freq='D'
                )
                expected = trend + np.tile(seasonal, len(forecast_index) // 7 + 1)[:len(forecast_index)]
                predictions[direction] = pd.DataFrame({
                    'timestamp': forecast_index,
                    direction: expected
                })
            except Exception as e:
                print(f"    Error in Seasonal Decompose {direction}: {str(e)}")
                predictions[direction] = pd.DataFrame()
        
        return predictions
    
    def evaluate_link(self, df, link_name):
        print(f"\nEvaluating forecasting models for link: {link_name}")
        df = self.prepare_data(df)
        df.name = link_name
        
        train_size = int(len(df) * 0.7)
        train_df = df[:train_size]
        test_df = df[train_size:]
        train_df.name = link_name
        
        link_results = {}
        anomaly_scores = {}
        f1_scores = {}
        
        for model_name, model_func in self.models.items():
            try:
                predictions = model_func(train_df, test_df)
                
                for direction in ['traffic_in', 'traffic_out']:
                    if direction in predictions and not predictions[direction].empty:
                        test_predictions = predictions[direction]
                        test_predictions = test_predictions[
                            test_predictions['timestamp'].isin(test_df['timestamp'])
                        ]
                        
                        # Calculate metrics
                        metrics = self.create_evaluation_metrics(
                            test_df[direction],
                            test_predictions[direction]
                        )
                        
                        # Calculate anomaly rate
                        threshold = test_df[direction].mean() + 2 * test_df[direction].std()
                        anomalies = (test_predictions[direction] > threshold).mean()
                        anomaly_scores[f"{model_name}_{direction}"] = anomalies
                        
                        # Calculate F1 score (using anomalies as a proxy)
                        actual_anomalies = (test_df[direction] > threshold)
                        predicted_anomalies = (test_predictions[direction] > threshold)
                        tp = (actual_anomalies & predicted_anomalies).sum()
                        fp = (~actual_anomalies & predicted_anomalies).sum()
                        fn = (actual_anomalies & ~predicted_anomalies).sum()
                        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                        f1_scores[f"{model_name}_{direction}"] = f1
                        
                        link_results[f"{model_name}_{direction}"] = metrics
                        
                self.visualizer.plot_forecasts(
                    df=df,
                    forecasts=predictions.get('traffic_in', pd.DataFrame()),
                    model_name=model_name,
                    link_name=link_name
                )
            except Exception as e:
                print(f"    Error in {model_name}: {str(e)}")
        
        if link_results:
            best_model = min(link_results.items(), key=lambda x: x[1]['mse'])
            self.detailed_results[link_name] = {
                'best_model': best_model[0],
                'mse': best_model[1]['mse'],
                'mae': best_model[1]['mae'],
                'f1_score': f1_scores[best_model[0]],
                'anomaly_rate': anomaly_scores[best_model[0]]
            }
        
        return link_results

    def evaluate_all_links(self, links_data):
        """
        Evaluate all forecasting models for each link in the dataset.
        
        Args:
            links_data (dict): Dictionary mapping link names to their respective DataFrames
            
        Returns:
            dict: Results for all links and models
        """
        print(f"Starting evaluation of {len(links_data)} links...")
        all_results = {}
        
        for link_name, df in links_data.items():
            try:
                print(f"\nProcessing link: {link_name}")
                all_results[link_name] = self.evaluate_link(df, link_name)
            except Exception as e:
                print(f"Error processing link {link_name}: {str(e)}")
                continue
        
        # Calculate aggregate metrics across all models
        self.results = {}
        for model_name in self.models.keys():
            model_metrics = {
                'mse': [],
                'mae': []
            }
            
            for direction in ['traffic_in', 'traffic_out']:
                model_key = f"{model_name}_{direction}"
                for link_results in all_results.values():
                    if model_key in link_results:
                        for metric in ['mse', 'mae']:
                            model_metrics[metric].append(link_results[model_key][metric])
            
            if model_metrics['mse']:  # Only include if we have results
                self.results[model_name] = {
                    'mse': np.mean(model_metrics['mse']),
                    'mae': np.mean(model_metrics['mae'])
                }
        
        # Generate visualization for model comparison
        if self.results:
            self.visualizer.plot_model_comparison(self.results)
            print("\nModel comparison visualization saved.")
        
        return all_results