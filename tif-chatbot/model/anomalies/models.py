# models.py
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from utils import ModelVisualizer, ModelCheckpoint

warnings.filterwarnings('ignore')

class BaseModel:
    def prepare_data(self, df):
        """Prepare data with time-based features."""
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['time_of_day'] = pd.cut(df['hour'], 
                                  bins=[-1, 6, 12, 18, 24], 
                                  labels=['night', 'morning', 'afternoon', 'evening'])
        return df

    def create_ground_truth(self, df, threshold=3):
        """Create ground truth labels using z-score method."""
        anomalies = []
        for direction in ['traffic_in', 'traffic_out']:
            mean = df[direction].mean()
            std = df[direction].std()
            z_scores = np.abs((df[direction] - mean) / std)
            anomalies.append(z_scores > threshold)
        return np.logical_or.reduce(anomalies)

class AnomalyDetector(BaseModel):
    def __init__(self):
        self.models = {
            'Statistical': self.statistical_detection,
            'IsolationForest': self.isolation_forest_detection,
            'Prophet': self.prophet_detection,
            'Exponential_Smoothing': self.exp_smoothing_detection,
            'Seasonal_Decompose': self.seasonal_decompose_detection
        }
        self.results = {}
        self.detailed_results = defaultdict(dict)
        self.checkpointer = ModelCheckpoint()
        self.visualizer = ModelVisualizer()

    def statistical_detection(self, train_df, test_df):
        """Simple statistical anomaly detection using z-scores."""
        anomalies = []
        for direction in ['traffic_in', 'traffic_out']:
            mean = train_df[direction].mean()
            std = train_df[direction].std()
            z_scores = np.abs((test_df[direction] - mean) / std)
            anomalies.append(z_scores > 3)
        return np.logical_or.reduce(anomalies)

    def isolation_forest_detection(self, train_df, test_df):
        """Isolation Forest based anomaly detection."""
        features = ['traffic_in', 'traffic_out', 'hour', 'day_of_week', 'is_weekend']
        
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df[features])
        test_scaled = scaler.transform(test_df[features])
        
        model = self.checkpointer.load_model('IsolationForest', train_df.name)
        if model is None:
            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(train_scaled)
            self.checkpointer.save_model(model, 'IsolationForest', train_df.name)
        
        return model.predict(test_scaled) == -1

    def prophet_detection(self, train_df, test_df):
        """Prophet-based anomaly detection."""
        anomalies = []
        for direction in ['traffic_in', 'traffic_out']:
            try:
                model_name = f'Prophet_{direction}'
                model = self.checkpointer.load_model(model_name, train_df.name)
                
                if model is None:
                    train_prophet = pd.DataFrame({
                        'ds': train_df['timestamp'],
                        'y': train_df[direction]
                    })
                    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
                    model.fit(train_prophet)
                    self.checkpointer.save_model(model, model_name, train_df.name)
                
                future = pd.DataFrame({'ds': test_df['timestamp']})
                forecast = model.predict(future)
                predictions = self._calculate_anomalies(test_df[direction].values, 
                                                      forecast['yhat'].values)
                anomalies.append(predictions)
                
            except Exception as e:
                print(f"Error in Prophet {direction}: {str(e)}")
                anomalies.append(self.statistical_detection(
                    train_df[[direction]], test_df[[direction]]))
        
        return np.logical_or.reduce(anomalies)

    def exp_smoothing_detection(self, train_df, test_df):
        """Exponential Smoothing based anomaly detection."""
        anomalies = []
        for direction in ['traffic_in', 'traffic_out']:
            try:
                model_name = f'Exponential_Smoothing_{direction}'
                model = self.checkpointer.load_model(model_name, train_df.name)
                
                if model is None:
                    model = ExponentialSmoothing(
                        train_df[direction],
                        seasonal_periods=24,
                        seasonal='add'
                    ).fit()
                    self.checkpointer.save_model(model, model_name, train_df.name)
                
                predictions = model.forecast(len(test_df))
                anomalies.append(self._calculate_anomalies(
                    test_df[direction].values, predictions))
                
            except Exception as e:
                print(f"Error in Exponential Smoothing {direction}: {str(e)}")
                anomalies.append(self.statistical_detection(
                    train_df[[direction]], test_df[[direction]]))
        
        return np.logical_or.reduce(anomalies)

    def seasonal_decompose_detection(self, train_df, test_df):
        """Seasonal Decomposition based anomaly detection."""
        anomalies = []
        for direction in ['traffic_in', 'traffic_out']:
            try:
                model_name = f'Seasonal_Decompose_{direction}'
                decomp_results = self.checkpointer.load_model(model_name, train_df.name)
                
                if decomp_results is None:
                    decomp_results = seasonal_decompose(
                        train_df[direction],
                        period=24
                    )
                    self.checkpointer.save_model(decomp_results, model_name, train_df.name)
                
                expected = self._calculate_expected_values(decomp_results, len(test_df))
                anomalies.append(self._calculate_anomalies(
                    test_df[direction].values, expected))
                
            except Exception as e:
                print(f"Error in Seasonal Decompose {direction}: {str(e)}")
                anomalies.append(self.statistical_detection(
                    train_df[[direction]], test_df[[direction]]))
        
        return np.logical_or.reduce(anomalies)

    def _calculate_anomalies(self, actual, predicted, threshold_percentile=95):
        """Calculate anomalies based on residuals."""
        residuals = np.abs(actual - predicted)
        threshold = np.percentile(residuals, threshold_percentile)
        return residuals > threshold

    def _calculate_expected_values(self, decomp_results, length):
        """Calculate expected values from seasonal decomposition."""
        seasonal = decomp_results.seasonal[-24:]
        trend = np.mean(decomp_results.trend[~np.isnan(decomp_results.trend)])
        return trend + np.tile(seasonal, length//24 + 1)[:length]

    def evaluate_link(self, df, link_name):
        """Evaluate all models on a single link."""
        print(f"\nEvaluating models for link: {link_name}")
        df = self.prepare_data(df)
        df.name = link_name
        
        train_size = int(len(df) * 0.7)
        train_df = df[:train_size]
        test_df = df[train_size:]
        train_df.name = link_name
        
        ground_truth = self.create_ground_truth(test_df)
        link_results = {}
        
        for model_name, model_func in self.models.items():
            try:
                print(f"  Testing {model_name}...")
                predictions = model_func(train_df, test_df)
                
                metrics = self._calculate_metrics(ground_truth, predictions)
                link_results[model_name] = metrics
                
                self.visualizer.plot_predictions(test_df, predictions, 
                                               model_name, link_name)
                
            except Exception as e:
                print(f"    Error in {model_name}: {str(e)}")
        
        if link_results:
            self._update_results(link_name, link_results)
        
        return link_results

    def evaluate_all_links(self, links_data):
        """Evaluate all models on all links."""
        all_results = {}
        for link_name, df in links_data.items():
            all_results[link_name] = self.evaluate_link(df, link_name)
        
        self._calculate_aggregate_results(all_results)
        self.visualizer.plot_model_comparison(self.results)
        return all_results

    def _calculate_metrics(self, ground_truth, predictions):
        """Calculate performance metrics."""
        return {
            'precision': precision_score(ground_truth, predictions),
            'recall': recall_score(ground_truth, predictions),
            'f1_score': f1_score(ground_truth, predictions),
            'total_anomalies': np.sum(predictions),
            'anomaly_rate': np.sum(predictions) / len(predictions)
        }

    def _update_results(self, link_name, link_results):
        """Update detailed results with best model."""
        best_model = max(link_results.items(), key=lambda x: x[1]['f1_score'])
        self.detailed_results[link_name] = {
            'best_model': best_model[0],
            'f1_score': best_model[1]['f1_score'],
            'anomaly_rate': best_model[1]['anomaly_rate']
        }

    def _calculate_aggregate_results(self, all_results):
        """Calculate aggregate results across all links."""
        self.results = {model: {
            metric: np.mean([
                link_results[model][metric]
                for link_results in all_results.values()
                if model in link_results
            ])
            for metric in ['precision', 'recall', 'f1_score', 'anomaly_rate']
        } for model in self.models.keys()}
        
        for model in self.models.keys():
            self.results[model]['total_anomalies'] = sum(
                link_results[model]['total_anomalies']
                for link_results in all_results.values()
                if model in link_results
            )