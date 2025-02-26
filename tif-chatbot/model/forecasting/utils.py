# File: utils.py
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns

import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

class ModelCheckpoint:
    def __init__(self, base_dir='model/forecasting/output'):
        self.base_dir = base_dir
        self.checkpoint_dir = os.path.join(base_dir, 'model_checkpoints')
        self._create_model_directories()
    
    def _create_model_directories(self):
        """Create directories for each model type"""
        model_types = [
            'Prophet_traffic_in',
            'Prophet_traffic_out',
            'Exponential_Smoothing_traffic_in',
            'Exponential_Smoothing_traffic_out',
            'Seasonal_Decompose_traffic_in',
            'Seasonal_Decompose_traffic_out'
        ]
        for model_type in model_types:
            model_dir = os.path.join(self.checkpoint_dir, model_type)
            os.makedirs(model_dir, exist_ok=True)
    
    def save_model(self, model, model_name, link_name):
        """Save model in its corresponding directory"""
        model_dir = os.path.join(self.checkpoint_dir, model_name)
        filename = os.path.join(model_dir, f"{link_name}_{model_name}.pkl")
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
            print(f"Model saved: {filename}")
    
    def load_model(self, model_name, link_name):
        """Load model from its corresponding directory"""
        model_dir = os.path.join(self.checkpoint_dir, model_name)
        filename = os.path.join(model_dir, f"{link_name}_{model_name}.pkl")
        
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                print(f"Model loaded: {filename}")
                return pickle.load(f)
        return None

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
        plt.figure(figsize=(15, 8))
        
        plt.plot(df['timestamp'], df['traffic_in'], label='Actual Traffic In', alpha=0.7)
        plt.plot(forecasts['timestamp'], forecasts['traffic_in'], label='Forecast Traffic In', alpha=0.7)
        
        plt.title(f'{model_name} - Traffic In Forecast for {link_name}')
        plt.legend()
        plt.xlabel('Timestamp')
        plt.ylabel('Traffic In')
        
        model_dir = os.path.join(self.viz_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        plt.savefig(os.path.join(model_dir, f"{link_name}_{model_name}_forecast.png"))
        plt.close()
    
    def plot_model_comparison(self, results):
        """Plot model comparison in Comparisons directory"""
        metrics = ['rmse', 'mae', 'mape']
        model_names = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Comparison Metrics')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            values = [results[model][metric] for model in model_names]
            sns.barplot(x=model_names, y=values, ax=ax)
            ax.set_title(metric.upper())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save in Comparisons directory
        comparison_dir = os.path.join(self.viz_dir, 'Comparisons')
        os.makedirs(comparison_dir, exist_ok=True)
        plt.savefig(os.path.join(comparison_dir, "model_comparison.png"))
        plt.close()
# In utils.py
class ModelReporter:
    def __init__(self, base_dir='model/forecasting/output'):
        self.base_dir = base_dir
        self.report_dir = os.path.join(base_dir, 'reports')
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.report_dir, exist_ok=True)
    
    def generate_report(self, detailed_results):
        """Generate and save evaluation report"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_content = f"Forecasting Evaluation Report\nGenerated: {datetime.now()}\n\n"
            report_content += f"Total Links Evaluated: {len(detailed_results)}\n\n"
            
            # Group results by model type
            model_stats = defaultdict(list)
            for link_name, results in detailed_results.items():
                model_stats[results['best_model']].append({
                    'link': link_name,
                    'mse': results['mse'],
                    'mae': results['mae']
                })
            
            # Add model summary statistics
            report_content += "Model Performance Summary:\n"
            report_content += "=" * 50 + "\n\n"
            for model, stats in model_stats.items():
                avg_mse = np.mean([s['mse'] for s in stats])
                avg_mae = np.mean([s['mae'] for s in stats])
                count = len(stats)
                report_content += f"{model}:\n"
                report_content += f"  Number of best performances: {count}\n"
                report_content += f"  Average MSE: {avg_mse:.4f}\n"
                report_content += f"  Average MAE: {avg_mae:.4f}\n\n"
            
            # Add detailed results
            report_content += "\nDetailed Results by Link:\n"
            report_content += "=" * 50 + "\n\n"
            for link_name, results in detailed_results.items():
                report_content += f"Link: {link_name}\n"
                report_content += f"  Best Model: {results['best_model']}\n"
                report_content += f"  MSE: {results['mse']:.4f}\n"
                report_content += f"  MAE: {results['mae']:.4f}\n\n"
            
            filename = os.path.join(self.report_dir, f"forecasting_report_{timestamp}.txt")
            with open(filename, 'w') as f:
                f.write(report_content)
                
            print(f"Report saved: {filename}")
            return report_content
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return None