# utils.py
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
import numpy as np

class ModelCheckpoint:
    """Handle model persistence operations."""
    def __init__(self, base_dir='model/anomalies/output'):
        self.base_dir = base_dir
        self.checkpoint_dir = os.path.join(base_dir, 'model_checkpoints')
        self._create_directories()
    
    def _create_directories(self):
        """Create model-specific directories."""
        model_types = [
            'IsolationForest', 
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
        """Save model state."""
        model_dir = os.path.join(self.checkpoint_dir, model_name)
        filename = os.path.join(model_dir, f"{link_name}_{model_name}.pkl")
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
            print(f"Model saved: {filename}")
    
    def load_model(self, model_name, link_name):
        """Load model state if exists."""
        filename = os.path.join(self.checkpoint_dir, model_name,
                              f"{link_name}_{model_name}.pkl")
        
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                print(f"Model loaded: {filename}")
                return pickle.load(f)
        return None

class ModelVisualizer:
    """Handle visualization of model results."""
    def __init__(self, base_dir='model/anomalies/output'):
        self.base_dir = base_dir
        self.viz_dir = os.path.join(base_dir, 'model_visualizations')
        self._create_directories()
    
    def _create_directories(self):
        """Create visualization directories."""
        model_types = [
            'Statistical',
            'IsolationForest', 
            'Prophet',
            'Exponential_Smoothing',
            'Seasonal_Decompose',
            'Comparisons'
        ]
        for model_type in model_types:
            model_dir = os.path.join(self.viz_dir, model_type)
            os.makedirs(model_dir, exist_ok=True)
    
    def plot_predictions(self, df, predictions, model_name, link_name):
        """Plot model predictions."""
        plt.figure(figsize=(15, 8))
        
        for idx, direction in enumerate(['traffic_in', 'traffic_out']):
            plt.subplot(2, 1, idx + 1)
            plt.plot(df['timestamp'], df[direction], 
                    label=f'Actual Traffic {direction.split("_")[1].title()}',
                    alpha=0.7)
            plt.scatter(df[predictions]['timestamp'],
                       df[predictions][direction],
                       color='red', label='Anomalies', alpha=0.5)
            plt.title(f'{model_name} - {direction.replace("_", " ").title()} '
                     f'Anomalies for {link_name}')
            plt.legend()
        
        plt.tight_layout()
        
        model_dir = os.path.join(self.viz_dir, model_name)
        plt.savefig(os.path.join(model_dir, 
                                f"{link_name}_{model_name}_predictions.png"))
        plt.close()

    def plot_model_comparison(self, results):
        """Plot comparison of model performances."""
        metrics = ['precision', 'recall', 'f1_score', 'anomaly_rate']
        model_names = list(results.keys())
        
        plt.figure(figsize=(15, 12))
        
        for idx, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, idx)
            
            # Extract metric values for each model
            values = [results[model][metric] for model in model_names]
            
            # Create bar plot
            sns.barplot(x=model_names, y=values)
            
            # Customize plot
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.xticks(rotation=45)
            plt.xlabel('Models')
            plt.ylabel(metric.replace('_', ' ').title())
            
            # Add value labels on top of bars
            for i, v in enumerate(values):
                plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        plt.suptitle('Model Performance Comparison', size=16)
        plt.tight_layout()
        
        # Save plot
        comparison_path = os.path.join(self.viz_dir, 'Comparisons', 'model_comparison.png')
        plt.savefig(comparison_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Model comparison plot saved to: {comparison_path}")