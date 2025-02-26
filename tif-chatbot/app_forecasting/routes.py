from flask import Flask, Blueprint, jsonify, request, render_template
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
import warnings
import logging
from logging.config import dictConfig
import os
import json
from pathlib import Path
import time
import psycopg2
from psycopg2.extras import execute_batch
import threading
import contextlib
import traceback
import pickle

# Suppress Prophet stdout
os.environ['CMDSTANPY_STDOUT'] = 'false'
os.environ['PROPHET_STDOUT'] = 'false'

# Create Blueprint
from flask import Blueprint

app_forecasting_bp = Blueprint('app_forecasting', __name__, template_folder='templates')

data_store = None
# Keep existing logging configuration
dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'null': {
            'class': 'logging.NullHandler',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'forecast.log',
            'formatter': 'standard',
            'level': 'ERROR'
        }
    },
    'loggers': {
        '': {
            'handlers': ['file'],
            'level': 'ERROR',
        },
        'cmdstanpy': {
            'handlers': ['null'],
            'propagate': False
        },
        'prophet': {
            'handlers': ['null'],
            'propagate': False
        },
        'stan': {
            'handlers': ['null'],
            'propagate': False
        },
        'stan.fit': {
            'handlers': ['null'],
            'propagate': False
        }
    }
})

# Configuration
CACHE_DIR = Path("cache/forecast")
CACHE_DURATION = 24 * 60 * 60  # 24 hours in seconds

DB_CONFIG = {
    'host': '36.67.62.245',
    'port': '8082',
    'database': 'sisai',
    'user': 'postgres',
    'password': 'uhuy123'
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS traffic_forecasts (
    id SERIAL PRIMARY KEY,
    rrd_path VARCHAR(255) NOT NULL,
    prediction_date TIMESTAMP NOT NULL,
    next_week_traffic_in FLOAT NOT NULL,
    next_week_traffic_out FLOAT NOT NULL,
    next_month_traffic_in FLOAT NOT NULL,
    next_month_traffic_out FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (rrd_path, prediction_date)
);
"""

class CacheManager:
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_lock = threading.Lock()
        self.load_cache()

    def load_cache(self):
        self.cache = {
            'links': {},
            'overview': None,
            'insights': {},
            'last_update': None
        }
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    if cache_file.stem == "network_cache":
                        self.cache = json.load(f)
                    else:
                        link_id = cache_file.stem
                        self.cache['links'][link_id] = json.load(f)
                print(f"Loaded cache: {cache_file.name}")
            except Exception as e:
                print(f"Error loading cache {cache_file}: {str(e)}")

    def save_cache(self):
        try:
            network_cache_file = self.cache_dir / "network_cache.json"
            with open(network_cache_file, 'w') as f:
                json.dump(self.cache, f)

            for link_id, link_data in self.cache['links'].items():
                link_cache_file = self.cache_dir / f"{link_id}.json"
                with open(link_cache_file, 'w') as f:
                    json.dump(link_data, f)
            print("Cache saved successfully")
        except Exception as e:
            print(f"Error saving cache: {str(e)}")

    def needs_update(self):
        if not self.cache['last_update']:
            return True
        last_update = datetime.fromtimestamp(self.cache['last_update'])
        return (datetime.now() - last_update).total_seconds() >= CACHE_DURATION

    def update_cache(self, data_store):
        with self.cache_lock:
            try:
                if not self.needs_update():
                    return True

                print("\nStarting cache update...")
                
                overview_data = data_store.process_traffic_data()
                if overview_data:
                    self.cache['overview'] = overview_data
                    self.cache['insights']['overview'] = data_store.calculate_insights()
                
                available_links = data_store.get_available_links()
                total_links = len(available_links)
                batch_size = 10
                
                for i in range(0, total_links, batch_size):
                    batch = available_links[i:i+batch_size]
                    print(f"\rProcessing links {i+1}-{min(i+batch_size, total_links)} of {total_links}", end='')
                    
                    for link in batch:
                        link_id = link['id']
                        link_data = data_store.process_traffic_data(link_id)
                        if link_data:
                            self.cache['links'][link_id] = link_data
                            self.cache['insights'][link_id] = data_store.calculate_insights(link_id)
                
                self.cache['last_update'] = time.time()
                self.save_cache()
                
                print("\nCache update completed")
                return True
            except Exception as e:
                print(f"Error updating cache: {str(e)}")
                traceback.print_exc()
                return False

class DataStorage:
    def __init__(self, base_dir="data/json_storage"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.forecasts_dir = self.base_dir / "forecasts"
        self.dashboard_dir = self.base_dir / "dashboard"
        self.forecasts_dir.mkdir(exist_ok=True)
        self.dashboard_dir.mkdir(exist_ok=True)

    def save_forecast_data(self, link_id, data):
        file_path = self.forecasts_dir / f"{link_id}_forecast.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def save_dashboard_data(self, link_id, data):
        file_path = self.dashboard_dir / f"{link_id}_dashboard.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_forecast_data(self, link_id):
        file_path = self.forecasts_dir / f"{link_id}_forecast.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None

    def load_dashboard_data(self, link_id):
        file_path = self.dashboard_dir / f"{link_id}_dashboard.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None

class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.setup_database()

    def get_connection(self):
        if self.conn is None:
            self.conn = psycopg2.connect(**DB_CONFIG)
        return self.conn

    def setup_database(self):
        conn = self.get_connection()
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()

    def save_forecast(self, rrd_path, allocations):
        if not allocations:
            return False

        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO traffic_forecasts 
                    (rrd_path, prediction_date, next_week_traffic_in, next_week_traffic_out, 
                     next_month_traffic_in, next_month_traffic_out)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (rrd_path, prediction_date) 
                    DO UPDATE SET
                        next_week_traffic_in = EXCLUDED.next_week_traffic_in,
                        next_week_traffic_out = EXCLUDED.next_week_traffic_out,
                        next_month_traffic_in = EXCLUDED.next_month_traffic_in,
                        next_month_traffic_out = EXCLUDED.next_month_traffic_out,
                        created_at = CURRENT_TIMESTAMP
                """, (
                    rrd_path,
                    datetime.now(),
                    allocations['next_week']['in'],
                    allocations['next_week']['out'],
                    allocations['next_month']['in'],
                    allocations['next_month']['out']
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving forecast: {str(e)}")
            conn.rollback()
            return False

class TrafficPredictor:
    def __init__(self):
        self.model_base_dir = Path("model") / "forecasting" / "output" / "model_checkpoints"
        self.models = {}
        self._load_model_checkpoints()

    def _load_model_checkpoints(self):
        try:
            traffic_in_dir = self.model_base_dir / "Prophet_traffic_in"
            traffic_out_dir = self.model_base_dir / "Prophet_traffic_out"

            print(f"Looking for models in:")
            print(f"Traffic IN directory: {traffic_in_dir}")
            print(f"Traffic OUT directory: {traffic_out_dir}")

            if not traffic_in_dir.exists() or not traffic_out_dir.exists():
                print("Warning: One or both model directories do not exist!")
                return

            for model_file in traffic_in_dir.glob("*.pkl"):
                try:
                    link_id = model_file.stem.replace('_Prophet_traffic_in', '')
                    with open(model_file, 'rb') as f:
                        self.models[f"{link_id}_in"] = pickle.load(f)
                    print(f"Successfully loaded traffic_in model for link {link_id}")
                except Exception as e:
                    print(f"Error loading traffic_in model for {model_file.name}: {str(e)}")

            for model_file in traffic_out_dir.glob("*.pkl"):
                try:
                    link_id = model_file.stem.replace('_Prophet_traffic_out', '')
                    with open(model_file, 'rb') as f:
                        self.models[f"{link_id}_out"] = pickle.load(f)
                    print(f"Successfully loaded traffic_out model for link {link_id}")
                except Exception as e:
                    print(f"Error loading traffic_out model for {model_file.name}: {str(e)}")

            print(f"\nTotal models loaded: {len(self.models)}")
            print("Loaded models for links:", [k.split('_')[0] for k in self.models.keys()])

        except Exception as e:
            print(f"Error loading model checkpoints: {str(e)}")
            traceback.print_exc()

    def prepare_data(self, df, column):
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df['timestamp']),
            'y': df[column]
        })
        return prophet_df

    def fit_predict(self, df, link_id, days=30):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model_in = self.models.get(f"{link_id}_in")
                model_out = self.models.get(f"{link_id}_out")

                if model_in is not None and model_out is not None:
                    print(f"Using pre-trained models for link {link_id}")
                    future_dates = model_in.make_future_dataframe(periods=days)
                    
                    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                        forecast_in = model_in.predict(future_dates)
                        forecast_out = model_out.predict(future_dates)
                    
                    return forecast_in, forecast_out
                else:
                    print(f"No pre-trained models found for link {link_id}, training new models")
                    
                    model_in = Prophet(
                        daily_seasonality=True,
                        weekly_seasonality=True,
                        yearly_seasonality=False,
                        changepoint_prior_scale=0.05
                    )
                    model_out = Prophet(
                        daily_seasonality=True,
                        weekly_seasonality=True,
                        yearly_seasonality=False,
                        changepoint_prior_scale=0.05
                    )

                    df_in = self.prepare_data(df, 'traffic_in')
                    df_out = self.prepare_data(df, 'traffic_out')

                    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                        model_in.fit(df_in)
                        model_out.fit(df_out)
                        
                    future_dates = model_in.make_future_dataframe(periods=days)
                    forecast_in = model_in.predict(future_dates)
                    forecast_out = model_out.predict(future_dates)

                    return forecast_in, forecast_out

        except Exception as e:
            print(f"Error in Prophet prediction for link {link_id}: {str(e)}")
            traceback.print_exc()
            return None, None

class TrafficDataStore:
    def __init__(self, log_file_path):
        self.links = {}
        self.predictor = TrafficPredictor()
        self.db_manager = DatabaseManager()
        self.cache_manager = CacheManager()
        self.data_storage = DataStorage()
        self._load_data(log_file_path)
        
        if self.cache_manager.needs_update():
            self.cache_manager.update_cache(self)

    def _load_data(self, log_file_path):
        try:
            print(f"Reading log file: {log_file_path}")
            with open(log_file_path, 'r') as file:
                current_link = None
                current_data = []
                link_count = 0
                LINK_LIMIT = 496
                
                for line in file:
                    line = line.strip()
                    
                    if line.startswith("Data from"):
                        if current_link and current_data:
                            df = pd.DataFrame(current_data)
                            if not df.empty:
                                self.links[current_link] = df
                                link_count += 1
                                if link_count >= LINK_LIMIT:
                                    break
                        
                        current_link = line.split('/')[-1].split('.')[0]
                        current_data = []
                        continue

                    if not line or 'traffic_in' in line:
                        continue

                    try:
                        timestamp_str, values_str = line.split(':')
                        timestamp = int(timestamp_str.strip())
                        traffic_values = values_str.strip().split()
                        
                        if len(traffic_values) == 2:
                            traffic_in = float(traffic_values[0])
                            traffic_out = float(traffic_values[1])
                            
                            if not (np.isnan(traffic_in) or np.isnan(traffic_out)):current_data.append({
                                    'timestamp': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d'),
                                    'traffic_in': traffic_in,
                                    'traffic_out': traffic_out
                                })
                                
                    except (ValueError, IndexError):
                        continue

                if current_link and current_data and link_count < LINK_LIMIT:
                    df = pd.DataFrame(current_data)
                    if not df.empty:
                        self.links[current_link] = df

            print(f"Successfully loaded {len(self.links)} links (limited to {LINK_LIMIT})")

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            self.links = {}
    def get_traffic_data(self, link_id=None):
        """Get traffic data, first trying from storage, then processing if needed"""
        try:
            storage_id = link_id if link_id else 'overview'
            
            # Try loading from storage first
            forecast_data = self.data_storage.load_forecast_data(storage_id)
            dashboard_data = self.data_storage.load_dashboard_data(storage_id)
            
            if forecast_data and dashboard_data:
                print(f"Retrieved cached data for {storage_id}")
                return {
                    'dates': forecast_data['dates'],
                    'trafficIn': dashboard_data['traffic']['in'],
                    'trafficOut': dashboard_data['traffic']['out'],
                    'forecastIn': forecast_data['forecasts']['in'],
                    'forecastOut': forecast_data['forecasts']['out'],
                    'allocations': forecast_data['allocations']
                }
            
            # If no stored data, process new data
            print(f"No cached data found for {storage_id}, processing new data")
            return self.process_traffic_data(link_id)

        except Exception as e:
            print(f"Error getting traffic data: {str(e)}")
            traceback.print_exc()
            return None
    def calculate_traffic_allocations(self, df):
        """
        Calculate traffic allocations based on historical data and forecasts.
        
        Args:
            df (pandas.DataFrame): DataFrame containing traffic data
        
        Returns:
            dict: Dictionary containing current and forecasted traffic allocations
        """
        try:
            print("Calculating traffic allocations...")
            
            df_gb = df.copy()
            df_gb['date'] = pd.to_datetime(df_gb['timestamp'])
            
            today = pd.Timestamp(1727197200, unit='s')
            print(f"Using reference date: {today}")
            week_ago = today - pd.Timedelta(days=7)
            month_ago = today - pd.Timedelta(days=30)
            
            # Calculate current traffic
            current_week = df_gb[df_gb['date'] >= week_ago]
            current_month = df_gb[df_gb['date'] >= month_ago]
            
            current_week_in = float(current_week['traffic_in'].sum() / 1e9)
            current_week_out = float(current_week['traffic_out'].sum() / 1e9)
            current_month_in = float(current_month['traffic_in'].sum() / 1e9)
            current_month_out = float(current_month['traffic_out'].sum() / 1e9)
            
            # Handle the case when there's no data
            if current_week.empty:
                print("Warning: No current week data available")
                current_week_in = current_week_out = 0
                
            if current_month.empty:
                print("Warning: No current month data available")
                current_month_in = current_month_out = 0
            
            # Calculate future traffic using simple average if no forecast available
            avg_daily_in = df_gb['traffic_in'].mean() / 1e9
            avg_daily_out = df_gb['traffic_out'].mean() / 1e9
            
            next_week_in = avg_daily_in * 7
            next_week_out = avg_daily_out * 7
            next_month_in = avg_daily_in * 30
            next_month_out = avg_daily_out * 30
            
            return {
                'current_week': {
                    'in': current_week_in,
                    'out': current_week_out
                },
                'next_week': {
                    'in': next_week_in,
                    'out': next_week_out
                },
                'current_month': {
                    'in': current_month_in,
                    'out': current_month_out
                },
                'next_month': {
                    'in': next_month_in,
                    'out': next_month_out
                }
            }

        except Exception as e:
            print(f"Error calculating traffic allocations: {str(e)}")
            return None
            
    def get_available_links(self):
        return [{"id": link_name, "name": link_name} for link_name in self.links.keys()]

    def process_traffic_data(self, link_id=None):
        try:
            print(f"Processing traffic data for {'overview' if link_id is None else link_id}")
            
            if link_id and link_id in self.links:
                df = self.links[link_id].copy()
                rrd_path = link_id
            else:
                if not self.links:
                    print("No links available in data store")
                    return None
                    
                dfs = []
                for link_name, link_df in self.links.items():
                    df_copy = link_df.copy()
                    dfs.append(df_copy)
                
                df = pd.concat(dfs)
                df = df.groupby('timestamp').sum().reset_index()
                rrd_path = 'overview'

            if df.empty:
                return None

            # Pass rrd_path as link_id to calculate_traffic_allocations
            allocations = self.calculate_traffic_allocations(df, rrd_path)

        except Exception as e:
            print(f"Error calculating traffic allocations: {str(e)}")
            return None

    def process_traffic_data(self, link_id=None):
        try:
            print(f"Processing traffic data for {'overview' if link_id is None else link_id}")
            
            if link_id and link_id in self.links:
                df = self.links[link_id].copy()
                rrd_path = link_id
            else:
                if not self.links:
                    print("No links available in data store")
                    return None
                    
                dfs = []
                for link_name, link_df in self.links.items():
                    df_copy = link_df.copy()
                    dfs.append(df_copy)
                
                df = pd.concat(dfs)
                df = df.groupby('timestamp').sum().reset_index()
                rrd_path = 'overview'

            if df.empty:
                return None

            allocations = self.calculate_traffic_allocations(df)
            if allocations:
                self.db_manager.save_forecast(rrd_path, allocations)

            forecast_in, forecast_out = self.predictor.fit_predict(df, rrd_path, days=7)

            df['traffic_in'] = df['traffic_in'] / 1e9
            df['traffic_out'] = df['traffic_out'] / 1e9

            if forecast_in is not None and forecast_out is not None:
                forecast_dates = forecast_in['ds'][:7].dt.strftime('%Y-%m-%d').tolist()
                forecast_in_values = (forecast_in['yhat'][:7] / 1e9).tolist()
                forecast_out_values = (forecast_out['yhat'][:7] / 1e9).tolist()
            else:
                last_date = pd.to_datetime(df['timestamp'].iloc[-1])
                forecast_dates = [(last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(7)]
                forecast_in_values = [df['traffic_in'].mean()] * 7
                forecast_out_values = [df['traffic_out'].mean()] * 7

            result = {
                'dates': df['timestamp'].tolist() + forecast_dates,
                'trafficIn': df['traffic_in'].tolist() + [None] * 7,
                'trafficOut': df['traffic_out'].tolist() + [None] * 7,
                'forecastIn': [None] * len(df) + forecast_in_values,
                'forecastOut': [None] * len(df) + forecast_out_values,
                'allocations': allocations
            }

            storage_id = link_id if link_id else 'overview'
            self.data_storage.save_forecast_data(storage_id, {
                'dates': result['dates'],
                'forecasts': {
                    'in': result['forecastIn'],
                    'out': result['forecastOut']
                },
                'allocations': result['allocations']
            })
            
            self.data_storage.save_dashboard_data(storage_id, {
                'traffic': {
                    'in': result['trafficIn'],
                    'out': result['trafficOut']
                },
                'insights': self.calculate_insights(link_id)
            })
            
            return result

        except Exception as e:
            print(f"Error processing traffic data: {str(e)}")
            traceback.print_exc()
            return None

    def calculate_insights(self, link_id=None):
        insights = []
        try:
            if link_id and link_id in self.links:
                df = self.links[link_id]
                allocations = self.calculate_traffic_allocations(df)
                
                if allocations:
                    week_change_in = ((allocations['next_week']['in'] - allocations['current_week']['in']) / 
                                allocations['current_week']['in'] * 100 if allocations['current_week']['in'] else 0)
                    week_change_out = ((allocations['next_week']['out'] - allocations['current_week']['out']) / 
                                    allocations['current_week']['out'] * 100 if allocations['current_week']['out'] else 0)
                    
                    month_change_in = ((allocations['next_month']['in'] - allocations['current_month']['in']) / 
                                    allocations['current_month']['in'] * 100 if allocations['current_month']['in'] else 0)
                    month_change_out = ((allocations['next_month']['out'] - allocations['current_month']['out']) / 
                                    allocations['current_month']['out'] * 100 if allocations['current_month']['out'] else 0)
                    
                    def get_status(change):
                        if abs(change) > 50:
                            return 'critical'
                        elif abs(change) > 20:
                            return 'warning'
                        return 'good'
                    
                    week_status = get_status(max(abs(week_change_in), abs(week_change_out)))
                    month_status = get_status(max(abs(month_change_in), abs(month_change_out)))
                    
                    current_utilization = max(
                        allocations['current_week']['in'] / 7,
                        allocations['current_week']['out'] / 7
                    )
                    utilization_status = 'good' if current_utilization < 5000 else 'warning' if current_utilization < 8000 else 'critical'
                    
                    insights.extend([
                        {
                            "name": "Weekly Traffic Trend",
                            "insight": f"Traffic {'increase' if week_change_in > 0 else 'decrease'} forecast",
                            "details": f"Incoming: {abs(week_change_in):.1f}% {'increase' if week_change_in > 0 else 'decrease'}\n"
                                    f"Outgoing: {abs(week_change_out):.1f}% {'increase' if week_change_out > 0 else 'decrease'}",
                            "status": week_status,
                            "metrics": [
                                {"label": "In", "value": f"{week_change_in:+.1f}%", "status": get_status(week_change_in)},
                                {"label": "Out", "value": f"{week_change_out:+.1f}%", "status": get_status(week_change_out)}
                            ]
                        },
                        {
                            "name": "Monthly Projection",
                            "insight": f"Traffic {'increase' if month_change_in > 0 else 'decrease'} forecast",
                            "details": f"Incoming: {abs(month_change_in):.1f}% {'increase' if month_change_in > 0 else 'decrease'}\n"
                                    f"Outgoing: {abs(month_change_out):.1f}% {'increase' if month_change_out > 0 else 'decrease'}",
                            "status": month_status,
                            "metrics": [
                                {"label": "In", "value": f"{month_change_in:+.1f}%", "status": get_status(month_change_in)},
                                {"label": "Out", "value": f"{month_change_out:+.1f}%", "status": get_status(month_change_out)}
                            ]
                        },
                        {
                            "name": "Utilization Status",
                            "insight": "Current network utilization level",
                            "details": f"Average daily traffic: {current_utilization:.2f} GB",
                            "status": utilization_status,
                            "metrics": [
                                {"label": "Load", "value": f"{(current_utilization/100):.1f}%", "status": utilization_status}
                            ]
                        }
                    ])
            else:
                total_links = len(self.links)
                critical_links = 0
                warning_links = 0
                
                for link_id in self.links:
                    df = self.links[link_id]
                    allocations = self.calculate_traffic_allocations(df)
                    if allocations:
                        current_util = max(
                            allocations['current_week']['in'] / 7,
                            allocations['current_week']['out'] / 7
                        )
                        if current_util >= 8000:
                            critical_links += 1
                        elif current_util >= 5000:
                            warning_links += 1
                
                network_status = 'critical' if critical_links > 0 else 'warning' if warning_links > 0 else 'good'
                
                insights.append({
                    "name": "Network Health Overview",
                    "insight": f"Monitoring {total_links} active links",
                    "details": f"Critical: {critical_links} links\nWarning: {warning_links} links\n"
                            f"Healthy: {total_links - critical_links - warning_links} links",
                    "status": network_status,
                    "metrics": [
                        {"label": "Critical", "value": str(critical_links), "status": "critical" if critical_links > 0 else "good"},
                        {"label": "Warning", "value": str(warning_links), "status": "warning" if warning_links > 0 else "good"}
                    ]
                })
                
        except Exception as e:
            print(f"Error generating insights: {str(e)}")
            insights.append({
                "name": "Status",
                "insight": "Unable to generate insights due to data processing error",
                "status": "critical",
                "details": str(e)
            })
        
        return insights

    def get_insights(self, link_id=None):
        try:
            if self.cache_manager.needs_update():
                self.cache_manager.update_cache(self)
            
            if link_id:
                return self.cache_manager.cache['insights'].get(link_id, [])
            return self.cache_manager.cache['insights'].get('overview', [])
            
        except Exception as e:
            print(f"Error getting insights: {str(e)}")
            return []

@app_forecasting_bp.route('/api/links-forecast')
def get_links():
    if data_store is None:
        return jsonify({'error': 'Data store not initialized'}), 500
    return jsonify(data_store.get_available_links())

@app_forecasting_bp.route('/api/traffic-data-forecast')
def get_traffic_data():
    try:
        if data_store is None:
            return jsonify({'error': 'Data store not initialized'}), 500
            
        link_id = request.args.get('link')
        print(f"Received request for traffic data. Link ID: {link_id}")
        
        data = data_store.get_traffic_data(link_id)
        
        if data is None:
            error_msg = f"No data available for {'overview' if link_id is None else f'link {link_id}'}"
            print(error_msg)
            return jsonify({'error': error_msg}), 404
            
        return jsonify(data)
        
    except Exception as e:
        error_msg = f"Error retrieving traffic data: {str(e)}"
        print(f"Error: {error_msg}")
        return jsonify({'error': error_msg}), 500
    
def update_cache_periodically():
    initial_delay = 300  # 5 minutes initial delay
    time.sleep(initial_delay)
    
    while True:
        try:
            if data_store.cache_manager.needs_update():
                print("\nScheduled cache update starting...")
                data_store.cache_manager.update_cache(data_store)
                print("Scheduled cache update completed")
            time.sleep(3600)  # Check every hour
        except Exception as e:
            print(f"Error in cache update thread: {str(e)}")
            time.sleep(300)  # Wait 5 minutes before retrying on error
