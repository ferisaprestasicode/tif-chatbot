# File: data_loader.py
import pandas as pd
import numpy as np
import os
from datetime import datetime

class RRDDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.links = {}
        self._load_data()
        
    def _load_data(self):
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")
                
            with open(self.file_path, 'r') as file:
                current_link = None
                current_data = []
                
                for line in file:
                    line = line.strip()
                    
                    if line.startswith("Data from"):
                        if current_link and current_data:
                            df = pd.DataFrame(current_data)
                            if not df.empty:
                                self.links[current_link] = df
                        
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
                            
                            if not (np.isnan(traffic_in) or np.isnan(traffic_out)):
                                current_data.append({
                                    'timestamp': pd.to_datetime(timestamp, unit='s'),
                                    'traffic_in': traffic_in,
                                    'traffic_out': traffic_out
                                })
                    except Exception as e:
                        print(f"Error parsing line: {str(e)}")
                        continue
                
                if current_link and current_data:
                    df = pd.DataFrame(current_data)
                    if not df.empty:
                        self.links[current_link] = df
                        
            print(f"Loaded {len(self.links)} links from RRD file")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            self.links = {}