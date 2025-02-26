from flask import Flask, Blueprint, jsonify, request, render_template
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import psycopg2
import psycopg2.extras
import pickle
import logging

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
app_anomalies_bp = Blueprint('app_anomalies', __name__, template_folder='templates')

class Config:
    DB_CONFIG = {
        'host': '36.67.62.245',
        'port': '8082',
        'database': 'sisai',
        'user': 'postgres',
        'password': 'uhuy123',
        'connect_timeout': 10,
        'application_name': 'network_monitoring'
    }
    LOG_FILE_PATH = r"data/processed/rrd_log_minutes.txt"
    NEAR_ZERO_THRESHOLD = 1e+06
    MODEL_BASE_PATH = os.path.join('model', 'anomalies', 'output', 'model_checkpoints')
    
class DatabaseManager:
    @staticmethod
    def get_connection():
        return psycopg2.connect(**Config.DB_CONFIG)

    @staticmethod
    def execute_query(query, params=None, fetch=True):
        conn = None
        try:
            conn = DatabaseManager.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cur.execute(query, params or ())
            
            if fetch:
                result = cur.fetchall()
            else:
                result = None
                conn.commit()
                
            cur.close()
            return result
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()

    @staticmethod
    def initialize_tables():
        queries = [
            """
            CREATE TABLE IF NOT EXISTS link_monitoring (
                id SERIAL PRIMARY KEY,
                monitoring_date DATE NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                link_id VARCHAR(255) NOT NULL,
                hostname VARCHAR(255),
                traffic_in FLOAT,
                traffic_out FLOAT,
                peak_traffic_in FLOAT,
                peak_traffic_out FLOAT,
                anomaly_count INTEGER,
                down_events INTEGER,
                degraded_events INTEGER,
                pattern_events INTEGER,
                model_events INTEGER,
                detection_method VARCHAR(50),
                status VARCHAR(50),
                details TEXT,
                CONSTRAINT idx_link_monitoring_unique UNIQUE (monitoring_date, link_id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_link_monitoring_link_id 
            ON link_monitoring(link_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_link_monitoring_date 
            ON link_monitoring(monitoring_date)
            """
        ]
        
        for query in queries:
            DatabaseManager.execute_query(query, fetch=False)

class PatternAnalyzer:
    def __init__(self, confidence_interval=2.0):
        self.confidence_interval = confidence_interval
        self.base_model_dir = Config.MODEL_BASE_PATH
        self.model_cache = {}

    def analyze_patterns(self, df):
        df = self._prepare_dataframe(df)
        return self._calculate_patterns(df)

    def _prepare_dataframe(self, df):
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['date'].dt.hour
        return df

    def _calculate_patterns(self, df):
        patterns = {}
        for direction in ['traffic_in', 'traffic_out']:
            hourly_stats = df.groupby('hour')[direction].agg(['mean', 'std', 'max', 'min'])
            patterns[direction] = self._calculate_direction_patterns(hourly_stats)
        return patterns

    def _calculate_direction_patterns(self, hourly_stats):
        direction_patterns = {}
        for hour in range(24):
            if hour in hourly_stats.index:
                stats = hourly_stats.loc[hour]
                direction_patterns[hour] = self._create_pattern_stats(stats)
            else:
                direction_patterns[hour] = self._create_empty_stats()
        return direction_patterns

    def _create_pattern_stats(self, stats):
        return {
            'mean': stats['mean'],
            'std': stats['std'],
            'max': stats['max'],
            'min': stats['min'],
            'upper': stats['mean'] + (self.confidence_interval * stats['std']),
            'lower': max(0, stats['mean'] - (self.confidence_interval * stats['std']))
        }

    def _create_empty_stats(self):
        return {
            'mean': 0, 'std': 0, 'max': 0, 'min': 0,
            'upper': 0, 'lower': 0
        }

    def detect_anomalies(self, df, patterns, hostname=None, link_id=None):
        df = df.copy()
        df = self._prepare_data_for_detection(df)
        
        anomalies = []
        detection_methods = {'MODEL': 0, 'PATTERN': 0, 'DOWN': 0, 'DEGRADED': 0}
        
        for idx, row in df.iterrows():
            for direction in ['traffic_in', 'traffic_out']:
                anomaly = self._detect_single_anomaly(
                    row, direction, patterns[direction][row['hour']],
                    hostname, detection_methods
                )
                if anomaly:
                    anomalies.append(anomaly)
        
        return anomalies

    def _prepare_data_for_detection(self, df):
        for direction in ['traffic_in', 'traffic_out']:
            df[f'{direction}_prev'] = df[direction].shift(1)
            df[f'{direction}_pct_change'] = (
                (df[direction] - df[f'{direction}_prev']) / 
                df[f'{direction}_prev'] * 100
            ).replace([np.inf, -np.inf], np.nan)
        return df

    def _detect_single_anomaly(self, row, direction, pattern, hostname, detection_methods):
        current = row[direction]
        pct_change = row[f'{direction}_pct_change']
        
        base_anomaly = {
            'date': str(row['date']),
            'hostname': hostname,
            'direction': direction,
            'value': float(current / 1e9),
            'expected': float(pattern['mean'] / 1e9)
        }
        
        if current <= Config.NEAR_ZERO_THRESHOLD:
            return self._create_anomaly(base_anomaly, 'DOWN', 100.0, 'critical', 'triangle', '#dc3545')
        
        if not pd.isna(pct_change) and abs(pct_change) >= 50:
            return self._create_anomaly(base_anomaly, 'DEGRADED', abs(pct_change), 
                                      'high', 'star', '#ffc107')
        
        if pattern['mean'] > 0:
            percent_diff = ((current - pattern['mean']) / pattern['mean'] * 100)
            if abs(percent_diff) > 30:
                return self._create_anomaly(base_anomaly, 'PATTERN', abs(percent_diff),
                                          'medium', 'circle', '#6c757d')
        
        return None

    def _create_anomaly(self, base_anomaly, type_, percentage, severity, symbol, color):
        return {
            **base_anomaly,
            'percentage': float(percentage),
            'type': type_,
            'severity': severity,
            'symbol': symbol,
            'color': color,
            'details': self._get_anomaly_details(type_, percentage, base_anomaly['direction'])
        }

    def _get_anomaly_details(self, type_, percentage, direction):
        direction_name = direction.replace("_", " ").title()
        if type_ == 'DOWN':
            return f'{direction_name} is down'
        elif type_ == 'DEGRADED':
            return f'{direction_name} changed by {abs(percentage):.1f}%'
        else:
            return f'Abnormal {direction_name}: {abs(percentage):.1f}% from mean'

class TrafficDataStore:
    def __init__(self, log_file_path):
        self.links = {}
        self.pattern_analyzer = PatternAnalyzer()
        self._load_data(log_file_path)

    def _load_data(self, log_file_path):
        try:
            current_link = None
            current_data = []
            
            with open(log_file_path, 'r') as file:
                skip_header = False
                for line in file:
                    line = line.strip()
                    
                    if line.startswith('Data from'):
                        if current_link and current_data:
                            self._save_current_data(current_link, current_data)
                        current_link = line.split('/')[-1].split('.')[0]
                        current_data = []
                        skip_header = True
                        continue
                        
                    if skip_header:
                        skip_header = False
                        continue
                        
                    if line and not line.startswith('traffic_in'):
                        try:
                            timestamp_str, values = line.split(':')
                            timestamp = int(timestamp_str.strip())
                            traffic_in, traffic_out = map(float, values.strip().split())
                            
                            if not (np.isnan(traffic_in) or np.isnan(traffic_out)):
                                current_data.append({
                                    'timestamp': pd.to_datetime(timestamp, unit='s'),
                                    'traffic_in': traffic_in,
                                    'traffic_out': traffic_out
                                })
                        except Exception as e:
                            logging.debug(f"Error parsing line: {str(e)}")
                            continue

            if current_link and current_data:
                self._save_current_data(current_link, current_data)

        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            self.links = {}

    def _save_current_data(self, link_id, data):
        if link_id and data:
            df = pd.DataFrame(data)
            if not df.empty:
                self.links[link_id] = df

    def get_traffic_data(self, link_id=None, time_filter='2d'):
        df = self._get_filtered_dataframe(link_id, time_filter)
        if df is None:
            return None

        hostname = self.get_hostname_from_rrd(link_id) if link_id else None
        patterns = self.pattern_analyzer.analyze_patterns(df)
        anomalies = self.pattern_analyzer.detect_anomalies(df, patterns, hostname, link_id)
        stats = self.calculate_current_stats(df)

        return {
            'dates': df['timestamp'].tolist(),
            'trafficIn': (df['traffic_in'] / 1e9).tolist(),
            'trafficOut': (df['traffic_out'] / 1e9).tolist(),
            'stats': stats,
            'anomalies': anomalies,
            'timeRange': {
                'start': df['date'].min().strftime('%Y-%m-%d %H:%M:%S'),
                'end': df['date'].max().strftime('%Y-%m-%d %H:%M:%S')
            }
        }

    def _get_filtered_dataframe(self, link_id, time_filter):
        if link_id:
            df = self.links.get(link_id)
        else:
            if not self.links:
                return None
            dfs = []
            for link_df in self.links.values():
                df_copy = link_df.copy()
                dfs.append(df_copy)
            
            df = pd.concat(dfs)
            numeric_cols = ['traffic_in', 'traffic_out']
            df = df.groupby('timestamp')[numeric_cols].sum().reset_index()

        if df is None or df.empty:
            return None

        return self.get_filtered_data(df, time_filter)

    def get_filtered_data(self, df, time_filter):
        df['date'] = pd.to_datetime(df['timestamp'])
        now = df['date'].max()
        
        filter_hours = {
            '6h': 6,
            '12h': 12,
            '1d': 24,
            '2d': 48
        }.get(time_filter, 48)
        
        start_time = now - timedelta(hours=filter_hours)
        return df[df['date'] >= start_time].copy()

    def calculate_current_stats(self, df):
        try:
            df['date'] = pd.to_datetime(df['timestamp'])
            last_hour = df['date'].max() - pd.Timedelta(hours=1)
            current_hour = df[df['date'] >= last_hour]
            
            return {
                'current_hour': {
                    'in': float(current_hour['traffic_in'].mean() / 1e9),
                    'out': float(current_hour['traffic_out'].mean() / 1e9)
                },
                'peak': {
                    'in': float(df['traffic_in'].max() / 1e9),
                    'out': float(df['traffic_out'].max() / 1e9)
                }
            }
        except Exception as e:
            logging.error(f"Error calculating statistics: {str(e)}")
            return {
                'current_hour': {'in': 0, 'out': 0},
                'peak': {'in': 0, 'out': 0}
            }

    def get_hostname_from_rrd(self, link_id):
        query = """
        SELECT hostname 
        FROM cacti 
        WHERE data_source_path LIKE %s 
        LIMIT 1
        """
        result = DatabaseManager.execute_query(query, (f'%{link_id}%',))
        return result[0]['hostname'] if result else None

    def get_available_links(self):
        return [{"id": link_name, "name": link_name} for link_name in self.links.keys()]

# Initialize data store
data_store = TrafficDataStore(Config.LOG_FILE_PATH)

# Routes
@app_anomalies_bp.route('/')
def anomalies():
    return render_template('app_anomalies.html')

@app_anomalies_bp.route('/api/links')
def get_links():
    return jsonify(data_store.get_available_links())

@app_anomalies_bp.route('/api/error-logs')
def get_error_logs():
    link_id = request.args.get('hostname')
    timestamp = request.args.get('timestamp')
    
    try:
        hostname = link_id.split('_')[0].upper()
        query_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        start_time = query_time - timedelta(minutes=10)
        end_time = query_time + timedelta(minutes=10)

        query = """
        SELECT _time, hostname, event_desc3, event_desc_last, _raw
        FROM network_logs
        WHERE (hostname = %s OR hostname = %s)
        AND _time BETWEEN %s::timestamp AND %s::timestamp
        ORDER BY _time
        """
        
        # Try both uppercase and original format
        results = DatabaseManager.execute_query(
            query, 
            (hostname, link_id.split('_')[0], start_time, end_time)
        )

        if not results:
            logging.info(f"No logs found for hostname: {hostname} or {link_id.split('_')[0]}")
            
        return jsonify([{
            'timestamp': row['_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'device': row['hostname'],
            'message': row['event_desc3'],
            'details': row['event_desc_last'],
            'raw': row['_raw']
        } for row in results])
        
    except Exception as e:
        logging.error(f"Error fetching logs: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app_anomalies_bp.route('/api/traffic-data')
def get_traffic_data():
    link_id = request.args.get('link')
    time_filter = request.args.get('timeFilter', '2d')
    data = data_store.get_traffic_data(link_id, time_filter)
    if data is None:
        return jsonify({'error': 'No data available'}), 500
    return jsonify(data)

@app_anomalies_bp.route('/api/monitoring', methods=['GET', 'POST'])
def handle_monitoring():
    if request.method == 'POST':
        return handle_monitoring_post()
    return handle_monitoring_get()

def handle_monitoring_post():
    link_id = request.args.get('link')
    if link_id:
        success = save_monitoring_data(link_id, data_store)
        return jsonify({
            'status': 'success' if success else 'error',
            'message': f'Monitoring data {"saved" if success else "failed to save"} for {link_id}'
        })
    
    results = [{
        'link': link_id['id'], 
        'success': save_monitoring_data(link_id['id'], data_store)
    } for link_id in data_store.get_available_links()]
    return jsonify({'results': results})

def handle_monitoring_get():
    try:
        link_id = request.args.get('link')
        limit = int(request.args.get('limit', 500))
        
        query = """
        SELECT * FROM network_monitoring 
        {}
        ORDER BY date_recorded DESC 
        LIMIT %s
        """.format("WHERE link_id = %s" if link_id else "")
        
        params = (link_id, limit) if link_id else (limit,)
        results = DatabaseManager.execute_query(query, params)
        
        return jsonify([dict(row) for row in results])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app_anomalies_bp.route('/api/verify-monitoring/<int:record_id>')
def verify_monitoring(record_id):
    query = "SELECT * FROM network_monitoring WHERE id = %s"
    result = DatabaseManager.execute_query(query, (record_id,))
    return jsonify(dict(result[0])) if result else (jsonify({'error': 'Record not found'}), 404)

@app_anomalies_bp.route('/api/update', methods=['POST'])
def force_update():
    try:
        link_id = request.args.get('link')
        if link_id:
            success = save_link_data(link_id, data_store)
            return jsonify({
                'status': 'success' if success else 'error',
                'message': f'Monitoring data {"updated" if success else "failed to update"} for {link_id}'
            })
        
        success_count, error_count = save_all_links(data_store)
        return jsonify({
            'status': 'success',
            'message': f'Updated {success_count} links, {error_count} failed'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def save_monitoring_data(link_id, data_store):
    try:
        df = data_store.links.get(link_id)
        if df is None or df.empty:
            return False
            
        hostname = data_store.get_hostname_from_rrd(link_id)
        if not hostname:
            return False
            
        patterns = data_store.pattern_analyzer.analyze_patterns(df)
        anomalies = data_store.pattern_analyzer.detect_anomalies(df, patterns, hostname, link_id)
        stats = data_store.calculate_current_stats(df)
        
        error_counts = {
            'down_errors': len([a for a in anomalies if a['type'] == 'DOWN']),
            'degraded_errors': len([a for a in anomalies if a['type'] == 'DEGRADED']),
            'pattern_errors': len([a for a in anomalies if a['type'] == 'PATTERN']),
            'model_errors': len([a for a in anomalies if a['type'] == 'MODEL'])
        }
        
        status = 'critical' if error_counts['down_errors'] > 0 else \
                'moderate' if error_counts['degraded_errors'] > 0 else \
                'warning' if anomalies else 'good'
        
        query = """
        INSERT INTO network_monitoring (
            link_id, hostname, down_errors, degraded_errors,
            pattern_errors, model_errors, model_type, confidence_level,
            status, traffic_in_gbps, traffic_out_gbps,
            peak_traffic_in_gbps, peak_traffic_out_gbps,
            total_anomalies, detection_method
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) RETURNING id
        """
        
        values = (
            link_id, hostname,
            error_counts['down_errors'],
            error_counts['degraded_errors'],
            error_counts['pattern_errors'],
            error_counts['model_errors'],
            'Exponential_Smoothing',
            float(data_store.pattern_analyzer.confidence_interval),
            status,
            float(stats['current_hour']['in']),
            float(stats['current_hour']['out']),
            float(stats['peak']['in']),
            float(stats['peak']['out']),
            len(anomalies),
            'hybrid'
        )
        
        DatabaseManager.execute_query(query, values, fetch=False)
        return True
        
    except Exception as e:
        logging.error(f"Error in save_monitoring_data: {str(e)}")
        return False

def save_link_data(link_id, data_store):
    try:
        df = data_store.links.get(link_id)
        if df is None or df.empty:
            return False
            
        hostname = data_store.get_hostname_from_rrd(link_id)
        if not hostname:
            return False
            
        monitoring_date = datetime.now().date()
        patterns = data_store.pattern_analyzer.analyze_patterns(df)
        anomalies = data_store.pattern_analyzer.detect_anomalies(df, patterns, hostname, link_id)
        stats = data_store.calculate_current_stats(df)
        
        anomaly_counts = {
            'down': len([a for a in anomalies if a['type'] == 'DOWN']),
            'degraded': len([a for a in anomalies if a['type'] == 'DEGRADED']),
            'pattern': len([a for a in anomalies if a['type'] == 'PATTERN']),
            'model': len([a for a in anomalies if a['type'] == 'MODEL'])
        }
        
        detection_method = 'model' if anomaly_counts['model'] > anomaly_counts['pattern'] else \
                         'pattern' if anomaly_counts['pattern'] > anomaly_counts['model'] else 'hybrid'
        
        status = 'critical' if anomaly_counts['down'] > 0 else \
                'warning' if anomaly_counts['degraded'] > 0 else \
                'normal' if sum(anomaly_counts.values()) > 0 else 'good'
        
        query = """
        INSERT INTO link_monitoring (
            monitoring_date, link_id, hostname, 
            traffic_in, traffic_out,
            peak_traffic_in, peak_traffic_out, 
            anomaly_count,
            down_events, degraded_events, 
            pattern_events, model_events,
            detection_method,
            status, details
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s
        )
        ON CONFLICT (monitoring_date, link_id) 
        DO UPDATE SET
            timestamp = CURRENT_TIMESTAMP,
            hostname = EXCLUDED.hostname,
            traffic_in = EXCLUDED.traffic_in,
            traffic_out = EXCLUDED.traffic_out,
            peak_traffic_in = EXCLUDED.peak_traffic_in,
            peak_traffic_out = EXCLUDED.peak_traffic_out,
            anomaly_count = EXCLUDED.anomaly_count,
            down_events = EXCLUDED.down_events,
            degraded_events = EXCLUDED.degraded_events,
            pattern_events = EXCLUDED.pattern_events,
            model_events = EXCLUDED.model_events,
            detection_method = EXCLUDED.detection_method,
            status = EXCLUDED.status,
            details = EXCLUDED.details
        """
        
        values = (
            monitoring_date,
            link_id,
            hostname,
            float(stats['current_hour']['in']),
            float(stats['current_hour']['out']),
            float(stats['peak']['in']),
            float(stats['peak']['out']),
            len(anomalies),
            anomaly_counts['down'],
            anomaly_counts['degraded'],
            anomaly_counts['pattern'],
            anomaly_counts['model'],
            detection_method,
            status,
            str(anomalies)
        )
        
        DatabaseManager.execute_query(query, values, fetch=False)
        return True
        
    except Exception as e:
        logging.error(f"Error saving link data: {str(e)}")
        return False

def save_all_links(data_store):
    success_count = error_count = 0
    for link in data_store.get_available_links():
        try:
            if save_link_data(link['id'], data_store):
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            logging.error(f"Error processing link {link['id']}: {str(e)}")
            error_count += 1
    return success_count, error_count

# Add these new routes to your Flask application
@app_anomalies_bp.route('/api/nodes')
def get_nodes():
    """Get all unique node1 values from weathermap"""
    try:
        query = """
        SELECT DISTINCT node1 as node
        FROM weathermap
        WHERE node1 IS NOT NULL
        ORDER BY node1
        """
        results = DatabaseManager.execute_query(query)
        if not results:
            return jsonify([])
        return jsonify([row['node'] for row in results])
    except Exception as e:
        logging.error(f"Error fetching nodes: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app_anomalies_bp.route('/api/related-nodes')
def get_related_nodes():
    """Get related node2 values based on selected node1"""
    node1 = request.args.get('node1')
    if not node1:
        return jsonify([])
    
    try:
        query = """
        SELECT DISTINCT node2 as node
        FROM weathermap
        WHERE node1 = %s AND node2 IS NOT NULL
        ORDER BY node2
        """
        results = DatabaseManager.execute_query(query, (node1,))
        if not results:
            return jsonify([])
        return jsonify([row['node'] for row in results])
    except Exception as e:
        logging.error(f"Error fetching related nodes: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app_anomalies_bp.route('/api/related-links')
def get_related_links():
    """Get related link names based on selected node1 and node2"""
    node1 = request.args.get('node1')
    node2 = request.args.get('node2')
    if not node1 or not node2:
        return jsonify([])
    
    try:
        query = """
        SELECT DISTINCT link_name
        FROM weathermap
        WHERE node1 = %s 
        AND node2 = %s 
        AND link_name IS NOT NULL
        ORDER BY link_name
        """
        results = DatabaseManager.execute_query(query, (node1, node2))
        if not results:
            return jsonify([])
        return jsonify([row['link_name'] for row in results])
    except Exception as e:
        logging.error(f"Error fetching related links: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app_anomalies_bp.route('/api/rrd-links')
def get_rrd_links():
    """Get RRD links based on selected node1 and node2"""
    node1 = request.args.get('node1')
    node2 = request.args.get('node2')
    if not node1 or not node2:
        return jsonify([])
    
    try:
        query = """
        SELECT DISTINCT target
        FROM weathermap
        WHERE node1 = %s 
        AND node2 = %s
        AND target IS NOT NULL
        """
        results = DatabaseManager.execute_query(query, (node1, node2))
        if not results:
            return jsonify([])
            
        rrd_links = []
        for row in results:
            try:
                rrd_file = row['target'].split('/')[-1].replace('.rrd', '')
                if rrd_file:
                    rrd_links.append(rrd_file)
            except:
                continue
                
        return jsonify(rrd_links)
    except Exception as e:
        logging.error(f"Error fetching RRD links: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
# Add this debugging route to check the database connection and data
@app_anomalies_bp.route('/api/debug/weathermap')
def debug_weathermap():
    """Debug endpoint to check weathermap table data"""
    try:
        query = """
        SELECT COUNT(*) as total,
               COUNT(DISTINCT node1) as unique_node1,
               COUNT(DISTINCT node2) as unique_node2,
               COUNT(DISTINCT link_name) as unique_links,
               COUNT(DISTINCT target) as unique_targets
        FROM weathermap
        """
        results = DatabaseManager.execute_query(query)
        return jsonify(dict(results[0]))
    except Exception as e:
        logging.error(f"Error in debug route: {str(e)}")
        return jsonify({'error': str(e)}), 500
 
@app_anomalies_bp.route('/api/ground-truth-stats')
def get_ground_truth_stats():
    try:
        # Check if table exists
        check_table_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'anomaly_ground_truth'
        )
        """
        table_exists = DatabaseManager.execute_query(check_table_query)[0][0]
        
        if not table_exists:
            return jsonify({
                'total': 0,
                'truePositives': 0,
                'falsePositives': 0,
                'accuracy': 0.0
            })

        # Get verification statistics
        stats_query = """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN is_true_anomaly THEN 1 ELSE 0 END) as true_positives,
            SUM(CASE WHEN NOT is_true_anomaly THEN 1 ELSE 0 END) as false_positives,
            ROUND(
                (SUM(CASE WHEN is_true_anomaly THEN 1 ELSE 0 END)::float / COUNT(*)::float * 100)::numeric,
                1
            ) as accuracy
        FROM anomaly_ground_truth
        WHERE verification_timestamp >= NOW() - INTERVAL '30 days'
        """
        
        result = DatabaseManager.execute_query(stats_query)
        stats = result[0]
        
        return jsonify({
            'total': stats['total'] or 0,
            'truePositives': stats['true_positives'] or 0,
            'falsePositives': stats['false_positives'] or 0,
            'accuracy': float(stats['accuracy'] or 0.0)
        })
        
    except Exception as e:
        logging.error(f"Error getting ground truth stats: {str(e)}")
        return jsonify({
            'total': 0,
            'truePositives': 0,
            'falsePositives': 0,
            'accuracy': 0.0
        })

def initialize_ground_truth():
    try:
        # Updated schema to include the comment column
        create_table_query = """
        CREATE TABLE IF NOT EXISTS anomaly_ground_truth (
            id SERIAL PRIMARY KEY,
            anomaly_timestamp TIMESTAMP NOT NULL,
            link_id VARCHAR(255) NOT NULL,
            anomaly_type VARCHAR(50) NOT NULL,
            actual_value FLOAT NOT NULL,
            expected_value FLOAT NOT NULL,
            is_true_anomaly BOOLEAN NOT NULL,
            verification_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            detection_method VARCHAR(50),
            comment TEXT  -- Added comment column
        );
        """
        
        # Check if comment column exists
        check_column_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_name = 'anomaly_ground_truth' 
            AND column_name = 'comment'
        );
        """
        
        # Add comment column if it doesn't exist
        add_column_query = """
        ALTER TABLE anomaly_ground_truth 
        ADD COLUMN IF NOT EXISTS comment TEXT;
        """
        
        create_index_query = """
        CREATE INDEX IF NOT EXISTS idx_ground_truth_timestamp 
        ON anomaly_ground_truth(verification_timestamp);
        CREATE INDEX IF NOT EXISTS idx_ground_truth_link 
        ON anomaly_ground_truth(link_id);
        """
        
        # Execute queries
        DatabaseManager.execute_query(create_table_query, fetch=False)
        column_exists = DatabaseManager.execute_query(check_column_query)[0][0]
        
        if not column_exists:
            DatabaseManager.execute_query(add_column_query, fetch=False)
            
        DatabaseManager.execute_query(create_index_query, fetch=False)
        logging.info("Ground truth table initialized successfully")
        
    except Exception as e:
        logging.error(f"Error initializing ground truth table: {str(e)}")
        
# Update the verify-anomaly route
@app_anomalies_bp.route('/api/verify-anomaly', methods=['POST'])
def verify_anomaly():
    try:
        data = request.json
        
        # Insert the verification data
        insert_query = """
        INSERT INTO anomaly_ground_truth (
            anomaly_timestamp,
            link_id,
            anomaly_type,
            actual_value,
            expected_value,
            is_true_anomaly,
            detection_method,
            comment
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        values = (
            data['timestamp'],
            data['linkId'],
            data['anomalyType'],
            data['value'],
            data['expected'],
            data['isTrue'],
            'hybrid',
            data.get('comment', '')  # Add comment field
        )
        
        DatabaseManager.execute_query(insert_query, values, fetch=False)
        return jsonify({'status': 'success'})
        
    except Exception as e:
        logging.error(f"Error verifying anomaly: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
# Call this during application startup
initialize_ground_truth()

# Initialize tables on import
DatabaseManager.initialize_tables()


## add minio config
from flask import Flask, Blueprint, jsonify, request, render_template
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import warnings
# import os
# import psycopg2
# import psycopg2.extras
# import pickle
# import logging
# from minio import Minio
# from io import BytesIO
# import json

# logging.basicConfig(level=logging.INFO,
#                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# app_anomalies_bp = Blueprint('app_anomalies', __name__, template_folder='templates')

# class Config:
#     DB_CONFIG = {
#         'host': '36.67.62.245',
#         'port': '8082',
#         'database': 'sisai',
#         'user': 'postgres',
#         'password': 'uhuy123',
#         'connect_timeout': 10,
#         'application_name': 'network_monitoring'
#     }
    
#     # MinIO configuration
#     MINIO_ENDPOINT = "10.62.179.53:9000"
#     MINIO_ACCESS_KEY = "pgF2QVgkdP5TucUv34Zm"
#     MINIO_SECRET_KEY = "tzCS5tVg6W8XpAXYSMoXMDRdo0W83IpTOhvFE0LH"
#     MINIO_SECURE = False
#     MINIO_BUCKET = "traffic-data"
    
#     # MinIO paths
#     MINIO_RRD_LOG = "processed/rrd_log_minutes.txt"
#     MINIO_LINK_DATA = "processed/link_data"
#     MINIO_MODELS = "models"
    
#     NEAR_ZERO_THRESHOLD = 1e+06

# class DatabaseManager:
#     @staticmethod
#     def get_connection():
#         return psycopg2.connect(**Config.DB_CONFIG)

#     @staticmethod
#     def execute_query(query, params=None, fetch=True):
#         conn = None
#         try:
#             conn = DatabaseManager.get_connection()
#             cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
#             cur.execute(query, params or ())
            
#             if fetch:
#                 result = cur.fetchall()
#             else:
#                 result = None
#                 conn.commit()
                
#             cur.close()
#             return result
#         except Exception as e:
#             if conn:
#                 conn.rollback()
#             raise e
#         finally:
#             if conn:
#                 conn.close()

#     @staticmethod
#     def initialize_tables():
#         queries = [
#             """
#             CREATE TABLE IF NOT EXISTS link_monitoring (
#                 id SERIAL PRIMARY KEY,
#                 monitoring_date DATE NOT NULL,
#                 timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 link_id VARCHAR(255) NOT NULL,
#                 hostname VARCHAR(255),
#                 traffic_in FLOAT,
#                 traffic_out FLOAT,
#                 peak_traffic_in FLOAT,
#                 peak_traffic_out FLOAT,
#                 anomaly_count INTEGER,
#                 down_events INTEGER,
#                 degraded_events INTEGER,
#                 pattern_events INTEGER,
#                 model_events INTEGER,
#                 detection_method VARCHAR(50),
#                 status VARCHAR(50),
#                 details TEXT,
#                 CONSTRAINT idx_link_monitoring_unique UNIQUE (monitoring_date, link_id)
#             )
#             """
#         ]
        
#         for query in queries:
#             DatabaseManager.execute_query(query, fetch=False)

# class MinioHandler:
#     _instance = None

#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super(MinioHandler, cls).__new__(cls)
#             cls._instance.initialize()
#         return cls._instance

#     def initialize(self):
#         self.client = Minio(
#             Config.MINIO_ENDPOINT,
#             access_key=Config.MINIO_ACCESS_KEY,
#             secret_key=Config.MINIO_SECRET_KEY,
#             secure=Config.MINIO_SECURE
#         )
#         self.ensure_bucket_exists()
    
#     def ensure_bucket_exists(self):
#         try:
#             if not self.client.bucket_exists(Config.MINIO_BUCKET):
#                 self.client.make_bucket(Config.MINIO_BUCKET)
#                 logging.info(f"Created MinIO bucket: {Config.MINIO_BUCKET}")
#         except Exception as e:
#             logging.error(f"Error ensuring bucket exists: {e}")
    
#     def read_file(self, path):
#         try:
#             data = self.client.get_object(Config.MINIO_BUCKET, path)
#             return data.read().decode('utf-8')
#         except Exception as e:
#             logging.error(f"Error reading from MinIO: {e}")
#             return None
            
#     def write_file(self, content, path):
#         try:
#             if isinstance(content, str):
#                 data_bytes = content.encode('utf-8')
#             else:
#                 data_bytes = content
                
#             data_stream = BytesIO(data_bytes)
#             self.client.put_object(
#                 Config.MINIO_BUCKET,
#                 path,
#                 data_stream,
#                 len(data_bytes)
#             )
#             return True
#         except Exception as e:
#             logging.error(f"Error writing to MinIO: {e}")
#             return False
    
#     def file_exists(self, path):
#         try:
#             self.client.stat_object(Config.MINIO_BUCKET, path)
#             return True
#         except:
#             return False

# class PatternAnalyzer:
#     def __init__(self, confidence_interval=2.0):
#         self.confidence_interval = confidence_interval
#         self.minio = MinioHandler()
        
#     def analyze_patterns(self, df):
#         df = self._prepare_dataframe(df)
#         return self._calculate_patterns(df)

#     def _prepare_dataframe(self, df):
#         if 'date' not in df.columns:
#             df['date'] = pd.to_datetime(df['timestamp'])
#         df['hour'] = df['date'].dt.hour
#         return df

#     def _calculate_patterns(self, df):
#         patterns = {}
#         for direction in ['traffic_in', 'traffic_out']:
#             hourly_stats = df.groupby('hour')[direction].agg(['mean', 'std', 'max', 'min'])
#             patterns[direction] = self._calculate_direction_patterns(hourly_stats)
#         return patterns

#     def _calculate_direction_patterns(self, hourly_stats):
#         direction_patterns = {}
#         for hour in range(24):
#             if hour in hourly_stats.index:
#                 stats = hourly_stats.loc[hour]
#                 direction_patterns[hour] = self._create_pattern_stats(stats)
#             else:
#                 direction_patterns[hour] = self._create_empty_stats()
#         return direction_patterns

#     def _create_pattern_stats(self, stats):
#         return {
#             'mean': stats['mean'],
#             'std': stats['std'],
#             'max': stats['max'],
#             'min': stats['min'],
#             'upper': stats['mean'] + (self.confidence_interval * stats['std']),
#             'lower': max(0, stats['mean'] - (self.confidence_interval * stats['std']))
#         }

#     def _create_empty_stats(self):
#         return {
#             'mean': 0, 'std': 0, 'max': 0, 'min': 0,
#             'upper': 0, 'lower': 0
#         }

#     def detect_anomalies(self, df, patterns, hostname=None, link_id=None):
#         df = df.copy()
#         df = self._prepare_data_for_detection(df)
#         anomalies = []
        
#         for idx, row in df.iterrows():
#             for direction in ['traffic_in', 'traffic_out']:
#                 anomaly = self._detect_single_anomaly(
#                     row, direction, patterns[direction][row['hour']],
#                     hostname
#                 )
#                 if anomaly:
#                     anomalies.append(anomaly)
        
#         return anomalies

#     def _prepare_data_for_detection(self, df):
#         for direction in ['traffic_in', 'traffic_out']:
#             df[f'{direction}_prev'] = df[direction].shift(1)
#             df[f'{direction}_pct_change'] = (
#                 (df[direction] - df[f'{direction}_prev']) / 
#                 df[f'{direction}_prev'] * 100
#             ).replace([np.inf, -np.inf], np.nan)
#         return df

#     def _detect_single_anomaly(self, row, direction, pattern, hostname):
#         current = row[direction]
#         pct_change = row[f'{direction}_pct_change']
        
#         base_anomaly = {
#             'date': str(row['date']),
#             'hostname': hostname,
#             'direction': direction,
#             'value': float(current / 1e9),
#             'expected': float(pattern['mean'] / 1e9)
#         }
        
#         if current <= Config.NEAR_ZERO_THRESHOLD:
#             return self._create_anomaly(base_anomaly, 'DOWN', 100.0, 'critical', 'triangle', '#dc3545')
        
#         if not pd.isna(pct_change) and abs(pct_change) >= 50:
#             return self._create_anomaly(base_anomaly, 'DEGRADED', abs(pct_change), 
#                                       'high', 'star', '#ffc107')
        
#         if pattern['mean'] > 0:
#             percent_diff = ((current - pattern['mean']) / pattern['mean'] * 100)
#             if abs(percent_diff) > 30:
#                 return self._create_anomaly(base_anomaly, 'PATTERN', abs(percent_diff),
#                                           'medium', 'circle', '#6c757d')
        
#         return None

#     def _create_anomaly(self, base_anomaly, type_, percentage, severity, symbol, color):
#         return {
#             **base_anomaly,
#             'percentage': float(percentage),
#             'type': type_,
#             'severity': severity,
#             'symbol': symbol,
#             'color': color,
#             'details': self._get_anomaly_details(type_, percentage, base_anomaly['direction'])
#         }

#     def _get_anomaly_details(self, type_, percentage, direction):
#         direction_name = direction.replace("_", " ").title()
#         if type_ == 'DOWN':
#             return f'{direction_name} is down'
#         elif type_ == 'DEGRADED':
#             return f'{direction_name} changed by {abs(percentage):.1f}%'
#         else:
#             return f'Abnormal {direction_name}: {abs(percentage):.1f}% from mean'

# class TrafficDataStore:
#     def __init__(self):
#         self.links = {}
#         self.pattern_analyzer = PatternAnalyzer()
#         self.minio = MinioHandler()
#         self._load_data()

#     def _load_data(self):
#         try:
#             data = self.minio.read_file(Config.MINIO_RRD_LOG)
#             if not data:
#                 logging.error("No data found in MinIO")
#                 return

#             current_link = None
#             current_data = []
            
#             for line in data.split('\n'):
#                 line = line.strip()
                
#                 if line.startswith('Data from'):
#                     if current_link and current_data:
#                         self._save_current_data(current_link, current_data)
#                     current_link = line.split('/')[-1].split('.')[0]
#                     current_data = []
#                     continue
                    
#                 if line and not line.startswith('traffic_in'):
#                     try:
#                         timestamp_str, values = line.split(':')
#                         timestamp = int(timestamp_str.strip())
#                         traffic_in, traffic_out = map(float, values.strip().split())
                        
#                         if not (np.isnan(traffic_in) or np.isnan(traffic_out)):
#                             current_data.append({
#                                 'timestamp': pd.to_datetime(timestamp, unit='s'),
#                                 'traffic_in': traffic_in,
#                                 'traffic_out': traffic_out
#                             })
#                     except Exception as e:
#                         continue

#             if current_link and current_data:
#                 self._save_current_data(current_link, current_data)

#         except Exception as e:
#             logging.error(f"Error loading data: {str(e)}")
#             self.links = {}

#     def _save_current_data(self, link_id, data):
#         if link_id and data:
#             df = pd.DataFrame(data)
#             if not df.empty:
#                 self.links[link_id] = df
#                 json_data = df.to_json(orient='records', date_format='iso')
#                 self.minio.write_file(
#                     json_data,
#                     f"{Config.MINIO_LINK_DATA}/{link_id}_data.json"
#                 )

#     def get_traffic_data(self, link_id=None, time_filter='2d'):
#         df = self._get_filtered_dataframe(link_id, time_filter)
#         if df is None:
#             return None

#         hostname = self.get_hostname_from_rrd(link_id) if link_id else None
#         patterns = self.pattern_analyzer.analyze_patterns(df)
#         anomalies = self.pattern_analyzer.detect_anomalies(df, patterns, hostname, link_id)
#         stats = self.calculate_current_stats(df)

#         return {
#             'dates': df['timestamp'].tolist(),
#             'trafficIn': (df['traffic_in'] / 1e9).tolist(),
#             'trafficOut': (df['traffic_out'] / 1e9).tolist(),
#             'stats': stats,
#             'anomalies': anomalies,
#             'timeRange': {
#                 'start': df['date'].min().strftime('%Y-%m-%d %H:%M:%S'),
#                 'end': df['date'].max().strftime('%Y-%m-%d %H:%M:%S')
#             }
#         }

#     def _get_filtered_dataframe(self, link_id, time_filter):
#         if link_id:
#             df = self.links.get(link_id)
#         else:
#             if not self.links:
#                 return None
#             dfs = []
#             for link_df in self.links.values():
#                 df_copy = link_df.copy()
#                 dfs.append(df_copy)
            
#             df = pd.concat(dfs)
#             numeric_cols = ['traffic_in', 'traffic_out']
#             df = df.groupby('timestamp')[numeric_cols].sum().reset_index()

#         if df is None or df.empty:
#             return None

#         return self.get_filtered_data(df, time_filter)

#     def get_filtered_data(self, df, time_filter):
#         df['date'] = pd.to_datetime(df['timestamp'])
#         now = df['date'].max()
        
#         filter_hours = {
#             '6h': 6,
#             '12h': 12,
#             '1d': 24,
#             '2d': 48
#         }.get(time_filter, 48)
        
#         start_time = now - timedelta(hours=filter_hours)
#         return df[df['date'] >= start_time].copy()

#     def calculate_current_stats(self, df):
#         try:
#             df['date'] = pd.to_datetime(df['timestamp'])
#             last_hour = df['date'].max() - pd.Timedelta(hours=1)
#             current_hour = df[df['date'] >= last_hour]
            
#             return {
#                 'current_hour': {
#                     'in': float(current_hour['traffic_in'].mean() / 1e9),
#                     'out': float(current_hour['traffic_out'].mean() / 1e9)
#                 },
#                 'peak': {
#                     'in': float(df['traffic_in'].max() / 1e9),
#                     'out': float(df['traffic_out'].max() / 1e9)
#                 }
#             }
#         except Exception as e:
#             logging.error(f"Error calculating statistics: {str(e)}")
#             return {
#                 'current_hour': {'in': 0, 'out': 0},
#                 'peak': {'in': 0, 'out': 0}
#             }

#     def get_hostname_from_rrd(self, link_id):
#         query = """
#         SELECT hostname 
#         FROM cacti 
#         WHERE data_source_path LIKE %s 
#         LIMIT 1
#         """
#         result = DatabaseManager.execute_query(query, (f'%{link_id}%',))
#         return result[0]['hostname'] if result else None

#     def get_available_links(self):
#         return [{"id": link_name, "name": link_name} for link_name in self.links.keys()]

# # Initialize MinIO and database
# minio_handler = MinioHandler()
# DatabaseManager.initialize_tables()

# # Initialize data store
# data_store = TrafficDataStore()

# # Routes
# @app_anomalies_bp.route('/')
# def anomalies():
#     return render_template('app_anomalies.html')

# @app_anomalies_bp.route('/api/links')
# def get_links():
#     return jsonify(data_store.get_available_links())

# @app_anomalies_bp.route('/api/error-logs')
# def get_error_logs():
#     link_id = request.args.get('hostname')
#     timestamp = request.args.get('timestamp')
    
#     try:
#         hostname = link_id.split('_')[0].upper()
#         query_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
#         start_time = query_time - timedelta(minutes=10)
#         end_time = query_time + timedelta(minutes=10)

#         query = """
#         SELECT _time, hostname, event_desc3, event_desc_last, _raw
#         FROM network_logs
#         WHERE (hostname = %s OR hostname = %s)
#         AND _time BETWEEN %s::timestamp AND %s::timestamp
#         ORDER BY _time
#         """
        
#         results = DatabaseManager.execute_query(
#             query, 
#             (hostname, link_id.split('_')[0], start_time, end_time)
#         )

#         return jsonify([{
#             'timestamp': row['_time'].strftime('%Y-%m-%d %H:%M:%S'),
#             'device': row['hostname'],
#             'message': row['event_desc3'],
#             'details': row['event_desc_last'],
#             'raw': row['_raw']
#         } for row in results])
        
#     except Exception as e:
#         logging.error(f"Error fetching logs: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app_anomalies_bp.route('/api/traffic-data')
# def get_traffic_data():
#     link_id = request.args.get('link')
#     time_filter = request.args.get('timeFilter', '2d')
#     data = data_store.get_traffic_data(link_id, time_filter)
#     if data is None:
#         return jsonify({'error': 'No data available'}), 500
#     return jsonify(data)

# @app_anomalies_bp.route('/api/monitoring', methods=['GET', 'POST'])
# def handle_monitoring():
#     if request.method == 'POST':
#         return handle_monitoring_post()
#     return handle_monitoring_get()

# def handle_monitoring_post():
#     link_id = request.args.get('link')
#     if link_id:
#         success = save_monitoring_data(link_id, data_store)
#         return jsonify({
#             'status': 'success' if success else 'error',
#             'message': f'Monitoring data {"saved" if success else "failed to save"} for {link_id}'
#         })
    
#     results = [{
#         'link': link_id['id'], 
#         'success': save_monitoring_data(link_id['id'], data_store)
#     } for link_id in data_store.get_available_links()]
#     return jsonify({'results': results})

# def handle_monitoring_get():
#     try:
#         link_id = request.args.get('link')
#         limit = int(request.args.get('limit', 500))
        
#         query = """
#         SELECT * FROM link_monitoring 
#         {}
#         ORDER BY monitoring_date DESC 
#         LIMIT %s
#         """.format("WHERE link_id = %s" if link_id else "")
        
#         params = (link_id, limit) if link_id else (limit,)
#         results = DatabaseManager.execute_query(query, params)
        
#         return jsonify([dict(row) for row in results])
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# def save_monitoring_data(link_id, data_store):
#     try:
#         df = data_store.links.get(link_id)
#         if df is None or df.empty:
#             return False
            
#         hostname = data_store.get_hostname_from_rrd(link_id)
#         if not hostname:
#             return False
            
#         patterns = data_store.pattern_analyzer.analyze_patterns(df)
#         anomalies = data_store.pattern_analyzer.detect_anomalies(df, patterns, hostname, link_id)
#         stats = data_store.calculate_current_stats(df)
        
#         monitoring_date = datetime.now().date()
#         anomaly_counts = {
#             'down': len([a for a in anomalies if a['type'] == 'DOWN']),
#             'degraded': len([a for a in anomalies if a['type'] == 'DEGRADED']),
#             'pattern': len([a for a in anomalies if a['type'] == 'PATTERN']),
#             'model': len([a for a in anomalies if a['type'] == 'MODEL'])
#         }
        
#         status = 'critical' if anomaly_counts['down'] > 0 else \
#                 'warning' if anomaly_counts['degraded'] > 0 else \
#                 'normal' if sum(anomaly_counts.values()) > 0 else 'good'
        
#         query = """
#         INSERT INTO link_monitoring (
#             monitoring_date, link_id, hostname,
#             traffic_in, traffic_out,
#             peak_traffic_in, peak_traffic_out,
#             anomaly_count, down_events, degraded_events,
#             pattern_events, model_events,
#             status, details
#         ) VALUES (
#             %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
#         )
#         ON CONFLICT (monitoring_date, link_id) DO UPDATE SET
#             timestamp = CURRENT_TIMESTAMP,
#             hostname = EXCLUDED.hostname,
#             traffic_in = EXCLUDED.traffic_in,
#             traffic_out = EXCLUDED.traffic_out,
#             peak_traffic_in = EXCLUDED.peak_traffic_in,
#             peak_traffic_out = EXCLUDED.peak_traffic_out,
#             anomaly_count = EXCLUDED.anomaly_count,
#             down_events = EXCLUDED.down_events,
#             degraded_events = EXCLUDED.degraded_events,
#             pattern_events = EXCLUDED.pattern_events,
#             model_events = EXCLUDED.model_events,
#             status = EXCLUDED.status,
#             details = EXCLUDED.details
#         """
        
#         values = (
#             monitoring_date, link_id, hostname,
#             float(stats['current_hour']['in']),
#             float(stats['current_hour']['out']),
#             float(stats['peak']['in']),
#             float(stats['peak']['out']),
#             len(anomalies),
#             anomaly_counts['down'],
#             anomaly_counts['degraded'],
#             anomaly_counts['pattern'],
#             anomaly_counts['model'],
#             status,
#             json.dumps(anomalies)
#         )
        
#         DatabaseManager.execute_query(query, values, fetch=False)
#         return True
        
#     except Exception as e:
#         logging.error(f"Error saving monitoring data: {str(e)}")
#         return False

# @app_anomalies_bp.route('/api/nodes')
# def get_nodes():
#     try:
#         query = """
#         SELECT DISTINCT node1 as node
#         FROM weathermap
#         WHERE node1 IS NOT NULL
#         ORDER BY node1
#         """
#         results = DatabaseManager.execute_query(query)
#         return jsonify([row['node'] for row in results])
#     except Exception as e:
#         logging.error(f"Error fetching nodes: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app_anomalies_bp.route('/api/related-nodes')
# def get_related_nodes():
#     node1 = request.args.get('node1')
#     if not node1:
#         return jsonify([])
    
#     try:
#         query = """
#         SELECT DISTINCT node2 as node
#         FROM weathermap
#         WHERE node1 = %s AND node2 IS NOT NULL
#         ORDER BY node2
#         """
#         results = DatabaseManager.execute_query(query, (node1,))
#         return jsonify([row['node'] for row in results])
#     except Exception as e:
#         logging.error(f"Error fetching related nodes: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app_anomalies_bp.route('/api/related-links')
# def get_related_links():
#     node1 = request.args.get('node1')
#     node2 = request.args.get('node2')
#     if not node1 or not node2:
#         return jsonify([])
    
#     try:
#         query = """
#         SELECT DISTINCT link_name
#         FROM weathermap
#         WHERE node1 = %s AND node2 = %s AND link_name IS NOT NULL
#         ORDER BY link_name
#         """
#         results = DatabaseManager.execute_query(query, (node1, node2))
#         return jsonify([row['link_name'] for row in results])
#     except Exception as e:
#         logging.error(f"Error fetching related links: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app_anomalies_bp.route('/api/rrd-links')
# def get_rrd_links():
#     node1 = request.args.get('node1')
#     node2 = request.args.get('node2')
#     if not node1 or not node2:
#         return jsonify([])
    
#     try:
#         query = """
#         SELECT DISTINCT target
#         FROM weathermap
#         WHERE node1 = %s AND node2 = %s AND target IS NOT NULL
#         """
#         results = DatabaseManager.execute_query(query, (node1, node2))
#         if not results:
#             return jsonify([])
            
#         rrd_links = []
#         for row in results:
#             try:
#                 rrd_file = row['target'].split('/')[-1].replace('.rrd', '')
#                 if rrd_file:
#                     rrd_links.append(rrd_file)
#             except:
#                 continue
                
#         return jsonify(rrd_links)
#     except Exception as e:
#         logging.error(f"Error fetching RRD links: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# # Initialize everything when the module loads
# DatabaseManager.initialize_tables()
