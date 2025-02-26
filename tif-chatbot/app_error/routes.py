from flask import Blueprint, Response, render_template, redirect, url_for, request, session, jsonify, flash, current_app
import re
from datetime import datetime, timedelta
import glob
import os
import pandas as pd
from collections import defaultdict, Counter
from itertools import groupby
from operator import itemgetter
import json
import hashlib
import time
import traceback
import threading
import pickle
import psycopg2
import psycopg2.extras
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProgressTracker:
    def __init__(self):
        self._progress = 0
        self._status = "Initializing..."
        self._detail = ""
        self._lock = threading.Lock()
    
    def update(self, progress, status, detail=""):
        with self._lock:
            self._progress = progress
            self._status = status
            self._detail = detail
    
    def get_state(self):
        with self._lock:
            return {
                "progress": self._progress,
                "status": self._status,
                "detail": self._detail
            }

class NetworkAnalyzer:
    def __init__(self):
        self.logs = []
        self.device_errors = defaultdict(list)
        self.error_types = defaultdict(int)
        self.monthly_stats = defaultdict(self._default_device_dict)
        self.ranked_issues = []
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
        
        # Database configuration
        self.db_config = {
            'host': '36.67.62.245',
            'port': '8082',
            'database': 'sisai',
            'user': 'postgres',
            'password': 'uhuy123'
        }
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _default_device_dict(self):
        return defaultdict(int)

    def get_db_connection(self):
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    def get_table_info(self):
        """Get information about partitioned tables"""
        conn = self.get_db_connection()
        cur = conn.cursor()
        try:
            # Check for partitions
            cur.execute("""
                SELECT parent.relname as table_name,
                    child.relname as partition_name,
                    pg_get_expr(child.relpartbound, child.oid) as partition_expression
                FROM pg_inherits
                JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
                JOIN pg_class child ON pg_inherits.inhrelid = child.oid
                WHERE parent.relname = 'network_logs'
            """)
            partitions = cur.fetchall()
            for partition in partitions:
                logger.info(f"Found partition: {partition}")
                
        except Exception as e:
            logger.error(f"Error getting partition info: {e}")
        finally:
            cur.close()
            conn.close()

    def parse_db_log(self, log_row):
        try:
            # Convert database row to our standard log format
            timestamp = log_row['time']
            hostname = log_row['hostname']
            raw_log = log_row['raw']
            event_desc = log_row['event_desc3'] or log_row['event_desc_last']

            # Extract severity and error type from event description
            severity = 'CRITICAL' if 'CRITICAL' in event_desc else (
                'WARNING' if 'WARNING' in event_desc else 'INFO'
            )
            
            error_match = re.search(r'%([^:]+)', raw_log)
            error_type = error_match.group(1) if error_match else 'OTHER'

            interface_match = re.search(r'Interface ([^,\s]+)', raw_log)
            status_match = re.search(r'(down|up)', raw_log.lower())

            return {
                'timestamp': timestamp,
                'device': hostname,
                'error_type': error_type,
                'severity': severity,
                'interface': interface_match.group(1) if interface_match else None,
                'status': status_match.group(1) if status_match else None,
                'details': raw_log.strip()
            }

        except Exception as e:
            logger.error(f"Error parsing log row: {e}")
            return None

    def get_cache_filename(self, prefix=''):
        hasher = hashlib.md5()
        # Use database config in hash to invalidate cache if DB changes
        hasher.update(json.dumps(self.db_config).encode())
        return os.path.join(self.cache_dir, f"{prefix}processed_logs_{hasher.hexdigest()}.pkl")

    def load_from_cache(self, cache_file):
        try:
            logger.info(f"Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                
            self.logs = cached_data['logs']
            self.device_errors = defaultdict(list, cached_data['device_errors'])
            self.error_types = defaultdict(int, cached_data['error_types'])
            self.monthly_stats = defaultdict(self._default_device_dict)
            for k, v in cached_data['monthly_stats'].items():
                self.monthly_stats[k] = defaultdict(int, v)
            self.ranked_issues = cached_data['ranked_issues']
            
            logger.info(f"Successfully loaded {len(self.logs):,} events from cache")
            return True
            
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            return False

    def save_to_cache(self, cache_file):
        try:
            cached_data = {
                'logs': self.logs,
                'device_errors': dict(self.device_errors),
                'error_types': dict(self.error_types),
                'monthly_stats': {k: dict(v) for k, v in self.monthly_stats.items()},
                'ranked_issues': self.ranked_issues
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
                
            logger.info(f"Successfully saved {len(self.logs):,} events to cache")
            return True
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
            return False

    def process_data(self, progress_tracker=None):
        try:
            cache_file = self.get_cache_filename()
            
            if os.path.exists(cache_file):
                logger.info("Found existing cache file")
                if progress_tracker:
                    progress_tracker.update(10, "Loading from cache...")
                if self.load_from_cache(cache_file):
                    if progress_tracker:
                        progress_tracker.update(30, "Cache loaded successfully")
                    return True
            
            if progress_tracker:
                progress_tracker.update(5, "Connecting to database...")

            logger.info("Fetching logs from database...")
            conn = self.get_db_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # Get total count first
            cur.execute("SELECT COUNT(*) FROM network_logs")
            total_logs = cur.fetchone()[0]
            
            if total_logs == 0:
                logger.warning("No logs found in database")
                if progress_tracker:
                    progress_tracker.update(100, "Error", "No logs found in database")
                return False

            # Fetch logs in batches
            batch_size = 1000
            processed = 0
            
            # In the NetworkAnalyzer class, update the query:
            cur.execute("""
                SELECT 
                    id,
                    _time as time,
                    host,
                    hostname,
                    event_desc3,
                    event_desc_last,
                    _raw as raw,
                    created_at
                FROM network_logs
                ORDER BY _time DESC
            """)
            
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                
                for row in rows:
                    event = self.parse_db_log(row)
                    if event:
                        self.logs.append(event)
                        self.device_errors[event['device']].append(event)
                        self.error_types[event['error_type']] += 1
                        month_key = event['timestamp'].strftime('%Y-%m')
                        self.monthly_stats[month_key][event['device']] += 1
                
                processed += len(rows)
                if progress_tracker:
                    progress_percent = min(90, 5 + (85 * processed / total_logs))
                    progress_tracker.update(
                        progress_percent,
                        "Processing logs...",
                        f"Processed {processed:,} of {total_logs:,} logs"
                    )
            
            cur.close()
            conn.close()
            
            logger.info(f"Processed total of {processed:,} logs")
            
            if progress_tracker:
                progress_tracker.update(95, "Calculating rankings...")
            self.calculate_ranked_issues()
            
            if progress_tracker:
                progress_tracker.update(98, "Saving to cache...")
            self.save_to_cache(cache_file)
            
            if progress_tracker:
                progress_tracker.update(100, "Processing complete!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing logs: {e}")
            if progress_tracker:
                progress_tracker.update(100, "Error", str(e))
            return False

    def calculate_ranked_issues(self):
        try:
            error_counts = Counter()
            device_error_counts = defaultdict(lambda: defaultdict(int))
            
            for device, errors in self.device_errors.items():
                for error in errors:
                    error_type = error['error_type']
                    error_counts[error_type] += 1
                    device_error_counts[error_type][device] += 1
            
            self.ranked_issues = []
            for error_type, count in error_counts.most_common():
                most_affected = max(
                    device_error_counts[error_type].items(),
                    key=lambda x: x[1]
                )
                
                self.ranked_issues.append({
                    'error_type': error_type,
                    'count': count,
                    'most_affected_device': most_affected[0],
                    'device_count': most_affected[1]
                })
                
        except Exception as e:
            logger.error(f"Error calculating rankings: {e}")

    def get_top_error_devices(self, month=None, limit=10):
        device_counts = defaultdict(int)
        
        for device, errors in self.device_errors.items():
            if month:
                count = len([e for e in errors 
                           if e['timestamp'].strftime('%Y-%m') == month])
            else:
                count = len(errors)
            
            if count > 0:
                device_counts[device] = count
        
        return dict(sorted(
            device_counts.items(), 
            key=itemgetter(1), 
            reverse=True
        )[:limit])

    def get_error_trends(self, month=None, device=None):
        filtered_errors = []
        
        for dev, errors in self.device_errors.items():
            if device and dev != device:
                continue
            filtered_errors.extend(errors)
        
        if month:
            filtered_errors = [e for e in filtered_errors 
                             if e['timestamp'].strftime('%Y-%m') == month]
        
        filtered_errors.sort(key=lambda x: x['timestamp'])
        daily_errors = defaultdict(lambda: defaultdict(int))
        
        for error in filtered_errors:
            day = error['timestamp'].strftime('%Y-%m-%d')
            daily_errors[day][error['error_type']] += 1
        
        return [{'date': day, **counts} 
                for day, counts in sorted(daily_errors.items())]

    def get_available_filters(self):
        return {
            'months': sorted(list(set(
                log['timestamp'].strftime('%Y-%m') 
                for log in self.logs
            ))),
            'devices': sorted(list(self.device_errors.keys())),
            'error_types': sorted(list(set(log['error_type'] for log in self.logs)))
        }

class DataCache:
    def __init__(self):
        self.filters = None
        self.ranked_issues = None
        self.device_stats = defaultdict(dict)
        self.trends_data = defaultdict(dict)
        self.logs = None
        self.total_logs = 0
        self.cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def get_cache_filename(self):
        hasher = hashlib.md5()
        hasher.update(str(time.time()).encode())  # Use timestamp for cache invalidation
        return os.path.join(self.cache_dir, f"dashboard_cache_{hasher.hexdigest()}.pkl")
    
    def load_from_cache(self, cache_file):
        try:
            logger.info(f"Loading dashboard cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            self.filters = cached_data['filters']
            self.ranked_issues = cached_data['ranked_issues']
            self.device_stats = cached_data['device_stats']
            self.trends_data = cached_data['trends_data']
            self.logs = cached_data['logs']
            self.total_logs = cached_data['total_logs']
            
            logger.info("Dashboard cache loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading dashboard cache: {e}")
            return False
    
    def save_to_cache(self, cache_file):
        try:
            cached_data = {
                'filters': self.filters,
                'ranked_issues': self.ranked_issues,
                'device_stats': self.device_stats,
                'trends_data': self.trends_data,
                'logs': self.logs,
                'total_logs': self.total_logs
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            
            logger.info("Dashboard cache saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving dashboard cache: {e}")
            return False
        
    def initialize(self, analyzer, progress_tracker):
        cache_file = self.get_cache_filename()
        
        if os.path.exists(cache_file):
            progress_tracker.update(45, "Loading dashboard cache...")
            if self.load_from_cache(cache_file):
                progress_tracker.update(100, "Dashboard ready!")
                return
        
        logger.info("Initializing data cache...")
        start_time = time.time()
        
        progress_tracker.update(50, "Caching filters...")
        self.filters = analyzer.get_available_filters()
        logger.info(f"Found {len(self.filters['months'])} months, {len(self.filters['devices'])} devices")
        
        progress_tracker.update(60, "Caching ranked issues...")
        self.ranked_issues = analyzer.ranked_issues
        logger.info(f"Processed {len(self.ranked_issues)} issue types")
        
        progress_tracker.update(70, "Caching device statistics...")
        months = self.filters['months'] + ['']
        for i, month in enumerate(months, 1):
            progress_tracker.update(
                70 + (10 * i / len(months)),
                "Caching device statistics...",
                f"Processing month {i}/{len(months)}"
            )
            self.device_stats[month] = analyzer.get_top_error_devices(month)
        
        progress_tracker.update(80, "Caching trend data...")
        top_devices = list(analyzer.get_top_error_devices(limit=20).keys()) + ['']
        months = self.filters['months'][-3:] + ['']
        total_combinations = len(months) * len(top_devices)
        current = 0
        
        for month in months:
            for device in top_devices:
                current += 1
                progress_tracker.update(
                    80 + (15 * current / total_combinations),
                    "Caching trend data...",
                    f"Processing combination {current}/{total_combinations}"
                )
                cache_key = f"{month}_{device}"
                self.trends_data[cache_key] = analyzer.get_error_trends(month, device)
        
        progress_tracker.update(95, "Caching logs...")
        self.logs = analyzer.logs
        self.total_logs = len(self.logs)
        
        duration = time.time() - start_time
        logger.info(f"Data cache initialization complete in {duration:.2f} seconds")
        
        progress_tracker.update(98, "Saving dashboard cache...")
        self.save_to_cache(cache_file)
        
        progress_tracker.update(100, "Dashboard ready!")

class DataManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.analyzer = NetworkAnalyzer()
                cls._instance.cache = DataCache()
                cls._instance.initialized = False
                cls._instance.progress_tracker = ProgressTracker()
                cls._instance.initialization_thread = None
            return cls._instance
    
    def initialize_async(self):
        """Start asynchronous initialization if not already initialized"""
        with self._lock:
            if not self.initialized and not self.initialization_thread:
                self.progress_tracker.update(0, "Starting initialization...")
                self.initialization_thread = threading.Thread(
                    target=self._initialize_process
                )
                self.initialization_thread.daemon = True
                self.initialization_thread.start()
    
    def _initialize_process(self):
        """Internal method to handle the initialization process"""
        try:
            logger.info("Starting data manager initialization")
            if self.analyzer.process_data(self.progress_tracker):
                self.cache.initialize(self.analyzer, self.progress_tracker)
                self.initialized = True
                logger.info("Data manager initialization complete!")
            else:
                self.progress_tracker.update(100, "Error", "Failed to process data")
                logger.error("Error initializing data manager!")
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            self.progress_tracker.update(100, "Error", str(e))

    def wait_for_initialization(self, timeout=300):
        """Wait for initialization to complete with timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.initialized or self.progress_tracker.get_state()['progress'] >= 100:
                return self.initialized
            time.sleep(0.5)
        return False

# Initialize data manager as a global instance
data_manager = DataManager()


app_error_bp = Blueprint('app_error', __name__, 
                        template_folder='templates',
                        static_folder='static')

@app_error_bp.before_app_request
def initialize_data():
    """Initialize data manager before first request"""
    data_manager.initialize_async()

@app_error_bp.route('/')
def index():
    """Handle root URL by rendering the main template directly"""
    return render_template('app_error.html')

@app_error_bp.route('/api/processing_progress')
def processing_progress():
    """Stream processing progress updates"""
    def generate():
        last_state = None
        timeout = time.time() + 300  # 5 minute timeout
        
        while time.time() < timeout:
            current_state = data_manager.progress_tracker.get_state()
            
            # Only send updates when state changes
            if current_state != last_state:
                data = json.dumps(current_state)
                yield f"data: {data}\n\n"
                last_state = current_state
            
            if current_state['progress'] >= 100:
                break
            
            time.sleep(0.5)
        
        if time.time() >= timeout:
            error_state = {
                "progress": 100,
                "status": "Error",
                "detail": "Initialization timed out"
            }
            yield f"data: {json.dumps(error_state)}\n\n"
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    )

@app_error_bp.route('/api/filters')
def get_filters():
    """Get available filter options"""
    if not data_manager.initialized:
        return jsonify({'error': 'Data still initializing'}), 503
    return jsonify(data_manager.cache.filters)

@app_error_bp.route('/api/top_devices')
def get_top_devices():
    """Get top devices by error count"""
    if not data_manager.initialized:
        return jsonify({'error': 'Data still initializing'}), 503
    month = request.args.get('month', '')
    return jsonify(data_manager.cache.device_stats[month])

@app_error_bp.route('/api/ranked_issues')
def get_ranked_issues():
    """Get ranked list of issues"""
    if not data_manager.initialized:
        return jsonify({'error': 'Data still initializing'}), 503
    return jsonify(data_manager.cache.ranked_issues)

@app_error_bp.route('/api/trends')
def get_trends():
    """Get error trends data"""
    if not data_manager.initialized:
        return jsonify({'error': 'Data still initializing'}), 503
    month = request.args.get('month', '')
    device = request.args.get('device', '')
    cache_key = f"{month}_{device}"
    
    if cache_key not in data_manager.cache.trends_data:
        return jsonify(data_manager.analyzer.get_error_trends(month, device))
    
    return jsonify(data_manager.cache.trends_data[cache_key])

@app_error_bp.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app_error_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

# Add this to the top of your routes.py file
from flask import current_app

class LogRemarks:
    def __init__(self):
        self.db_config = {
            'host': '36.67.62.245',
            'port': '8082',
            'database': 'sisai',
            'user': 'postgres',
            'password': 'uhuy123'
        }

    def create_table(self):
        """Create the log_remarks table if it doesn't exist"""
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Create the table with proper schema
            cur.execute("""
                CREATE TABLE IF NOT EXISTS log_remarks (
                    id SERIAL PRIMARY KEY,
                    log_time TIMESTAMP NOT NULL,
                    device VARCHAR(255) NOT NULL,
                    error_type VARCHAR(255) NOT NULL,
                    severity VARCHAR(50) NOT NULL,
                    message TEXT NOT NULL,
                    cause TEXT NOT NULL,
                    solution TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(log_time, device)
                )
            """)
            
            # Create index for faster lookups
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_log_remarks_device_time 
                ON log_remarks(device, log_time)
            """)
            
            conn.commit()
            logger.info("log_remarks table created successfully")
            
        except Exception as e:
            logger.error(f"Error creating log_remarks table: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

# Update the before_app_first_request handler
@app_error_bp.before_app_request
def initialize_data():
    """Initialize data manager and create necessary tables before first request"""
    try:
        # Initialize remarks table
        remarks_handler = LogRemarks()
        remarks_handler.create_table()
        logger.info("Remarks table initialized")
        
        # Initialize data manager
        data_manager.initialize_async()
        logger.info("Data manager initialization started")
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise

@app_error_bp.route('/api/remarks', methods=['POST', 'GET'])
def handle_remarks():
    """Handle remarks creation and retrieval"""
    try:
        if request.method == 'POST':
            data = request.json
            
            # Validate required fields
            required_fields = ['time', 'device', 'type', 'severity', 'message', 'cause', 'solution']
            if not all(field in data for field in required_fields):
                return jsonify({
                    'success': False,
                    'error': 'Missing required fields'
                }), 400

            # Convert timestamp string to datetime if needed
            if isinstance(data['time'], str):
                try:
                    timestamp = datetime.fromisoformat(data['time'].replace('Z', '+00:00'))
                except ValueError:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid timestamp format'
                    }), 400
            else:
                timestamp = data['time']

            conn = None
            try:
                conn = psycopg2.connect(**data_manager.analyzer.db_config)
                cur = conn.cursor()
                
                # Insert the remark
                cur.execute("""
                    INSERT INTO log_remarks 
                    (log_time, device, error_type, severity, message, cause, solution)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    timestamp,
                    data['device'],
                    data['type'],
                    data['severity'],
                    data['message'],
                    data['cause'],
                    data['solution']
                ))
                
                remark_id = cur.fetchone()[0]
                conn.commit()
                
                return jsonify({
                    'success': True,
                    'id': remark_id
                })

            except psycopg2.Error as e:
                if conn:
                    conn.rollback()
                logger.error(f"Database error while saving remark: {e}\n{traceback.format_exc()}")
                return jsonify({
                    'success': False,
                    'error': 'Database error occurred'
                }), 500
            finally:
                if cur:
                    cur.close()
                if conn:
                    conn.close()

        elif request.method == 'GET':
            # Handle GET requests to fetch remarks for a specific log
            device = request.args.get('device')
            time = request.args.get('time')
            
            if not device or not time:
                return jsonify({
                    'success': False,
                    'error': 'Missing required parameters'
                }), 400

            try:
                timestamp = datetime.fromisoformat(time.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid timestamp format'
                }), 400

            conn = None
            try:
                conn = psycopg2.connect(**data_manager.analyzer.db_config)
                cur = conn.cursor(cursor_factory=DictCursor)
                
                cur.execute("""
                    SELECT id, log_time, device, error_type, severity, 
                           message, cause, solution, created_at
                    FROM log_remarks
                    WHERE device = %s AND log_time = %s
                    ORDER BY created_at DESC
                """, (device, timestamp))
                
                remarks = []
                for row in cur:
                    remark = dict(row)
                    # Convert datetime objects to ISO format strings
                    remark['log_time'] = remark['log_time'].isoformat()
                    remark['created_at'] = remark['created_at'].isoformat()
                    remarks.append(remark)

                return jsonify(remarks)

            except psycopg2.Error as e:
                logger.error(f"Database error while fetching remarks: {e}\n{traceback.format_exc()}")
                return jsonify({
                    'success': False,
                    'error': 'Database error occurred'
                }), 500
            finally:
                if cur:
                    cur.close()
                if conn:
                    conn.close()

    except Exception as e:
        logger.error(f"Error handling remarks: {e}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred'
        }), 500

# Update the get_detailed_logs route to handle missing table gracefully
@app_error_bp.route('/api/detailed_logs')
def get_detailed_logs():
    """Get detailed log entries with filtering and pagination"""
    if not data_manager.initialized:
        return jsonify({'error': 'Data still initializing'}), 503
    
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        error_type = request.args.get('issue_type', '')
        severity = request.args.get('severity', '')
        search = request.args.get('search', '').lower()
        
        filtered = data_manager.cache.logs
        
        if error_type:
            filtered = [log for log in filtered if log['error_type'] == error_type]
        if severity:
            filtered = [log for log in filtered if log['severity'] == severity]
        if search:
            filtered = [log for log in filtered if search in str(log['details']).lower()]
            
        total = len(filtered)
        filtered.sort(key=lambda x: x['timestamp'], reverse=True)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        page_logs = []
        for log in filtered[start_idx:end_idx]:
            log_copy = log.copy()
            log_copy['timestamp'] = log_copy['timestamp'].isoformat() if hasattr(log_copy['timestamp'], 'isoformat') else str(log_copy['timestamp'])
            page_logs.append(log_copy)

        # Get remarks count with proper error handling
        remarks_count = {}
        try:
            conn = psycopg2.connect(**data_manager.analyzer.db_config)
            cur = conn.cursor()
            
            for log in page_logs:
                cur.execute("""
                    SELECT COUNT(*) FROM log_remarks 
                    WHERE device = %s AND log_time = %s
                """, (log['device'], log['timestamp']))
                count = cur.fetchone()[0]
                remarks_count[f"{log['device']}_{log['timestamp']}"] = count
                
            cur.close()
            conn.close()
        except psycopg2.errors.UndefinedTable:
            # If table doesn't exist, initialize it
            remarks_handler = LogRemarks()
            remarks_handler.create_table()
            remarks_count = {f"{log['device']}_{log['timestamp']}": 0 for log in page_logs}
        except Exception as e:
            logger.error(f"Error getting remarks count: {e}")
            remarks_count = {f"{log['device']}_{log['timestamp']}": 0 for log in page_logs}
        
        return jsonify({
            'logs': page_logs,
            'total': total,
            'remarks_count': remarks_count
        })

    except Exception as e:
        logger.error(f"Error in detailed_logs: {e}\n{traceback.format_exc()}")
        return jsonify({
            'error': str(e),
            'logs': [],
            'total': 0,
            'remarks_count': {}
        }), 500
    
# Add this to your initialization code
def init_app():
    remarks_handler = LogRemarks()
    remarks_handler.create_table()