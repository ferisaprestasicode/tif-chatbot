import psycopg2
import os
from datetime import datetime, timedelta
import re
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from concurrent.futures import ThreadPoolExecutor
import io
from itertools import islice
import multiprocessing
import logging
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from threading import Lock

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('network_logs_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_PARAMS = {
    "host": "36.67.62.245",
    "port": "8082",
    "user": "postgres",
    "password": "uhuy123"
}

# SQL Statements
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS network_logs (
    id BIGSERIAL,
    _time TIMESTAMP WITH TIME ZONE NOT NULL,
    host INET,
    hostname VARCHAR(50),
    event_desc3 VARCHAR(100),
    event_desc_last TEXT,
    _raw TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, _time)
) PARTITION BY RANGE (_time);
"""

CREATE_PARTITION_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = 'network_logs_{}'
    ) THEN
        CREATE TABLE network_logs_{} 
        PARTITION OF network_logs
        FOR VALUES FROM ('{}') TO ('{}');
    END IF;
END
$$;
"""

CREATE_INDEX_IF_NOT_EXISTS = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = '{}'
    ) THEN
        {};
    END IF;
END
$$;
"""

CREATE_INDICES_SQL = [
    "CREATE INDEX idx_logs_time_{} ON network_logs_{}(_time)",
    "CREATE INDEX idx_logs_host_{} ON network_logs_{} USING gist (host inet_ops)",
    "CREATE INDEX idx_logs_hostname_{} ON network_logs_{}(hostname)",
    "CREATE INDEX idx_logs_event_desc3_{} ON network_logs_{}(event_desc3)"
]

# Precompiled patterns
TIMESTAMP_PATTERN = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{2}:\d{2})')
IP_PATTERN = re.compile(r'(\d+\.\d+\.\d+\.\d+)')
HOSTNAME_PATTERN = re.compile(r'P-D\d+-\w+')
EVENT_PATTERN = re.compile(r'(%[\w-]+-FM-\d+-[\w_]+)\s*:\s*(.+?)$')
KEYWORDS = {'UPDOWN', 'INTERFACE', 'CRITICAL', 'WARNING'}

class DatabaseManager:
    def __init__(self):
        self.existing_partitions = set()
        self.partition_lock = Lock()

    def get_connection(self, dbname=None):
        """Create a new database connection"""
        params = DB_PARAMS.copy()
        if dbname:
            params['dbname'] = dbname
        return psycopg2.connect(**params)

    # def drop_database(self):
    #     """Drop the existing database if it exists"""
    #     try:
    #         conn = self.get_connection()
    #         conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    #         cur = conn.cursor()
            
    #         cur.execute("""
    #             SELECT pg_terminate_backend(pg_stat_activity.pid)
    #             FROM pg_stat_activity
    #             WHERE pg_stat_activity.datname = 'sisai'
    #             AND pid <> pg_backend_pid();
    #         """)
            
    #         cur.execute("DROP DATABASE IF EXISTS sisai")
    #         logger.info("Database dropped successfully")
    #     except Exception as e:
    #         logger.error(f"Error dropping database: {e}")
    #         raise
    #     finally:
    #         if 'cur' in locals():
    #             cur.close()
    #         if 'conn' in locals():
    #             conn.close()

    def create_database(self):
        """Create the database"""
        try:
            conn = self.get_connection()
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cur = conn.cursor()
            cur.execute("CREATE DATABASE sisai")
            logger.info("Database created successfully")
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            raise
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()

    def ensure_partition_exists(self, timestamp):
        """Ensure partition exists for given timestamp with thread safety"""
        try:
            date = parse(timestamp) if isinstance(timestamp, str) else timestamp
            partition_key = date.strftime("%Y%m")
            
            with self.partition_lock:
                if partition_key in self.existing_partitions:
                    return
                
                start_date = date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                end_date = (start_date + relativedelta(months=1))
                
                conn = self.get_connection('sisai')
                cur = conn.cursor()
                
                # Create partition using DO block for idempotency
                partition_sql = CREATE_PARTITION_SQL.format(
                    partition_key, partition_key,
                    start_date.isoformat(),
                    end_date.isoformat()
                )
                cur.execute(partition_sql)
                
                # Create indices using DO block for idempotency
                for index_sql in CREATE_INDICES_SQL:
                    index_name = f"idx_logs_{partition_key}"
                    full_sql = CREATE_INDEX_IF_NOT_EXISTS.format(
                        index_name,
                        index_sql.format(partition_key, partition_key)
                    )
                    cur.execute(full_sql)
                
                conn.commit()
                self.existing_partitions.add(partition_key)
                logger.info(f"Ensured partition exists for {partition_key}")
                
        except Exception as e:
            logger.error(f"Error ensuring partition exists: {e}")
            # Don't raise the exception, just log it
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()

    def setup_database(self):
        """Set up the database schema"""
        try:
            conn = self.get_connection('sisai')
            cur = conn.cursor()
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()
            logger.info("Database schema setup completed")
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            raise
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()

class LogProcessor:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.batch_lock = Lock()

    def parse_log_line(self, line):
        """Parse a log line and return structured data"""
        try:
            if not any(keyword in line.upper() for keyword in KEYWORDS):
                return None
                
            timestamp_match = TIMESTAMP_PATTERN.search(line)
            if not timestamp_match:
                return None
            
            return {
                '_time': timestamp_match.group(1),
                'host': (IP_PATTERN.search(line) or (None,))[0],
                'hostname': (HOSTNAME_PATTERN.search(line) or (None,))[0],
                'event_desc3': (EVENT_PATTERN.search(line) or (None, None))[1],
                'event_desc_last': (EVENT_PATTERN.search(line) or (None, None))[2],
                '_raw': line.strip()
            }
        except Exception:
            return None

    def process_chunk(self, chunk):
        """Process a chunk of log lines"""
        results = []
        for line in chunk:
            log_data = self.parse_log_line(line)
            if log_data:
                # Ensure partition exists for this timestamp
                self.db_manager.ensure_partition_exists(log_data['_time'])
                results.append((
                    log_data['_time'],
                    log_data['host'],
                    log_data['hostname'],
                    log_data['event_desc3'],
                    log_data['event_desc_last'],
                    log_data['_raw']
                ))
        return results

    def bulk_insert(self, conn, data):
        """Perform bulk insert using COPY command with thread safety"""
        if not data:
            return
            
        with self.batch_lock:
            try:
                sio = io.StringIO()
                for record in data:
                    line = '\t'.join(str(field) if field is not None else '\\N' for field in record)
                    sio.write(line + '\n')
                sio.seek(0)
                
                with conn.cursor() as cur:
                    cur.copy_from(
                        sio,
                        'network_logs',
                        columns=('_time', 'host', 'hostname', 'event_desc3', 'event_desc_last', '_raw')
                    )
                conn.commit()
            except Exception as e:
                logger.error(f"Error in bulk insert: {e}")
                conn.rollback()

    def process_file(self, filepath, num_workers=None):
        """Process a file using parallel processing"""
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        
        chunk_size = 10000
        total_processed = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                conn = self.db_manager.get_connection('sisai')
                
                while True:
                    chunk = list(islice(file, chunk_size))
                    if not chunk:
                        break
                    
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        sub_chunks = [chunk[i::num_workers] for i in range(num_workers)]
                        results = list(executor.map(self.process_chunk, sub_chunks))
                    
                    flat_results = [item for sublist in results for item in sublist]
                    if flat_results:
                        self.bulk_insert(conn, flat_results)
                        total_processed += len(flat_results)
                        logger.info(f"Processed {total_processed} records...")
                
                logger.info(f"Completed processing {total_processed} total records from {filepath}")
                
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {e}", exc_info=True)
        finally:
            if 'conn' in locals():
                conn.close()

def main():
    try:
        logger.info("Starting log processing...")
        
        # Initialize database manager and processor
        db_manager = DatabaseManager()
        processor = LogProcessor(db_manager)
        
        # Setup database
        logger.info("Setting up database...")
        # db_manager.drop_database()
        db_manager.create_database()
        db_manager.setup_database()
        
        # Process log files
        log_dir = r'data\raw\telkomdb-output-01july2024-24sept2024'
        for filename in os.listdir(log_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(log_dir, filename)
                logger.info(f"Processing file: {filename}")
                processor.process_file(filepath)
        
        logger.info("Log processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()