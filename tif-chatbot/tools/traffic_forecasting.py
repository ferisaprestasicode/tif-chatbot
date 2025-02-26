from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd
import psycopg2
from psycopg2.extras import DictCursor
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom exception for database-related errors"""
    pass

class ForecastingInput(BaseModel):
    date: str = Field(
        description="Tanggal prediksi (format: DD:MM.Y)", 
        pattern=r'\d{2}:\d{2}\.\d{1}'
    )
    link: Optional[str] = Field(
        default=None,
        description="Nama link spesifik (opsional)"
    )

    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        if not re.match(r'\d{2}:\d{2}\.\d{1}', v):
            raise ValueError('Format tanggal harus DD:MM.Y')
        return v

class DBConnection:
    def __init__(self, db_params: Optional[Dict[str, Any]] = None):
        self.db_params = db_params or {
            "host": "36.67.62.245",
            "port": "8082",
            "database": "sisai",
            "user": "postgres",
            "password": "uhuy123",
            "connect_timeout": 10,
            "application_name": 'network_monitoring'
        }

    def _get_connection(self):
        try:
            return psycopg2.connect(**self.db_params)
        except psycopg2.OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            raise DatabaseError(f"Tidak dapat terhubung ke database: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during database connection: {e}")
            raise DatabaseError(f"Error tidak terduga saat menghubungi database: {e}")

    def _execute_query(self, query: str, params: tuple = None) -> list:
        """Execute query and return results as list of dictionaries"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    logger.info(f"Executing query with params: {params}")
                    cur.execute(query, params)
                    columns = [desc[0] for desc in cur.description]
                    results = [dict(zip(columns, row)) for row in cur.fetchall()]
                    return results
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise DatabaseError(f"Query execution failed: {str(e)}")

class TrafficForecasting(DBConnection):
    name = "traffic_forecasting"
    description = "Tool untuk memprediksi traffic jaringan berdasarkan data forecast yang tersedia"
    args_schema = ForecastingInput

    def _format_traffic_value(self, value: str) -> str:
        """Format traffic values with appropriate units"""
        try:
            value = float(value)
            if value >= 1000:
                return f"{value/1000:.2f} Tbps"
            elif value >= 1:
                return f"{value:.2f} Gbps"
            else:
                return f"{value*1000:.2f} Mbps"
        except (ValueError, TypeError):
            return "N/A"

    def _extract_link_name(self, rrd_path: str) -> str:
        """Extract and format link name from RRD path"""
        if not rrd_path:
            return 'Unknown'
            
        if rrd_path == 'overview':
            return 'Overview (All Links)'
        
        try:
            match = re.search(r'_(\d+)$', rrd_path)
            if match:
                return f"Link {match.group(1)}"
            return rrd_path
        except (AttributeError, IndexError):
            return 'Unknown'

    def _build_response(self, row: Dict[str, Any]) -> str:
        """Build formatted response string"""
        try:
            link_name = self._extract_link_name(row.get('rrd_path', ''))
            
            if row.get('rrd_path') == 'overview':
                response = [
                    f"Prediksi traffic untuk {link_name}:",
                    "\nPrediksi Minggu Depan:",
                    f"• Total Traffic In: {self._format_traffic_value(row.get('next_week_traffic_in'))}",
                    f"• Total Traffic Out: {self._format_traffic_value(row.get('next_week_traffic_out'))}",
                    "\nPrediksi Bulan Depan:",
                    f"• Total Traffic In: {self._format_traffic_value(row.get('next_month_traffic_in'))}",
                    f"• Total Traffic Out: {self._format_traffic_value(row.get('next_month_traffic_out'))}"
                ]
            else:
                response = [
                    f"Prediksi traffic untuk {link_name}:",
                    "\nPrediksi Minggu Depan:",
                    f"• Traffic In: {self._format_traffic_value(row.get('next_week_traffic_in'))}",
                    f"• Traffic Out: {self._format_traffic_value(row.get('next_week_traffic_out'))}",
                    "\nPrediksi Bulan Depan:",
                    f"• Traffic In: {self._format_traffic_value(row.get('next_month_traffic_in'))}",
                    f"• Traffic Out: {self._format_traffic_value(row.get('next_month_traffic_out'))}"
                ]
            
            if 'prediction_date' in row:
                response.append(f"\nPrediksi untuk: {row['prediction_date']}")
            if 'created_at' in row and row['created_at']:
                response.append(f"Waktu pembuatan: {row['created_at']}")
            
            return "\n".join(response)
        except Exception as e:
            logger.error(f"Error building response: {e}")
            raise DatabaseError(f"Error memformat respons: {str(e)}")

    def run(self, date: str, link: Optional[str] = None) -> str:
        """Execute traffic forecasting with improved error handling and validation"""
        try:
            input_data = ForecastingInput(date=date, link=link)
            
            query = """
            SELECT 
                id,
                rrd_path,
                prediction_date::text as prediction_date,
                next_week_traffic_in::text as next_week_traffic_in,
                next_week_traffic_out::text as next_week_traffic_out,
                next_month_traffic_in::text as next_month_traffic_in,
                next_month_traffic_out::text as next_month_traffic_out,
                created_at::text as created_at
            FROM public.traffic_forecasts
            WHERE prediction_date::text LIKE %s
            """
            
            params = [f'%{input_data.date}%']
            
            if input_data.link:
                query += " AND rrd_path LIKE %s"
                params.append(f'%{input_data.link}%')
            
            query += " ORDER BY created_at DESC LIMIT 1;"
            
            results = self._execute_query(query, tuple(params))
            
            if not results:
                return (f"Tidak ada prediksi traffic untuk tanggal {input_data.date}" + 
                       (f" dan link {input_data.link}" if input_data.link else ""))
            
            return self._build_response(results[0])

        except ValueError as e:
            logger.error(f"Input validation error: {e}")
            return f"Error validasi input: {str(e)}"
        except DatabaseError as e:
            logger.error(f"Database error: {e}")
            return str(e)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Terjadi kesalahan tidak terduga: {str(e)}"

if __name__ == "__main__":
    forecaster = TrafficForecasting()
    
    def print_test_header(test_name):
        print("\n" + "="*50)
        print(f"TEST: {test_name}")
        print("="*50)
    
    def run_test(date, link=None, test_name=None):
        if test_name:
            print(f"\n{test_name}:")
        print(f"Query parameters - Date: {date}" + (f", Link: {link}" if link else ""))
        print("-" * 30)
        result = forecaster.run(date, link)
        print(result)
        print("-" * 30)
        return result
    
    # Test 1: Known Working Queries
    print_test_header("Known Working Queries")
    test_cases = [
        ("25:23.1", "146008", "Link 146008 Forecast"),
        ("31:12.9", None, "Latest Overall Forecast")
    ]
    for date, link, name in test_cases:
        run_test(date, link, name)
    
    # Test 2: Time Range Queries
    print_test_header("Time Range Queries")
    for time in ["31:12.8", "31:12.9", "31:13.0"]:
        run_test(time, test_name=f"Time: {time}")
    
    # Test 3: Link Range Tests
    print_test_header("Link Range Tests")
    for link_id in ["146389", "146390", "146391"]:
        run_test("31:12.9", link_id, f"Link {link_id}")
    
    # Test 4: Error Cases
    print_test_header("Error Cases")
    error_cases = [
        ("2024-01-01", None, "Invalid date format"),
        ("99:99.9", None, "Non-existent date"),
        ("31:12.9", "999999", "Non-existent link"),
        ("31:12.9", "!@#$%", "Special characters in link")
    ]
    for date, link, case_name in error_cases:
        run_test(date, link, case_name)
    
    # Test 5: Database Connection Error
    print_test_header("Database Connection Error Test")
    try:
        bad_params = {
            "host": "invalid_host",
            "port": "8082",
            "database": "invalid_db",
            "user": "invalid_user",
            "password": "invalid_password",
        }
        bad_forecaster = TrafficForecasting(bad_params)
        print("\nTesting invalid database connection:")
        print(bad_forecaster.run("31:12.9"))
    except Exception as e:
        print(f"Expected error occurred: {str(e)}")
    
    # Test 6: Data Consistency Check
    print_test_header("Data Consistency Check")
    baseline_result = None
    for i in range(3):
        print(f"\nIteration {i+1}:")
        current_result = run_test("31:12.9")
        if i == 0:
            baseline_result = current_result
        elif current_result != baseline_result:
            print("WARNING: Inconsistent results detected!")
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)