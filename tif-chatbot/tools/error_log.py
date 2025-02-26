import psycopg2
import pandas as pd
import os
import re
from pydantic import BaseModel, Field
from typing import Type
from datetime import datetime
from psycopg2.extras import DictCursor

class CountErrorsInput(BaseModel):
    date: str = Field(
        description="Date in format YYYY-MM-DD or YYYY-MM",
        pattern=r'\d{4}-\d{2}(-\d{2})?'
    )

class TopErrorDevicesInput(BaseModel):
    date: str = Field(
        description="Date in format YYYY-MM-DD or YYYY-MM",
        pattern=r'\d{4}-\d{2}(-\d{2})?'
    )

class TopErrorCategoriesInput(BaseModel):
    date: str = Field(
        description="Date in format YYYY-MM-DD or YYYY-MM",
        pattern=r'\d{4}-\d{2}(-\d{2})?'
    )

class DBConnection:
    def __init__(self):
        self.db_params = {
            "host": "36.67.62.245",
            "port": "8082",
            "user": "postgres",
            "password": "uhuy123",
            "dbname": "sisai"
        }

    def _get_connection(self):
        try:
            return psycopg2.connect(**self.db_params)
        except psycopg2.Error as e:
            raise Exception(f"Failed to connect to database: {e}")

    def _execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        try:
            with self._get_connection() as conn:
                if params:
                    return pd.read_sql_query(query, conn, params=params)
                return pd.read_sql_query(query, conn)
        except Exception as e:
            raise Exception(f"Query execution failed: {e}")

class CountErrors(DBConnection):
    name: str = "count_errors"
    description: str = "Tool untuk menghitung jumlah error pada tanggal atau bulan tertentu"
    args_schema: Type[BaseModel] = CountErrorsInput

    def run(self, date: str):
        """Count total errors for a specific date/month"""
        is_full_date = len(date.split('-')) == 3
        date_filter = "DATE(_time) = %s" if is_full_date else "TO_CHAR(_time, 'YYYY-MM') = %s"
        
        query = f"SELECT COUNT(*) as error_count FROM network_logs WHERE {date_filter}"
        df = self._execute_query(query, (date,))
        count = df.iloc[0]['error_count']
        period = "tanggal" if is_full_date else "bulan"
        
        return f"Jumlah error pada {period} {date}: {count:,}"

class TopErrorDevices(DBConnection):
    name: str = "top_error_devices"
    description: str = "Tool untuk melihat daftar device dengan error terbanyak"
    args_schema: Type[BaseModel] = TopErrorDevicesInput

    def run(self, date: str):
        """Find top 10 devices with most errors and their recurrent errors"""
        is_full_date = len(date.split('-')) == 3
        date_filter = "TO_CHAR(_time, 'YYYY-MM') = %s" if not is_full_date else "DATE(_time) = %s"

        device_query = f"""
        SELECT 
            hostname as device,
            COUNT(*) as error_count
        FROM network_logs
        WHERE {date_filter}
        GROUP BY hostname
        ORDER BY error_count DESC
        LIMIT 10
        """

        error_query = f"""
        WITH TopDevices AS (
            SELECT hostname as device
            FROM network_logs
            WHERE {date_filter}
            GROUP BY hostname
            ORDER BY COUNT(*) DESC
            LIMIT 10
        )
        SELECT 
            l.hostname as device,
            l.event_desc3 as error_category,
            COUNT(*) as category_count
        FROM network_logs l
        JOIN TopDevices t ON l.hostname = t.device
        WHERE {date_filter}
        GROUP BY l.hostname, l.event_desc3
        """

        device_df = self._execute_query(device_query, (date,))
        error_df = self._execute_query(error_query, (date, date))
        
        if device_df.empty:
            return f"No errors found for {date}"

        period = "tanggal" if is_full_date else "bulan"
        result = [f"Top Device dengan Error Terbanyak ({period} {date}):", ""]
        
        for _, row in device_df.iterrows():
            result.append(f"Host: {row['device']}")
            result.append(f"Total Error: {row['error_count']:,}")
            
            device_errors = error_df[error_df['device'] == row['device']]\
                .sort_values('category_count', ascending=False)\
                .head(3)
            
            if not device_errors.empty:
                result.append("Recurrent Errors:")
                for _, err_row in device_errors.iterrows():
                    result.append(f"- {err_row['error_category']}: {err_row['category_count']:,} error")
            result.append("-" * 80 + "\n")

        return "\n".join(result)

class TopErrorCategories(DBConnection):
    name: str = "top_error_categories"
    description: str = "Tool untuk melihat kategori error yang paling sering terjadi"
    args_schema: Type[BaseModel] = TopErrorCategoriesInput

    def run(self, date: str):
        """Find top 10 error categories with device breakdown"""
        is_full_date = len(date.split('-')) == 3
        date_filter = "DATE(_time) = %s" if is_full_date else "TO_CHAR(_time, 'YYYY-MM') = %s"
        
        query = f"""
        WITH ErrorSummary AS (
            SELECT 
                event_desc3 as error_category,
                COUNT(*) as error_count,
                COUNT(DISTINCT hostname) as device_count
            FROM network_logs 
            WHERE {date_filter}
            GROUP BY event_desc3
            ORDER BY error_count DESC
            LIMIT 10
        )
        SELECT 
            e.*,
            l.hostname as device,
            COUNT(*) as device_error_count
        FROM ErrorSummary e
        JOIN network_logs l ON e.error_category = l.event_desc3
        WHERE {date_filter}
        GROUP BY e.error_category, l.hostname, e.error_count, e.device_count
        ORDER BY e.error_count DESC, device_error_count DESC
        """
        
        df = self._execute_query(query, (date, date))
        if df.empty:
            return f"No errors found for {date}"

        period = "tanggal" if is_full_date else "bulan"
        result = [f"Top 10 Kategori Error ({period} {date}):", ""]
        
        current_category = None
        for _, row in df.iterrows():
            if current_category != row['error_category']:
                if current_category is not None:
                    result.append("")  # Add blank line between categories
                current_category = row['error_category']
                result.append(f"- {current_category}:")
                result.append(f"  * {row['error_count']:,} error")
                result.append(f"  * Terjadi pada {row['device_count']} device:")
            
            result.append(f"    - {row['device']}: {row['device_error_count']:,} error")

        return "\n".join(result)
class TelkomAnalyzer:
    name = "error"
    description = "Tool untuk menganalisis error pada jaringan Telkom"
    args_schema = CountErrorsInput

    def __init__(self):
        self.db_connection = DBConnection()

    def run(self, date: str) -> dict:
        """Process error analysis requests"""
        try:
            results = self._analyze_errors(date)
            return {
                "results": results,
                "tools": True
            }
        except Exception as e:
            return {
                "error": f"Error analyzing data: {str(e)}",
                "tools": False
            }

    def _analyze_errors(self, date: str) -> str:
        """Analyze errors for a given date/month"""
        is_full_date = len(date.split('-')) == 3
        date_filter = "DATE(_time) = %s" if is_full_date else "TO_CHAR(_time, 'YYYY-MM') = %s"
        
        # Get total error count
        count_query = f"""
        SELECT COUNT(*) as total_errors 
        FROM network_logs 
        WHERE {date_filter}
        """
        total_df = self.db_connection._execute_query(count_query, (date,))
        total_errors = total_df.iloc[0]['total_errors']

        # Get top devices with their errors
        device_query = f"""
        WITH TopDevices AS (
            SELECT 
                hostname,
                COUNT(*) as error_count,
                ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) as rank
            FROM network_logs
            WHERE {date_filter}
            GROUP BY hostname
            ORDER BY error_count DESC
            LIMIT 10
        ),
        DeviceErrors AS (
            SELECT 
                d.hostname,
                d.error_count,
                l.event_desc3,
                COUNT(*) as category_count,
                ROW_NUMBER() OVER (PARTITION BY d.hostname ORDER BY COUNT(*) DESC) as error_rank
            FROM TopDevices d
            JOIN network_logs l ON d.hostname = l.hostname
            WHERE {date_filter}
            GROUP BY d.hostname, d.error_count, l.event_desc3
        )
        SELECT *
        FROM DeviceErrors
        WHERE error_rank <= 3
        ORDER BY error_count DESC, hostname, error_rank
        """
        
        device_df = self.db_connection._execute_query(device_query, (date, date))

        # Get top error categories with device breakdown
        category_query = f"""
        WITH TopCategories AS (
            SELECT 
                event_desc3,
                COUNT(*) as total_count,
                COUNT(DISTINCT hostname) as device_count,
                ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) as rank
            FROM network_logs
            WHERE {date_filter}
            GROUP BY event_desc3
            ORDER BY total_count DESC
            LIMIT 10
        ),
        CategoryDetails AS (
            SELECT 
                c.event_desc3,
                c.total_count,
                c.device_count,
                l.hostname,
                COUNT(*) as device_error_count,
                ROW_NUMBER() OVER (PARTITION BY c.event_desc3 ORDER BY COUNT(*) DESC) as device_rank
            FROM TopCategories c
            JOIN network_logs l ON c.event_desc3 = l.event_desc3
            WHERE {date_filter}
            GROUP BY c.event_desc3, c.total_count, c.device_count, l.hostname
        )
        SELECT *
        FROM CategoryDetails
        WHERE device_rank <= 3
        ORDER BY total_count DESC, device_error_count DESC
        """
        
        category_df = self.db_connection._execute_query(category_query, (date, date))

        # Format the response
        period = "tanggal" if is_full_date else "bulan"
        response = [f"Pada {period} {date}, total jumlah error yang terjadi adalah {total_errors:,} error. "
                   "Berikut adalah beberapa perangkat dengan jumlah error terbanyak dan kategori error yang sering terjadi:"]
        
        # Format device information
        response.append("**Top Device dengan Error Terbanyak:**")
        current_device = None
        for _, row in device_df.iterrows():
            if current_device != row['hostname']:
                current_device = row['hostname']
                response.append(f"{len(response)-1}. **{row['hostname']}**")
                response.append(f"   * Total Error: {row['error_count']:,}")
                response.append("   * Error yang sering terjadi:")
            response.append(f"      * {row['event_desc3']}: {row['category_count']:,} error")

        # Format category information
        response.append("**Top Kategori Error:**")
        current_category = None
        for _, row in category_df.iterrows():
            if current_category != row['event_desc3']:
                current_category = row['event_desc3']
                response.append(f"{len(response)-1}. **{row['event_desc3']}**")
                response.append(f"   * Total: {row['total_count']:,} error")
                response.append("   * Terjadi pada:")
            response.append(f"      * {row['hostname']}: {row['device_error_count']:,} error")

        response.append("\nJika Anda mempunyai pertanyaan lebih lanjut atau memerlukan analisis mendalam, silahkan beri tahu.")
        
        return "\n".join(response)
    
def main():
    """Test the analyzer with all cases"""
    try:
        analyzer = TelkomAnalyzer()
        
        test_cases = [
            "berapa jumlah error tanggal 2024-08-01",
            "berapa jumlah error bulan 2024-08",
            "device mana yang paling banyak error tanggal 2024-08-01",
            "device mana yang paling banyak error bulan 2024-08",
            "jenis error apa yang paling banyak terjadi tanggal 2024-08-01",
            "jenis error apa yang paling banyak terjadi bulan 2024-08"
        ]
        
        print("Telkom Log Analyzer Test Cases")
        print("=" * 80)
        
        for query in test_cases:
            print(f"\nQuery: {query}")
            print("-" * 50)
            result = analyzer.process_query(query)
            print(result)
            print("\n" + "=" * 80)
            
    except Exception as e:
        print(f"Error running test cases: {str(e)}")

if __name__ == "__main__":
    main()