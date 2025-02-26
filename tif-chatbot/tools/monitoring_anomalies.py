from pydantic import BaseModel, Field
from typing import Type
import pandas as pd
import psycopg2
from datetime import datetime, timedelta

class AnomaliesInput(BaseModel):
    date: str = Field(
        description="Date in format YYYY-MM-DD or YYYY-MM",
        pattern=r'\d{4}-\d{2}(-\d{2})?'
    )

class DBConnection:
    def __init__(self):
        self.db_params = {
            "host": "36.67.62.245",
            "port": "8082",
            "database": "sisai",
            "user": "postgres",
            "password": "uhuy123",
            "connect_timeout": 10,
            "application_name": 'network_monitoring'
        }

    def _get_connection(self):
        """Establish database connection"""
        try:
            return psycopg2.connect(**self.db_params)
        except psycopg2.Error as e:
            raise Exception(f"Failed to connect to database: {e}")

    def _execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            with self._get_connection() as conn:
                if params:
                    return pd.read_sql_query(query, conn, params=params)
                return pd.read_sql_query(query, conn)
        except Exception as e:
            raise Exception(f"Query execution failed: {e}")
class AnomaliesMonitoring(DBConnection):
    name = "anomalies"
    description = "Tool untuk monitoring dan analisis anomali pada jaringan secara harian atau bulanan berdasarkan link jaringans"
    args_schema = AnomaliesInput

    def run(self, date: str) -> str:
        """Analyze anomalies for a specific date or month"""
        try:
            is_full_date = len(date.split('-')) == 3
            
            # Check data existence and available dates
            if is_full_date:
                available_dates_query = """
                SELECT DISTINCT monitoring_date::date as dates
                FROM public.link_monitoring
                WHERE DATE(monitoring_date) >= DATE(%s) - INTERVAL '7 days'
                  AND DATE(monitoring_date) <= DATE(%s) + INTERVAL '7 days'
                ORDER BY dates DESC;
                """
                dates_df = self._execute_query(available_dates_query, (date, date))
            else:
                available_dates_query = """
                SELECT DISTINCT TO_CHAR(monitoring_date, 'YYYY-MM') as month,
                       MIN(monitoring_date::date) as start_date,
                       MAX(monitoring_date::date) as end_date
                FROM public.link_monitoring
                WHERE TO_CHAR(monitoring_date, 'YYYY-MM') = %s
                GROUP BY TO_CHAR(monitoring_date, 'YYYY-MM');
                """
                dates_df = self._execute_query(available_dates_query, (date,))
            
            if dates_df.empty:
                # Get nearest available dates for reference
                nearby_dates_query = """
                SELECT DISTINCT monitoring_date::date as dates
                FROM public.link_monitoring
                ORDER BY dates DESC
                LIMIT 5;
                """
                nearby_df = self._execute_query(nearby_dates_query)
                nearby_df['dates'] = pd.to_datetime(nearby_df['dates'])
                available_dates = nearby_df['dates'].dt.strftime('%Y-%m-%d').tolist() if not nearby_df.empty else []
                
                period = "tanggal" if is_full_date else "bulan"
                return (f"Tidak ditemukan data untuk {period} {date}. "
                       f"Data tersedia untuk tanggal: {', '.join(available_dates)}")
            
            return self._get_daily_anomalies(date) if is_full_date else self._get_monthly_anomalies(date)
            
        except Exception as e:
            return f"Error saat mengambil data anomali: {str(e)}"

    def _get_daily_anomalies(self, date: str) -> str:
        """Get anomalies for a specific date"""
        query = """
        WITH AnomalyStats AS (
            SELECT 
                hostname,
                COUNT(*) as record_count,
                SUM(COALESCE(anomaly_count, 0)) as total_anomalies,
                SUM(COALESCE(down_events, 0)) as total_down,
                SUM(COALESCE(degraded_events, 0)) as total_degraded,
                SUM(COALESCE(pattern_events, 0)) as total_pattern,
                STRING_AGG(DISTINCT COALESCE(status, 'Unknown'), ', ') as health_statuses,
                AVG(COALESCE(traffic_in, 0)) as avg_traffic_in,
                AVG(COALESCE(traffic_out, 0)) as avg_traffic_out,
                MAX(COALESCE(peak_traffic_in, 0)) as max_traffic_in,
                MAX(COALESCE(peak_traffic_out, 0)) as max_traffic_out
            FROM public.link_monitoring
            WHERE DATE(monitoring_date) = %s
            GROUP BY hostname
            HAVING SUM(COALESCE(anomaly_count, 0)) > 0
                OR SUM(COALESCE(down_events, 0)) > 0
                OR SUM(COALESCE(degraded_events, 0)) > 0
                OR SUM(COALESCE(pattern_events, 0)) > 0
        )
        SELECT *,
               RANK() OVER (ORDER BY total_anomalies DESC, hostname) as device_rank
        FROM AnomalyStats
        ORDER BY total_anomalies DESC, hostname;
        """
        
        df = self._execute_query(query, (date,))
        return self._format_anomalies_report(df, date, "daily")

    def _get_monthly_anomalies(self, month: str) -> str:
        """Get anomalies for a specific month"""
        query = """
        WITH AnomalyStats AS (
            SELECT 
                hostname,
                COUNT(*) as record_count,
                SUM(COALESCE(anomaly_count, 0)) as total_anomalies,
                SUM(COALESCE(down_events, 0)) as total_down,
                SUM(COALESCE(degraded_events, 0)) as total_degraded,
                SUM(COALESCE(pattern_events, 0)) as total_pattern,
                STRING_AGG(DISTINCT COALESCE(status, 'Unknown'), ', ') as health_statuses,
                AVG(COALESCE(traffic_in, 0)) as avg_traffic_in,
                AVG(COALESCE(traffic_out, 0)) as avg_traffic_out,
                MAX(COALESCE(peak_traffic_in, 0)) as max_traffic_in,
                MAX(COALESCE(peak_traffic_out, 0)) as max_traffic_out,
                COUNT(DISTINCT DATE(monitoring_date)) as days_with_anomalies
            FROM public.link_monitoring
            WHERE TO_CHAR(monitoring_date, 'YYYY-MM') = %s
            GROUP BY hostname
            HAVING SUM(COALESCE(anomaly_count, 0)) > 0
                OR SUM(COALESCE(down_events, 0)) > 0
                OR SUM(COALESCE(degraded_events, 0)) > 0
                OR SUM(COALESCE(pattern_events, 0)) > 0
        )
        SELECT *,
               RANK() OVER (ORDER BY total_anomalies DESC, hostname) as device_rank
        FROM AnomalyStats
        ORDER BY total_anomalies DESC, hostname;
        """
        
        df = self._execute_query(query, (month,))
        return self._format_anomalies_report(df, month, "monthly")

    def _format_anomalies_report(self, df: pd.DataFrame, date: str, report_type: str) -> str:
        """Format anomalies report for both daily and monthly views"""
        if df.empty:
            period = "tanggal" if report_type == "daily" else "bulan"
            return f"Tidak ditemukan anomali pada {period} {date}"

        # Calculate totals
        total_anomalies = int(df['total_anomalies'].sum())
        total_devices = len(df)
        period = "tanggal" if report_type == "daily" else "bulan"

        # Start building response
        response = [
            f"Laporan Anomali untuk {period} {date}:",
            f"\nRingkasan:",
            f"• Total Anomali: {total_anomalies:,}",
            f"• Jumlah Device Terdampak: {total_devices}"
        ]

        if report_type == "monthly":
            avg_days = df['days_with_anomalies'].mean()
            response.append(f"• Rata-rata Hari dengan Anomali per Device: {avg_days:.1f} hari")

        response.append("\nDetail per Device:")

        for _, row in df.iterrows():
            anomaly_details = []
            if row['total_down'] > 0:
                anomaly_details.append(f"Down Events: {int(row['total_down']):,}")
            if row['total_degraded'] > 0:
                anomaly_details.append(f"Degraded Events: {int(row['total_degraded']):,}")
            if row['total_pattern'] > 0:
                anomaly_details.append(f"Pattern Events: {int(row['total_pattern']):,}")

            device_details = [
                f"\n{row['device_rank']}. Hostname: {row['hostname']}",
                f"• Total Anomali: {int(row['total_anomalies']):,}",
                f"• Events Breakdown: {', '.join(anomaly_details)}",
                f"• Traffic Statistics:",
                f"  - Average In: {row['avg_traffic_in']:.2f} Gbps (Peak: {row['max_traffic_in']:.2f} Gbps)",
                f"  - Average Out: {row['avg_traffic_out']:.2f} Gbps (Peak: {row['max_traffic_out']:.2f} Gbps)",
                f"• Status Kesehatan: {row['health_statuses']}"
            ]
            
            if report_type == "monthly":
                device_details.append(f"• Hari dengan Anomali: {int(row['days_with_anomalies'])} hari")
            
            response.extend(device_details)

        return "\n".join(response)
    

def main():
    # Create instance of AnomaliesMonitoring
    monitor = AnomaliesMonitoring()
    
    # Test cases for known available dates
    test_cases = [
        # Daily tests for known dates
        "2024-09-24",  # Known available date
        "2024-12-09",  # Known available date
        "2024-12-12",  # Known available date
        
        # Monthly tests
        "2024-09",  # September (has some data)
        "2024-12"   # December (has some data)
    ]
    
    # Run tests
    for date in test_cases:
        print(f"\n{'='*80}")
        print(f"Testing date: {date}")
        print('='*80)
        try:
            result = monitor.run(date)
            print(result)
        except Exception as e:
            print(f"Error testing {date}: {str(e)}")

if __name__ == "__main__":
    main()