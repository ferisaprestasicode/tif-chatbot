import pandas as pd
import psycopg2
from psycopg2 import Error

def create_table(conn):
    """Create cacti table if it doesn't exist"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS cacti (
        id SERIAL PRIMARY KEY,
        description VARCHAR(255),
        hostname VARCHAR(255),
        field_name VARCHAR(255),
        field_value VARCHAR(255),
        snmp_index INTEGER,
        data_source_path TEXT
    );
    """
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_query)
        conn.commit()
        print("Table created successfully (if it didn't exist)")
    except Error as e:
        print(f"Error creating table: {e}")
        raise e
    finally:
        cursor.close()

def insert_data(conn, df):
    """Insert data from DataFrame into cacti table"""
    insert_query = """
    INSERT INTO cacti (description, hostname, field_name, field_value, snmp_index, data_source_path)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    try:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            cursor.execute(insert_query, (
                row['description'],
                row['hostname'],
                row['field_name'],
                row['field_value'],
                row['snmp_index'],
                row['data_source_path']
            ))
        conn.commit()
        print("Data inserted successfully")
    except Error as e:
        print(f"Error inserting data: {e}")
        raise e
    finally:
        cursor.close()

def main():
    # Database connection parameters
    db_params = {
        "host": "36.67.62.245",
        "port": "8082",
        "database": "sisai",
        "user": "postgres",
        "password": "uhuy123"
    }
    
    try:
        # Read CSV file
        df = pd.read_csv(r'data\raw\cacti.csv')
        print(f"Read {len(df)} rows from CSV")
        
        # Connect to database
        conn = psycopg2.connect(**db_params)
        print("Connected to database successfully")
        
        # Create table and insert data
        create_table(conn)
        insert_data(conn, df)
        
    except Error as e:
        print(f"Database error: {e}")
    except FileNotFoundError:
        print("CSV file not found")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
            print("Database connection closed")

if __name__ == "__main__":
    main()