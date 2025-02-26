import psycopg2
from psycopg2 import Error

def parse_weathermap_file(file_path):
    """Parse the weathermap.txt file and return list of data dictionaries"""
    data_list = []
    current_data = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('LINK '):
                if current_data:  # Save previous entry if exists
                    data_list.append(current_data)
                current_data = {'link_name': line[5:]}  # Remove 'LINK ' prefix
            elif line.startswith('INFOURL '):
                current_data['info_url'] = line[8:]
            elif line.startswith('OVERLIBGRAPH '):
                current_data['overlib_graph'] = line[12:]
            elif line.startswith('TARGET '):
                current_data['target'] = line[7:]
            elif line.startswith('NODES '):
                nodes = line[6:].split()
                current_data['node1'] = nodes[0]
                current_data['node2'] = nodes[1]
            elif line.startswith('VIA '):
                via_values = line[4:].split()
                current_data['via_in'] = via_values[0]
                current_data['via_out'] = via_values[1]
            elif line.startswith('BANDWIDTH '):
                current_data['bandwidth'] = line[10:]
    
    if current_data:  # Add the last entry
        data_list.append(current_data)
    
    return data_list

def create_weathermap_table(conn):
    """Create weathermap table if it doesn't exist"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS weathermap (
        id SERIAL PRIMARY KEY,
        link_name VARCHAR(255),
        info_url TEXT,
        overlib_graph TEXT,
        target TEXT,
        node1 VARCHAR(255),
        node2 VARCHAR(255),
        via_in VARCHAR(50),
        via_out VARCHAR(50),
        bandwidth VARCHAR(50)
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

def insert_weathermap_data(conn, data_list):
    """Insert data into weathermap table"""
    insert_query = """
    INSERT INTO weathermap (link_name, info_url, overlib_graph, target, node1, node2, via_in, via_out, bandwidth)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        cursor = conn.cursor()
        for data in data_list:
            cursor.execute(insert_query, (
                data.get('link_name'),
                data.get('info_url'),
                data.get('overlib_graph'),
                data.get('target'),
                data.get('node1'),
                data.get('node2'),
                data.get('via_in'),
                data.get('via_out'),
                data.get('bandwidth')
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
        # Parse weathermap file
        data_list = parse_weathermap_file('data/raw/weathermap.txt')
        print(f"Parsed {len(data_list)} entries from weathermap.txt")
        
        # Connect to database
        conn = psycopg2.connect(**db_params)
        print("Connected to database successfully")
        
        # Create table and insert data
        create_weathermap_table(conn)
        insert_weathermap_data(conn, data_list)
        
    except Error as e:
        print(f"Database error: {e}")
    except FileNotFoundError:
        print("Weathermap file not found")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
            print("Database connection closed")

if __name__ == "__main__":
    main()