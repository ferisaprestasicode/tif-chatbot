import os
from dotenv import load_dotenv
import logging
from datetime import datetime

from google.cloud import bigquery
from google.cloud import bigquery
from google.oauth2 import service_account
import vertexai

import psycopg2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'SISAI-TIF_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('SISAI-TIF')

SERVICE_ACCOUNT_FILE =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dark-carport-442807-t8-f50c26239631.json')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = SERVICE_ACCOUNT_FILE

project_id = 'dark-carport-442807-t8'

def get_connection():
    """Establish and return a connection to the PostgreSQL database."""
    return psycopg2.connect(
        host="36.67.62.245",
        port="8082",
        database="sisai",
        user="postgres",
        password="uhuy123"
    )