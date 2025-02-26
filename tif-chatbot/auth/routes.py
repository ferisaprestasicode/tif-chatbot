from flask import Blueprint, render_template, redirect, url_for, request, session, jsonify, make_response
import hashlib
import datetime
import pyotp  # For OTP generation
import qrcode  # For QR code generation
import io
import base64
import uuid
from datetime import datetime, timedelta
import psycopg2

from config import get_connection


auth_bp = Blueprint('auth', __name__, template_folder='templates')

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    """Initialize the PostgreSQL database and create tables if they don't exist."""
    try:
        connection = get_connection()
        cursor = connection.cursor()

        # Create users table
        create_user_table = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            reset_token TEXT,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            name TEXT,
            department TEXT,
            role TEXT DEFAULT 'user',
            otp_secret TEXT
        );
        """

        # Create chat_history table
        create_chat_history_table = """
        CREATE TABLE IF NOT EXISTS chat_history (
            session_id TEXT NOT NULL,
            email TEXT NOT NULL,
            user_chat TEXT,
            bot_chat TEXT,
            timestamp TIMESTAMP NOT NULL,
            category TEXT
        );
        """

        # Create conversation_history table
        create_conversation_history_table = """
        CREATE TABLE IF NOT EXISTS conversation_history (
            user_session TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL
        );
        """

        # Create save_secure_session table 
        create_secure_session_table = '''
        CREATE TABLE IF NOT EXISTS user_sessions (
                email TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                expiry_time TIMESTAMP NOT NULL
            )
        '''

        # Execute queries
        cursor.execute(create_user_table)
        cursor.execute(create_chat_history_table)
        cursor.execute(create_conversation_history_table)
        cursor.execute(create_secure_session_table)
        connection.commit()
        print("Database initialized successfully!")

    except Exception as e:
        print(f"Error initializing database: {e}")
        if connection:
            connection.rollback()
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

@auth_bp.route('/')
def index():
    session.clear()
    return render_template('index.html')

@auth_bp.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and Password are required!'}), 400

        password_hash = hash_password(password)
        
        connection = get_connection()
        cursor = connection.cursor()
        
        cursor.execute('SELECT password_hash, role FROM users WHERE email = %s', (email,))
        row = cursor.fetchone()
        
        if not row:
            return jsonify({'error': 'Email not found'}), 404

        user_password_hash, role = row
        if user_password_hash == password_hash:
            session.clear()
            session['email'] = email
            session['role'] = role
            session['logged_in'] = True

            secure_session_id = str(uuid.uuid4())
            expiry_time = datetime.utcnow() + timedelta(minutes=120)

            cursor.execute('''
                    INSERT INTO user_sessions (email, session_id, expiry_time)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (email) 
                    DO UPDATE SET 
                        session_id = EXCLUDED.session_id,
                        expiry_time = EXCLUDED.expiry_time;
                    ''', (email, secure_session_id, expiry_time))
            connection.commit()
            response = make_response(jsonify({'success': True}))
            response.set_cookie(
                'session_id',
                secure_session_id,
                httponly=True,
                samesite=None,
                secure=False
            )
            return response
        else:
            return jsonify({'success': False,
                            'error': 'Password incorrect'}), 401

    except Exception as e:
        return jsonify({'error': f'An error occurred: {e}'}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()