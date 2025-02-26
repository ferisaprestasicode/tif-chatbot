from flask import Flask
from flask_cors import CORS
import os
from config import SERVICE_ACCOUNT_FILE
from auth.routes import auth_bp, init_db
from app_chatbot.routes import chat_gen_bp
from app_anomalies.routes import app_anomalies_bp
from app_error.routes import app_error_bp
from app_forecasting.routes import app_forecasting_bp, TrafficDataStore


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your_secret_key_here'
    CORS(app, supports_credentials=True, resources={r"/*": {"origins": ["http://127.0.0.1:8083", "http://ai.siscloud.id:8083", "http://36.67.62.245:8083"]}})
    
    # Initialize database
    init_db()
    
    # Initialize data store
    log_file_path = "data/processed/rrd_log.txt"
    data_store = TrafficDataStore(log_file_path)
    
    # Configure data store in forecasting routes
    import app_forecasting.routes as routes
    routes.data_store = data_store
    
    # Register blueprints
    blueprints = [
        auth_bp,
        chat_gen_bp,
        app_anomalies_bp,
        app_error_bp,
        app_forecasting_bp
    ]
    
    for blueprint in blueprints:
        app.register_blueprint(blueprint, url_prefix='/')
    
    # Set Google credentials
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = SERVICE_ACCOUNT_FILE
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host="0.0.0.0", port=8080)
