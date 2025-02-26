# /app_error/__init__.py
from .routes import app_error_bp

# Expose the blueprint so it can be easily imported elsewhere in the app
__all__ = ['app_error_bp']