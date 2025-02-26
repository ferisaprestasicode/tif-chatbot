# /auth/__init__.py
from .routes import auth_bp

# Expose the blueprint so it can be easily imported elsewhere in the app
__all__ = ['auth_bp']