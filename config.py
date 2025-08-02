"""
Deployment Configuration for Kotori.ai
Handles environment variables and path configuration for different deployment environments.
"""

import os
from pathlib import Path

# Base directory - where this config file is located
BASE_DIR = Path(__file__).parent

# Environment variables with fallbacks for deployment
class Config:
    # Database paths
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma"))
    DATA_DIR_PATH = os.getenv("DATA_DIR_PATH", str(BASE_DIR / "data"))
    
    # Asset paths
    ASSETS_DIR = os.getenv("ASSETS_DIR", str(BASE_DIR / "assets"))
    LOGO_PATH = os.getenv("LOGO_PATH", str(BASE_DIR / "assets" / "images" / "image.png"))
    
    # API Configuration
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # App Configuration
    APP_TITLE = os.getenv("APP_TITLE", "Kotori.ai")
    APP_DESCRIPTION = os.getenv("APP_DESCRIPTION", "Your Compassionate Companion for Empty Nest Syndrome")
    
    # Deployment settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development, staging, production
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        Path(cls.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
        Path(cls.DATA_DIR_PATH).mkdir(parents=True, exist_ok=True)
        Path(cls.ASSETS_DIR).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Validate that required environment variables are set"""
        required_vars = ["HUGGINGFACE_API_TOKEN"]
        missing_vars = []
        
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True

# Initialize configuration
config = Config()
