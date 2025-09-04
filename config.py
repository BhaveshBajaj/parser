"""Main configuration file for the document processing system."""
import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = Field(default="Document Processing API", env="APP_NAME")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Database
    DATABASE_URL: str = Field(default="sqlite:///./documents.db", env="DATABASE_URL")
    DATABASE_ECHO: bool = Field(default=False, env="DATABASE_ECHO")
    
    # Azure OpenAI
    AZURE_OPENAI_API_KEY: Optional[str] = Field(default=None, env="AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT: Optional[str] = Field(default=None, env="AZURE_OPENAI_DEPLOYMENT")
    AZURE_OPENAI_API_VERSION: str = Field(default="2024-02-15-preview", env="AZURE_OPENAI_API_VERSION")
    
    # OpenAI (alternative)
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field(default="gpt-4", env="OPENAI_MODEL")
    
    # LangChain
    LANGCHAIN_API_KEY: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str = Field(default="document-processor", env="LANGCHAIN_PROJECT")
    LANGCHAIN_TRACING_V2: bool = Field(default=True, env="LANGCHAIN_TRACING_V2")
    
    # File Storage
    UPLOAD_DIR: str = Field(default="./data/uploads", env="UPLOAD_DIR")
    EMBEDDINGS_DIR: str = Field(default="./data/embeddings", env="EMBEDDINGS_DIR")
    METRICS_DIR: str = Field(default="./data/metrics", env="METRICS_DIR")
    
    # API
    API_BASE_URL: str = Field(default="http://localhost:8000", env="API_BASE_URL")
    CORS_ORIGINS: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Security
    SECRET_KEY: str = Field(default="your-secret-key-change-this", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Performance
    MAX_FILE_SIZE: int = Field(default=10485760, env="MAX_FILE_SIZE")  # 10MB
    MAX_CONCURRENT_UPLOADS: int = Field(default=5, env="MAX_CONCURRENT_UPLOADS")
    REQUEST_TIMEOUT: int = Field(default=300, env="REQUEST_TIMEOUT")  # 5 minutes
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()

# Ensure directories exist
def ensure_directories():
    """Ensure required directories exist."""
    directories = [
        settings.UPLOAD_DIR,
        settings.EMBEDDINGS_DIR,
        settings.METRICS_DIR,
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Initialize directories on import
ensure_directories()
