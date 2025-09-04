"""Configuration management for the document processor."""
import os
from pathlib import Path
from typing import List, Optional, Any, Dict

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    # Application settings
    APP_NAME: str = Field("Document Processor")
    DEBUG: bool = Field(True)
    LOG_LEVEL: str = Field("DEBUG")
    
    # Server settings
    API_HOST: str = Field("0.0.0.0")
    API_PORT: int = Field(8000)
    API_V1_PREFIX: str = Field("/api/v1")
    SECRET_KEY: str = Field("your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(60 * 24 * 8)  # 8 days
    
    # File upload settings
    UPLOAD_FOLDER: str = Field("data/uploads")
    MAX_CONTENT_LENGTH: int = Field(16 * 1024 * 1024)  # 16MB
    ALLOWED_EXTENSIONS: str = Field("pdf,docx,doc,txt", description="Comma-separated list of allowed file extensions")
    ALLOWED_MIME_TYPES: List[str] = Field(default_factory=lambda: [
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    ])
    
    # Azure OpenAI settings (optional for now)
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(None, description="Azure OpenAI endpoint URL")
    AZURE_OPENAI_API_KEY: Optional[str] = Field(None, description="Azure OpenAI API key")
    AZURE_OPENAI_API_VERSION: str = Field("2024-02-15-preview", description="Azure OpenAI API version")
    AZURE_OPENAI_DEPLOYMENT: str = Field("gpt-4o", description="Azure OpenAI deployment name")
    
    # Entity extraction settings
    ENTITY_TYPES: List[str] = Field(
        default_factory=lambda: ["PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY", "PERCENT", "PRODUCT", "EVENT"],
        description="List of entity types to extract"
    )
    
    # LangSmith settings (optional)
    LANGSMITH_API_KEY: Optional[str] = Field(None)
    LANGSMITH_PROJECT: str = Field("document-processor")

    # Pydantic v2 config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

# Create settings instance
settings = Settings()
