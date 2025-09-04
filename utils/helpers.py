"""Helper functions and utilities for the document processing system."""
import os
import json
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime, timezone
from uuid import UUID, uuid4

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading JSON file {file_path}: {e}")
        return {}

def save_json_file(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """Save data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Error saving JSON file {file_path}: {e}")
        return False

def ensure_directory_exists(directory_path: Union[str, Path]) -> bool:
    """Ensure a directory exists, create it if it doesn't."""
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Error creating directory {directory_path}: {e}")
        return False

def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes."""
    try:
        return Path(file_path).stat().st_size
    except Exception:
        return 0

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid4())

def is_valid_uuid(uuid_string: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        UUID(uuid_string)
        return True
    except ValueError:
        return False

def get_current_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()

def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse timestamp string to datetime object."""
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError:
        return None

def clean_filename(filename: str) -> str:
    """Clean filename by removing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename.strip()

def get_file_extension(filename: str) -> str:
    """Get file extension from filename."""
    return Path(filename).suffix.lower()

def is_supported_file_type(filename: str) -> bool:
    """Check if file type is supported."""
    supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.html', '.htm']
    return get_file_extension(filename) in supported_extensions

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries."""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result

def safe_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get value from dictionary with default."""
    return dictionary.get(key, default) if dictionary else default

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists."""
    return [item for sublist in nested_list for item in sublist]

def remove_duplicates(lst: List[Any]) -> List[Any]:
    """Remove duplicates from list while preserving order."""
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]

def validate_email(email: str) -> bool:
    """Basic email validation."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_url(url: str) -> bool:
    """Basic URL validation."""
    import re
    pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
    return re.match(pattern, url) is not None
