# Document Processing System

A lightweight document processing system with FastAPI backend and Streamlit UI, featuring document upload, status tracking, and summarization capabilities.

## Features

- **Document Upload**: Upload PDF, TXT, MD, and DOCX files for processing
- **Status Tracking**: Monitor the status of document processing
- **Summarization**: Generate summaries of processed documents
- **Q&A**: Ask questions about the uploaded documents (mock implementation)
- **Agent System**: Extensible agent-based architecture for document processing
- **LangSmith Integration**: Optional integration with LangSmith for logging and tracing

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd parser
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Update the `.env` file with your configuration (if needed)

## Running the Application

### Backend (FastAPI)

Start the FastAPI backend server:
```bash
uvicorn doc_processor.app.main:app --reload
```

The API will be available at `http://localhost:8000`

### Frontend (Streamlit)

In a separate terminal, start the Streamlit UI:
```bash
streamlit run doc_processor/frontend/app.py
```

The UI will be available at `http://localhost:8501`

## API Endpoints

- `POST /api/v1/upload`: Upload a document for processing
- `GET /api/v1/status/{id}`: Get the status of a document
- `GET /api/v1/summaries/{id}`: Get the summary of a processed document
- `POST /api/v1/qa`: Ask a question about a document (mock implementation)

## Project Structure

```
doc_processor/
├── app/
│   ├── api/               # API endpoints
│   ├── core/              # Core application configuration
│   ├── models/            # Data models
│   ├── services/          # Business logic and services
│   ├── utils/             # Utility functions
│   └── main.py            # FastAPI application
├── frontend/              # Streamlit UI
├── tests/                 # Unit and integration tests
└── data/                  # Uploaded documents and metadata
```

## Testing

Run the test suite with pytest:
```bash
pytest
```

## Configuration

Edit the `.env` file to configure the application:

```env
# Application settings
DEBUG=True
LOG_LEVEL=INFO

# Server settings
HOST=0.0.0.0
PORT=8000

# File upload settings
UPLOAD_FOLDER=./data/uploads
MAX_CONTENT_LENGTH=16777216  # 16MB in bytes
ALLOWED_EXTENSIONS=pdf,txt,md,docx

# LangSmith settings (optional)
# LANGSMITH_API_KEY=your_api_key_here
# LANGSMITH_PROJECT=doc-processor
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [LangSmith](https://www.langchain.com/langsmith) (optional)
