# Human-in-the-Loop Feedback UI - Usage Guide

## Quick Start

### Option 1: Using the Startup Script (Recommended)
```bash
cd /Users/bhavesh/Desktop/parser
source venv/bin/activate
python start_feedback_ui.py
```

### Option 2: Manual Startup
```bash
# Terminal 1: Start Backend
cd /Users/bhavesh/Desktop/parser
source venv/bin/activate
python run_api.py

# Terminal 2: Start Frontend
cd /Users/bhavesh/Desktop/parser
source venv/bin/activate
streamlit run doc_processor/frontend/app.py
```

## Access the Application

- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## How to Use the Feedback System

### 1. Upload a Document
1. Go to the "📤 Upload" page
2. Select a document (PDF, DOCX, TXT, HTML)
3. Add optional metadata
4. Click "📤 Upload & Process"
5. Wait for processing to complete

### 2. Review and Approve Entities
1. After upload, click "🔄 Review & Approve"
2. In the "🔍 Entity Editor" tab:
   - **Filter entities** by type, confidence, or search terms
   - **Edit entities** by clicking on them and modifying text, type, or confidence
   - **Bulk approve/reject** using the bulk action buttons
   - **Add new entities** using the "Add New Entity" section
   - **Delete incorrect entities** using the delete button

### 3. Edit and Approve Summary
1. Go to the "📝 Summary Editor" tab
2. **Review the original summary** in the expandable section
3. **Edit the summary** in the text area
4. **Use section context** for reference while editing
5. **Save changes** and **approve** the final summary

### 4. Submit Feedback
1. Go to the "📊 Review & Export" tab
2. **Review the summary** of all changes
3. **Click "Submit Feedback to Backend"** to save all changes
4. **Export results** in JSON or CSV format if needed

## Key Features

### Entity Management
- ✅ **Visual Status Indicators**: Green (approved), Red (rejected), Yellow (pending)
- 🔍 **Advanced Filtering**: Filter by entity type, confidence threshold, or search terms
- 📝 **Inline Editing**: Click on entities to edit text, type, and confidence
- ⚡ **Bulk Operations**: Approve/reject multiple entities at once
- ➕ **Add New Entities**: Manually add entities that were missed
- 🗑️ **Delete Entities**: Remove incorrect or irrelevant entities

### Summary Management
- 📄 **Side-by-Side View**: Compare original and edited summaries
- 📑 **Section Context**: View document sections for reference
- ✏️ **Real-time Editing**: Edit summaries with live preview
- ✅ **Approval Workflow**: Clear approve/reject/reset options

### Data Export
- 📊 **JSON Export**: Complete feedback data in structured format
- 📈 **CSV Export**: Entity data in spreadsheet format
- 📋 **Statistics**: Summary of all changes and approvals
- ⏰ **Audit Trail**: Timestamped changes for tracking

## Troubleshooting

### Import Errors
If you see "ImportError: attempted relative import with no known parent package":
- Make sure you're running from the correct directory
- Use the startup script instead of running streamlit directly
- Check that all files are in the correct locations

### Backend Connection Issues
- Ensure the backend API is running on port 8000
- Check that no other services are using the same ports
- Verify the API_BASE_URL environment variable if needed

### Entity Extraction Issues
- Check that Azure OpenAI credentials are configured
- Verify that the document was processed successfully
- Look at the backend logs for detailed error messages

## API Usage

### Submit Feedback Programmatically
```python
import requests

feedback_data = {
    "document_id": "your-document-id",
    "entities": {
        "approved": ["entity1", "entity2"],
        "rejected": ["entity3"],
        "edited": {"entity1": {"text": "new text", "type": "PERSON"}},
        "new": [{"text": "new entity", "type": "ORG"}]
    },
    "summary": {
        "original_summary": "original text",
        "edited_summary": "edited text",
        "is_approved": True
    }
}

response = requests.post(
    "http://localhost:8000/documents/your-document-id/feedback",
    json=feedback_data
)
```

### Get Existing Feedback
```python
response = requests.get(
    "http://localhost:8000/documents/your-document-id/feedback"
)
feedback = response.json()
```

## Tips for Best Results

1. **Start with High-Confidence Entities**: Focus on entities with high confidence scores first
2. **Use Bulk Operations**: Save time by approving/rejecting similar entities in groups
3. **Review Context**: Always check the entity context before making decisions
4. **Edit Summaries Carefully**: Use section context to ensure summary accuracy
5. **Export Regularly**: Download your feedback data for backup and analysis
6. **Check Previous Feedback**: The system remembers previous feedback for each document

## Support

If you encounter any issues:
1. Check the console logs for error messages
2. Verify that all services are running correctly
3. Ensure you have the required dependencies installed
4. Check the API documentation at http://localhost:8000/docs

The feedback system is designed to be intuitive and user-friendly, but don't hesitate to explore all the features to get the most out of the human-in-the-loop workflow!
