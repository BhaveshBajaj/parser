# Human-in-the-Loop Feedback UI Implementation

## Overview

This implementation provides a comprehensive human-in-the-loop feedback system for the document processing pipeline, allowing users to review, edit, and approve extracted entities and document summaries.

## Features Implemented

### 1. Entity Editor & Approval Workflow
- **Interactive Entity Review**: View all extracted entities with filtering and search capabilities
- **Entity Editing**: Modify entity text, type, and confidence scores
- **Bulk Operations**: Approve/reject multiple entities at once
- **Entity Management**: Add new entities, delete incorrect ones
- **Status Tracking**: Visual indicators for approved, rejected, and pending entities

### 2. Summary Editor & Approval Workflow
- **Summary Editing**: Edit document summaries with real-time preview
- **Section Context**: View document sections for reference while editing
- **Approval Workflow**: Approve or reset summaries
- **Version Control**: Track changes between original and edited summaries

### 3. Backend Integration
- **Feedback API Endpoints**: 
  - `POST /documents/{document_id}/feedback` - Submit feedback
  - `GET /documents/{document_id}/feedback` - Retrieve existing feedback
- **Data Persistence**: Store feedback in document metadata
- **Error Handling**: Comprehensive error handling and user feedback

### 4. Export & Reporting
- **Multiple Export Formats**: JSON and CSV export options
- **Complete Results**: Export all feedback data including statistics
- **Downloadable Reports**: Generate comprehensive feedback reports

## File Structure

```
doc_processor/
├── frontend/
│   ├── app.py                    # Main Streamlit application (updated)
│   └── entity_editor.py          # New: Entity and summary editor components
└── app/
    └── main.py                   # FastAPI backend (updated with feedback endpoints)
```

## Key Components

### EntityEditor Class
- Manages entity editing workflow
- Handles bulk operations and filtering
- Provides export functionality
- Maintains session state for editing progress

### SummaryEditor Class
- Manages summary editing workflow
- Provides context from document sections
- Handles approval and reset operations
- Tracks changes and versions

### Backend API Endpoints
- **FeedbackSubmission Model**: Structured data model for feedback
- **FeedbackResponse Model**: Response model for API calls
- **Error Handling**: Comprehensive error handling and logging

## Usage Instructions

### 1. Starting the Application
```bash
# Start the backend API
cd /Users/bhavesh/Desktop/parser
python run_api.py

# Start the frontend (in another terminal)
cd /Users/bhavesh/Desktop/parser/doc_processor/frontend
streamlit run app.py
```

### 2. Using the Feedback System
1. **Upload a Document**: Use the upload page to process a document
2. **Navigate to Feedback**: Click "🔄 Review & Approve" from the upload results or dashboard
3. **Review Entities**: 
   - Filter entities by type, confidence, or search terms
   - Edit entity text, types, and confidence scores
   - Approve or reject entities individually or in bulk
   - Add new entities if needed
4. **Edit Summary**: 
   - Review and edit the document summary
   - Use section context for reference
   - Approve the final summary
5. **Submit Feedback**: Click "Submit Feedback to Backend" to save all changes
6. **Export Results**: Download feedback data in JSON or CSV format

### 3. API Usage
```python
# Submit feedback
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

## Features in Detail

### Entity Management
- **Filtering**: Filter by entity type, confidence threshold, or search terms
- **Bulk Actions**: Approve/reject all visible entities at once
- **Individual Editing**: Modify text, type, and confidence for each entity
- **Context Display**: Show entity context and metadata
- **Position Tracking**: Display entity positions in the document

### Summary Management
- **Side-by-Side Editing**: Compare original and edited summaries
- **Section Context**: View document sections for reference
- **Real-time Preview**: See changes as you type
- **Approval Workflow**: Clear approve/reject/reset options

### Data Export
- **JSON Export**: Complete feedback data in structured format
- **CSV Export**: Entity data in spreadsheet format
- **Statistics**: Summary of approved, rejected, and edited items
- **Timestamp Tracking**: All changes timestamped for audit trail

## Benefits

1. **Improved Accuracy**: Human feedback improves entity extraction accuracy
2. **Quality Control**: Review and approve all extracted content
3. **Flexibility**: Edit and customize extracted information
4. **Audit Trail**: Complete tracking of all changes and approvals
5. **Export Options**: Multiple formats for downstream processing
6. **User-Friendly**: Intuitive interface for non-technical users

## Future Enhancements

1. **User Authentication**: Add user management and authentication
2. **Batch Processing**: Handle multiple documents simultaneously
3. **Machine Learning**: Use feedback to improve extraction models
4. **Collaboration**: Multiple users reviewing the same document
5. **Advanced Analytics**: Detailed reporting on feedback patterns
6. **Integration**: Connect with external systems and workflows

## Technical Notes

- **Session State**: Uses Streamlit session state for maintaining editing progress
- **Error Handling**: Comprehensive error handling throughout the application
- **Performance**: Optimized for handling large documents with many entities
- **Scalability**: Backend designed to handle multiple concurrent users
- **Data Integrity**: All changes validated before submission

This implementation provides a complete human-in-the-loop feedback system that significantly enhances the document processing pipeline by allowing users to review, edit, and approve extracted entities and summaries.
