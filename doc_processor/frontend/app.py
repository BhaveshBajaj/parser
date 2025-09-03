"""
Streamlit UI for the Document Processing System with Entity Extraction

This application allows you to upload documents, view their processing status,
extract entities, and analyze the content.
"""
import base64
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID, uuid4

import pandas as pd
import requests
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Import the new entity editor components
try:
    from .entity_editor import display_feedback_workflow, EntityEditor, SummaryEditor
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from entity_editor import display_feedback_workflow, EntityEditor, SummaryEditor

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")  # Base URL without /api/v1 prefix

# Set page config
st.set_page_config(
    page_title="Document Processor",
    page_icon="📄",
    layout="wide",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .success { color: #28a745; }
    .error { color: #dc3545; }
    .warning { color: #ffc107; }
    .info { color: #17a2b8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper functions
def make_api_request(
    method: str, 
    endpoint: str, 
    timeout: int = 30,
    **kwargs
) -> Tuple[int, Dict]:
    """
    Make an API request with improved error handling and timeout.
    
    Args:
        method: HTTP method (GET, POST, PUT, DELETE)
        endpoint: API endpoint (without base URL)
        timeout: Request timeout in seconds
        **kwargs: Additional arguments to pass to requests
        
    Returns:
        Tuple of (status_code, response_data)
    """
    url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"
    headers = kwargs.pop('headers', {})
    
    # Only set Content-Type for non-file requests
    if 'files' not in kwargs:
        headers.setdefault('Content-Type', 'application/json')
    
    headers.setdefault('Accept', 'application/json')
    
    # Add request ID for tracking
    request_id = str(uuid4())
    headers['X-Request-ID'] = request_id
    
    try:
        start_time = time.time()
        
        if method.upper() == 'GET':
            response = requests.get(
                url, 
                headers=headers, 
                timeout=timeout,
                **kwargs
            )
        elif method.upper() == 'POST':
            response = requests.post(
                url, 
                headers=headers, 
                timeout=timeout,
                **kwargs
            )
        elif method.upper() == 'PUT':
            response = requests.put(
                url, 
                headers=headers, 
                timeout=timeout,
                **kwargs
            )
        elif method.upper() == 'DELETE':
            response = requests.delete(
                url, 
                headers=headers, 
                timeout=timeout,
                **kwargs
            )
        else:
            return 405, {"error": f"Method not allowed: {method}"}
        
        response_time = (time.time() - start_time) * 1000  # in ms
        
        # Log the request
        st.session_state.setdefault('api_logs', []).append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'method': method.upper(),
            'url': url,
            'status_code': response.status_code,
            'response_time_ms': round(response_time, 2),
            'request_id': request_id
        })
        
        # Handle different response types
        content_type = response.headers.get('Content-Type', '')
        
        if 'application/json' in content_type:
            try:
                return response.status_code, response.json()
            except ValueError:
                return response.status_code, {"error": "Invalid JSON response"}
        elif 'text/' in content_type:
            return response.status_code, {"text": response.text}
        else:
            # For binary responses
            return response.status_code, {"content": response.content}
            
    except requests.exceptions.Timeout:
        return 504, {"error": "Request timed out"}
    except requests.exceptions.ConnectionError:
        return 503, {"error": "Could not connect to the server"}
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        status_code = getattr(e.response, 'status_code', 500)
        
        try:
            if e.response is not None:
                error_data = e.response.json()
                error_msg = error_data.get('detail', error_data.get('error', error_msg))
        except (ValueError, AttributeError):
            pass
            
        return status_code, {"error": error_msg}

def upload_document(file: UploadedFile, metadata: Optional[Dict] = None) -> Tuple[bool, str, Optional[Dict]]:
    """
    Upload a document to the API for processing.
    
    Args:
        file: The uploaded file object from Streamlit
        metadata: Optional metadata to include with the upload
        
    Returns:
        Tuple of (success: bool, message: str, data: Optional[Dict])
    """
    if not file:
        return False, "No file provided", None
    
    # Prepare the file for upload
    files = {"file": (file.name, file, file.type or "application/octet-stream")}
    data = {"metadata": json.dumps(metadata or {})}
    
    try:
        status_code, response = make_api_request(
            "POST",
            "/documents",
            files=files,
            data=data,
            timeout=60  # Longer timeout for file uploads
        )
        
        if status_code == 202:  # Accepted
            return True, "Document uploaded successfully", response
        else:
            error_msg = response.get("detail", response.get("error", "Unknown error"))
            return False, f"Upload failed: {error_msg}", None
            
    except Exception as e:
        return False, f"Error during upload: {str(e)}", None

def display_upload_progress(progress_bar, status_text, progress: float, message: str):
    """Update the upload progress bar and status text."""
    if progress_bar:
        progress_bar.progress(progress)
    if status_text:
        status_text.text(message)

# Page: Upload
def upload_page():
    """Upload a document for processing."""
    st.title("📤 Upload Document")
    
    with st.container():
        st.markdown("""
        Upload a document to extract structured information. 
        Supported formats: PDF, DOCX, DOC, TXT, HTML
        """)
    
    with st.form("upload_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a document",
                type=["pdf", "docx", "doc", "txt", "html", "htm"],
        )
        
        # Document metadata
        with st.expander("Document Metadata (Optional)", expanded=False):
            doc_metadata = {}
            col1, col2 = st.columns(2)
            with col1:
                doc_metadata["title"] = st.text_input("Title", "")
                doc_metadata["author"] = st.text_input("Author", "")
            with col2:
                doc_metadata["source"] = st.text_input("Source", "")
                doc_metadata["language"] = st.selectbox(
                    "Language", ["en", "es", "fr", "de", "zh", "ja"], index=0
                )
        
        submitted = st.form_submit_button("📤 Upload & Process")
        
        if submitted and uploaded_file is not None:
            # Initialize UI elements
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Upload the file
                display_upload_progress(progress_bar, status_text, 0.2, "Uploading document...")
                
                # Prepare metadata
                metadata = {
                    "filename": uploaded_file.name,
                    "content_type": uploaded_file.type or "application/octet-stream",
                    "size_bytes": uploaded_file.size,
                    **{k: v for k, v in doc_metadata.items() if v}  # Only include non-empty metadata
                }
                
                # Upload the file
                success, message, data = upload_document(uploaded_file, metadata)
                
                if not success:
                    st.error(f"❌ {message}")
                    return
                
                document_id = data.get("id")
                if not document_id:
                    st.error("❌ No document ID returned from server")
                    return
                
                # Store document ID in session for status tracking
                st.session_state.current_document = {
                    "id": document_id,
                    "filename": uploaded_file.name,
                    "upload_time": datetime.now(timezone.utc).isoformat(),
                    "status": "uploaded"
                }
                
                # Step 2: Start document processing
                display_upload_progress(progress_bar, status_text, 0.5, "Processing document...")
                
                # Poll for processing status
                max_attempts = 30  # 30 attempts with 2s delay = 1 minute max wait
                for attempt in range(max_attempts):
                    status_code, status_data = make_api_request(
                        "GET", 
                        f"documents/{document_id}/status"
                    )
                    
                    if status_code == 200:
                        status = status_data.get("status", "unknown").lower()
                        st.session_state.current_document["status"] = status
                        
                        if status in ["processed", "completed"]:
                            display_upload_progress(
                                progress_bar, status_text, 1.0, 
                                "✅ Document processed successfully!"
                            )
                            st.balloons()
                            
                            # Get full document details
                            doc_status_code, doc_data = make_api_request("GET", f"documents/{document_id}")
                            if doc_status_code == 200:
                                st.session_state.current_document.update({
                                    "sections": doc_data.get("sections", []),
                                    "entities": doc_data.get("entities", [])
                                })
                            
                            st.experimental_rerun()  # Refresh to show results
                            return
                            
                        elif status in ["failed", "error"]:
                            error_msg = status_data.get("error", "Unknown error")
                            display_upload_progress(
                                progress_bar, status_text, 0.0,
                                f"❌ Processing failed: {error_msg}"
                            )
                            return
                            
                        # Update progress based on status
                        progress = min(0.5 + (attempt / max_attempts * 0.5), 0.9)
                        display_upload_progress(
                            progress_bar, status_text, progress,
                            f"Processing document... (Status: {status.capitalize()})"
                        )
                    
                    # Wait before next poll
                    time.sleep(2)
                
                # If we get here, processing is taking too long
                display_upload_progress(
                    progress_bar, status_text, 0.9,
                    "⚠️ Processing is taking longer than expected. "
                    "Check the status in the dashboard later."
                )
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
            finally:
                # Clean up UI elements
                with st.spinner("Finalizing..."):
                    if 'progress_bar' in locals():
                        progress_bar.empty()
                    if 'status_text' in locals():
                        status_text.empty()
                    
                    # Show error message if processing failed
                    if st.session_state.current_document and \
                       st.session_state.current_document.get('status') not in ['processed', 'completed']:
                        st.error("Document processing failed or timed out.")
    
    # Display results and action buttons outside the form
    if 'current_document' in st.session_state and st.session_state.current_document:
        # Show results from previous upload
        doc = st.session_state.current_document
        
        if doc.get('status') in ['processed', 'completed']:
            with st.expander("📄 Document Details", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Filename", doc.get('filename', 'N/A'))
                    st.metric("Status", doc.get('status', 'unknown').title())
                with col2:
                    st.metric("Sections", len(doc.get('sections', [])))
                    st.metric("Entities", len(doc.get('entities', [])))
            
            # Show quick actions
            st.markdown("### Next Steps")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔍 View Document", use_container_width=True):
                    st.session_state.current_page = "view"
                    st.experimental_set_query_params(page="view")
                    st.experimental_rerun()
            with col2:
                if st.button("🔄 Review & Approve", use_container_width=True):
                    st.session_state.current_page = "feedback"
                    st.experimental_set_query_params(page="feedback")
                    st.experimental_rerun()
            with col3:
                if st.button("📝 Process Another", use_container_width=True):
                    st.session_state.current_document = None
                    st.experimental_rerun()
        elif doc.get('status') in ['uploaded', 'processing']:
            st.info("Document is still being processed. Please wait...")
            if st.button("🔄 Refresh Status", use_container_width=True):
                st.experimental_rerun()

def display_document_summary(doc_id: str):
    """Display structured summary of a document."""
    try:
        status_code, doc_data = make_api_request("GET", f"documents/{doc_id}")
        
        if status_code != 200:
            st.error(f"Failed to load document: {doc_data.get('detail', 'Unknown error')}")
            return
        
        with st.container():
            st.subheader("📄 Document Overview")
            
            # Document metadata
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Status", doc_data.get("status", "unknown").title())
                st.metric("Pages", doc_data.get("page_count", "N/A"))
                st.metric("Language", doc_data.get("language", "en").upper())
            
            with col2:
                st.metric("Created", doc_data.get("created_at", "N/A"))
                st.metric("Last Updated", doc_data.get("updated_at", "N/A"))
                
            # Document sections
            st.subheader("📑 Document Sections")
            sections = doc_data.get("sections", [])
            
            if not sections:
                st.info("No sections found in this document.")
            else:
                for section in sections:
                    with st.expander(f"{section.get('title', 'Untitled Section')}", expanded=False):
                        st.markdown(f"**Page {section.get('page_number', 'N/A')}**")
                        st.caption(f"Type: {section.get('type', 'text').title()}")
                        st.write(section.get("content", "No content"))
                        
                        # Show section metadata if available
                        if "metadata" in section and section["metadata"]:
                            with st.expander("Section Metadata"):
                                st.json(section["metadata"])
            
            # Document entities
            st.subheader("🔍 Extracted Entities")
            entities = doc_data.get("entities", [])
            
            if not entities:
                st.info("No entities were extracted from this document.")
            else:
                # Group entities by type
                entities_by_type = {}
                for entity in entities:
                    entity_type = entity.get("type", entity.get("entity_type", "other"))
                    if entity_type not in entities_by_type:
                        entities_by_type[entity_type] = []
                    entities_by_type[entity_type].append(entity)
                
                # Create tabs for each entity type
                tabs = st.tabs([f"{k.title()} ({len(v)})" for k, v in entities_by_type.items()])
                
                for (entity_type, entities_list), tab in zip(entities_by_type.items(), tabs):
                    with tab:
                        for entity in entities_list:
                            with st.expander(entity["text"], expanded=False):
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    entity_type = entity.get("type", entity.get("entity_type", "unknown"))
                                    st.metric("Type", entity_type.title())
                                    if "confidence" in entity:
                                        st.metric("Confidence", f"{entity['confidence']:.0%}")
                                with col2:
                                    if "metadata" in entity and entity["metadata"]:
                                        st.json(entity["metadata"])
    
    except Exception as e:
        st.error(f"Error displaying document: {str(e)}")


def display_entity_analysis(doc_id: str):
    """Display detailed entity analysis for a document."""
    try:
        status_code, doc_data = make_api_request("GET", f"documents/{doc_id}")
        
        if status_code != 200:
            st.error(f"Failed to load document: {doc_data.get('detail', 'Unknown error')}")
            return
        
        entities = doc_data.get("entities", [])
        if not entities:
            st.info("No entities were extracted from this document.")
            return
        
        # Entity type distribution
        st.subheader("📊 Entity Type Distribution")
        entity_counts = {}
        for entity in entities:
            entity_type = entity.get("entity_type", "other")
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        if entity_counts:
            import plotly.express as px
            
            # Create pie chart
            fig = px.pie(
                names=list(entity_counts.keys()),
                values=list(entity_counts.values()),
                title="Entity Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Entity list with filters
            st.subheader("🔍 Entity Explorer")
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                selected_type = st.selectbox(
                    "Filter by type",
                    ["All"] + sorted(list(entity_counts.keys()))
                )
            with col2:
                min_confidence = st.slider(
                    "Minimum confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    format="%.2f"
                )
            
            # Filter entities
            filtered_entities = [
                e for e in entities 
                if (selected_type == "All" or e.get("type", e.get("entity_type")) == selected_type)
                and e.get("confidence", 1.0) >= min_confidence
            ]
            
            # Display filtered entities
            for entity in filtered_entities:
                entity_type = entity.get("type", entity.get("entity_type", "unknown"))
                with st.expander(f"{entity['text']} ({entity_type})", expanded=False):
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Type", entity_type.title())
                    with cols[1]:
                        st.metric("Confidence", f"{entity.get('confidence', 1.0):.0%}")
                    with cols[2]:
                        if "section" in entity:
                            st.metric("Section", entity["section"].get("title", "Unknown"))
                    
                    # Show entity context if available
                    if "context" in entity:
                        st.caption("Context:")
                        st.write(f"...{entity['context']}...")
                    
                    # Show metadata if available
                    if "metadata" in entity and entity["metadata"]:
                        with st.expander("Metadata"):
                            st.json(entity["metadata"])
    
    except Exception as e:
        st.error(f"Error in entity analysis: {str(e)}")


# Page: Status
def status_page():
    """View document status and details."""
    # Get document ID from URL or session
    query_params = st.experimental_get_query_params()
    doc_id = query_params.get("doc_id", [None])[0] or st.session_state.get("last_doc_id")
    
    if not doc_id:
        st.warning("No document ID provided. Please upload a document first.")
        return
    
    # Navigation
    st.sidebar.title("Navigation")
    view_mode = st.sidebar.radio(
        "View Mode",
        ["📄 Summary", "🔍 Entities", "📊 Analysis"],
        index=0
    )
    
    # Document actions
    st.sidebar.title("Document Actions")
    if st.sidebar.button("🔄 Refresh"):
        st.experimental_rerun()
    
    if st.sidebar.button("📝 Process New"):
        st.session_state.current_document = None
        st.experimental_set_query_params()
        st.experimental_rerun()
    
    # Main content area
    if view_mode == "📄 Summary":
        st.title("📄 Document Summary")
        display_document_summary(doc_id)
    
    elif view_mode == "🔍 Entities":
        st.title("🔍 Extracted Entities")
        display_entity_analysis(doc_id)
    
    elif view_mode == "📊 Analysis":
        st.title("📊 Document Analysis")
        
        # Get document data
        status_code, doc_data = make_api_request("GET", f"documents/{doc_id}")
        if status_code != 200:
            st.error(f"Failed to load document: {doc_data.get('detail', 'Unknown error')}")
            return
        
        # Display analysis tabs
        tab1, tab2, tab3 = st.tabs(["📈 Overview", "📊 Statistics", "🔍 Advanced Analysis"])
        
        with tab1:
            st.subheader("Document Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Pages", doc_data.get("page_count", 0))
                st.metric("Word Count", doc_data.get("word_count", "N/A"))
            
            with col2:
                st.metric("Sections", len(doc_data.get("sections", [])))
                st.metric("Entities", len(doc_data.get("entities", [])))
            
            with col3:
                st.metric("Created", doc_data.get("created_at", "N/A"))
                st.metric("Status", doc_data.get("status", "unknown").title())
            
            # Document preview
            st.subheader("Document Preview")
            if "sections" in doc_data and doc_data["sections"]:
                preview_section = doc_data["sections"][0]
                st.text_area(
                    "First Section Preview",
                    value=preview_section.get("content", "No content"),
                    height=200,
                    disabled=True
                )
        
        with tab2:
            st.subheader("Document Statistics")
            
            # Entity type distribution
            entities = doc_data.get("entities", [])
            if entities:
                import pandas as pd
                import plotly.express as px
                
                # Entity type distribution
                entity_counts = {}
                for entity in entities:
                    entity_type = entity.get("type", entity.get("entity_type", "other"))
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                
                if entity_counts:
                    df = pd.DataFrame({
                        "Entity Type": list(entity_counts.keys()),
                        "Count": list(entity_counts.values())
                    })
                    
                    fig = px.bar(
                        df,
                        x="Entity Type",
                        y="Count",
                        title="Entity Type Distribution",
                        color="Entity Type"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Document structure
            sections = doc_data.get("sections", [])
            if sections:
                section_types = {}
                for section in sections:
                    section_type = section.get("type", "unknown")
                    section_types[section_type] = section_types.get(section_type, 0) + 1
                
                if section_types:
                    df = pd.DataFrame({
                        "Section Type": list(section_types.keys()),
                        "Count": list(section_types.values())
                    })
                    
                    fig = px.pie(
                        df,
                        names="Section Type",
                        values="Count",
                        title="Section Type Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Advanced Analysis")
            
            # NER visualization
            st.markdown("### Named Entity Recognition")
            
            if "sections" in doc_data and doc_data["sections"] and entities:
                # Show a section with highlighted entities
                section = st.selectbox(
                    "Select section to analyze",
                    [f"Section {i+1}: {s.get('title', 'Untitled')}" for i, s in enumerate(doc_data["sections"])],
                    index=0
                )
                
                section_idx = int(section.split(":")[0].split()[-1]) - 1
                selected_section = doc_data["sections"][section_idx]
                
                # Get entities in this section
                section_entities = [
                    e for e in entities 
                    if e.get("section", {}).get("id") == selected_section.get("id")
                ]
                
                if section_entities:
                    # Display section text with highlighted entities
                    text = selected_section.get("content", "")
                    marked_text = text
                    
                    # Sort entities by start position (descending) to avoid offset issues
                    sorted_entities = sorted(
                        section_entities,
                        key=lambda x: x.get("start_pos", 0),
                        reverse=True
                    )
                    
                    # Highlight entities in text
                    for entity in sorted_entities:
                        start = entity.get("start", entity.get("start_pos", 0))
                        end = entity.get("end", entity.get("end_pos", 0))
                        if 0 <= start < end <= len(text):
                            entity_text = text[start:end]
                            entity_type = entity.get("type", entity.get("entity_type", ""))
                            marked_text = (
                                marked_text[:start] +
                                f'<span style="background-color: #ffd70080; border-radius: 3px; padding: 0 2px;">' +
                                f'<strong>{entity_text}</strong> ({entity_type})' +
                                '</span>' +
                                marked_text[end:]
                            )
                    
                    st.markdown("### Entity Highlights")
                    st.markdown(marked_text, unsafe_allow_html=True)
                    
                    # Entity details
                    st.markdown("### Entity Details")
                    for entity in section_entities:
                        entity_type = entity.get("type", entity.get("entity_type", "unknown"))
                        with st.expander(f"{entity.get('text')} ({entity_type})"):
                            st.metric("Type", entity_type.title())
                            if "confidence" in entity:
                                st.metric("Confidence", f"{entity['confidence']:.0%}")
                            if "metadata" in entity and entity["metadata"]:
                                st.json(entity["metadata"])
                else:
                    st.info("No entities found in this section.")
                    
                    # Fallback to show section content
                    st.text_area(
                        "Section Content",
                        value=selected_section.get("content", "No content"),
                        height=300,
                        disabled=True
                    )
            else:
                st.info("No sections or entities available for advanced analysis.")
    
    # Add some spacing at the bottom
    st.markdown("<br><br>", unsafe_allow_html=True)
    with st.spinner("Fetching document status..."):
        status_code, response = make_api_request("GET", f"status/{doc_id}")
        
        if status_code == 200:
            st.json(response)
            
            # Display status with appropriate icon
            status = response.get("status", "unknown").lower()
            if status == "uploaded":
                st.info("Document has been uploaded and is waiting to be processed.")
            elif status == "processing":
                st.warning("Document is currently being processed...")
            elif status == "summarized":
                st.success("Document has been processed successfully!")
                
                # Add a button to view the summary
                if st.button("View Summary"):
                    st.experimental_set_query_params(page="summary", doc_id=doc_id)
                    st.experimental_rerun()
            elif status == "failed":
                st.error(f"Document processing failed: {response.get('error', 'Unknown error')}")
        else:
            error = response.get("error", "Failed to fetch document status")
            st.error(f"Error: {error}")

# Page: Summary
def summary_page():
    """View document summary."""
    st.title("📝 Document Summary")
    
    # Get document ID from multiple sources
    query_params = st.experimental_get_query_params()
    doc_id = (query_params.get("doc_id", [None])[0] or 
              st.session_state.get("last_doc_id") or
              (st.session_state.get("current_document", {}).get("id") if st.session_state.get("current_document") else None))
    
    if not doc_id:
        st.warning("No document selected. Please select a document from the dashboard or upload a new one.")
        
        # Show available documents for selection
        st.markdown("### Available Documents")
        with st.spinner("Loading documents..."):
            status_code, response = make_api_request("GET", "documents")
            
            if status_code == 200 and response.get("documents"):
                documents = response["documents"]
                
                # Create a selectbox for document selection
                doc_options = {}
                for doc in documents:
                    doc_id_key = doc.get("id")
                    filename = doc.get("filename", "Unknown")
                    status = doc.get("status", "unknown")
                    doc_options[f"{filename} ({status})"] = doc_id_key
                
                if doc_options:
                    selected_doc = st.selectbox(
                        "Select a document to view its summary:",
                        options=list(doc_options.keys())
                    )
                    
                    if st.button("View Summary", use_container_width=True):
                        selected_doc_id = doc_options[selected_doc]
                        st.session_state["last_doc_id"] = selected_doc_id
                        st.experimental_set_query_params(doc_id=selected_doc_id)
                        st.experimental_rerun()
            else:
                st.info("No documents found. Please upload a document first.")
        return
    
    # Display document summary
    with st.spinner("Fetching document summary..."):
        # Try to get the document details first
        status_code, doc_data = make_api_request("GET", f"documents/{doc_id}")
        
        if status_code == 200:
            summary = doc_data.get("summary", "No summary available.")
            filename = doc_data.get("filename", "Unknown Document")
            
            st.markdown(f"### Summary for: {filename}")
            
            if summary and summary != "No summary available.":
                st.markdown(summary)
                
                # Add regenerate summary button
                if st.button("🤖 Regenerate Summary with AI", use_container_width=True):
                    with st.spinner("Generating new summary..."):
                        regen_status, regen_response = make_api_request("POST", f"documents/{doc_id}/regenerate-summary")
                        
                        if regen_status == 200:
                            new_summary = regen_response.get("summary", "")
                            st.success("✅ Summary regenerated successfully!")
                            st.markdown("### New Summary:")
                            st.markdown(new_summary)
                        else:
                            st.error("❌ Failed to regenerate summary")
            else:
                st.warning("No summary available for this document.")
                
                if st.button("🤖 Generate Summary", use_container_width=True):
                    with st.spinner("Generating summary..."):
                        regen_status, regen_response = make_api_request("POST", f"documents/{doc_id}/regenerate-summary")
                        
                        if regen_status == 200:
                            new_summary = regen_response.get("summary", "")
                            st.success("✅ Summary generated successfully!")
                            st.markdown("### Generated Summary:")
                            st.markdown(new_summary)
                        else:
                            st.error("❌ Failed to generate summary")
        else:
            st.error(f"Failed to load document: {doc_data.get('detail', 'Unknown error') if isinstance(doc_data, dict) else 'Unknown error'}")
            
            # Add a button to go back to status
            if st.button("Back to Status"):
                st.experimental_set_query_params(page="status", doc_id=doc_id)
                st.experimental_rerun()

# Page: Dashboard
def dashboard_page():
    """Display a dashboard with all processed documents."""
    st.title("📊 Document Dashboard")
    
    # Get all documents
    with st.spinner("Loading documents..."):
        status_code, response = make_api_request("GET", "/documents")
        
        if status_code != 200:
            st.error(f"Failed to load documents: {response.get('detail', 'Unknown error')}")
            return
        
        documents = response.get("documents", [])
        total = response.get("total", 0)
        
        if not documents:
            st.info("No documents found. Upload a document to get started!")
            return
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", total)
        
        with col2:
            processed_count = len([d for d in documents if d.get("status") in ["processed", "completed", "summarized"]])
            st.metric("Processed", processed_count)
        
        with col3:
            failed_count = len([d for d in documents if d.get("status") == "failed"])
            st.metric("Failed", failed_count)
        
        with col4:
            processing_count = len([d for d in documents if d.get("status") in ["uploaded", "processing"]])
            st.metric("Processing", processing_count)
        
        # Filter options
        st.subheader("📋 Document List")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            status_filter = st.selectbox(
                "Filter by status",
                ["All", "processed", "completed", "summarized", "failed", "processing", "uploaded"],
                index=0
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                ["created_at", "filename", "status"],
                index=0
            )
        
        # Filter documents
        filtered_docs = documents
        if status_filter != "All":
            filtered_docs = [d for d in documents if d.get("status") == status_filter]
        
        # Sort documents
        if sort_by == "created_at":
            filtered_docs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        elif sort_by == "filename":
            filtered_docs.sort(key=lambda x: x.get("filename", ""))
        elif sort_by == "status":
            filtered_docs.sort(key=lambda x: x.get("status", ""))
        
        # Display documents in a table
        if filtered_docs:
            for doc in filtered_docs:
                with st.expander(f"📄 {doc.get('filename', 'Unknown')} - {doc.get('status', 'unknown').title()}", expanded=False):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**ID:** {doc.get('id', 'N/A')}")
                        st.write(f"**Created:** {doc.get('created_at', 'N/A')}")
                        if doc.get('summary'):
                            st.write(f"**Summary:** {doc.get('summary', 'N/A')}")
                    
                    with col2:
                        # Status badge
                        status = doc.get('status', 'unknown')
                        if status in ['processed', 'completed', 'summarized']:
                            st.success(f"✅ {status.title()}")
                        elif status == 'failed':
                            st.error(f"❌ {status.title()}")
                        elif status in ['uploaded', 'processing']:
                            st.warning(f"⏳ {status.title()}")
                        else:
                            st.info(f"ℹ️ {status.title()}")
                        
                        # Entity count if available
                        if 'extra_data' in doc and doc['extra_data'] and 'all_entities' in doc['extra_data']:
                            entity_count = len(doc['extra_data']['all_entities'])
                            st.metric("Entities", entity_count)
                    
                    with col3:
                        # Action buttons
                        doc_id = doc.get('id')
                        if doc_id:
                            if st.button("👁️ View", key=f"view_{doc_id}"):
                                st.session_state.current_document = {
                                    "id": doc_id,
                                    "filename": doc.get('filename', 'Unknown'),
                                    "status": doc.get('status', 'unknown'),
                                    "sections": doc.get('extra_data', {}).get('sections', []),
                                    "entities": doc.get('extra_data', {}).get('all_entities', [])
                                }
                                st.session_state.current_page = "view"
                                st.experimental_set_query_params(page="view")
                                st.experimental_rerun()
                            
                            if st.button("🔄 Review & Approve", key=f"feedback_{doc_id}"):
                                st.session_state.current_document = {
                                    "id": doc_id,
                                    "filename": doc.get('filename', 'Unknown'),
                                    "status": doc.get('status', 'unknown'),
                                    "sections": doc.get('extra_data', {}).get('sections', []),
                                    "entities": doc.get('extra_data', {}).get('all_entities', [])
                                }
                                st.session_state.current_page = "feedback"
                                st.experimental_set_query_params(page="feedback")
                                st.experimental_rerun()
        else:
            st.info(f"No documents found with status: {status_filter}")

# Page: View Document
def view_page():
    """View a specific document."""
    if 'current_document' not in st.session_state or not st.session_state.current_document:
        st.warning("No document selected. Please go to the dashboard to select a document.")
        return
    
    doc = st.session_state.current_document
    st.title(f"📄 {doc.get('filename', 'Unknown Document')}")
    
    # Document info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", doc.get('status', 'unknown').title())
    with col2:
        st.metric("Sections", len(doc.get('sections', [])))
    with col3:
        st.metric("Entities", len(doc.get('entities', [])))
    
    # Display sections
    st.subheader("📑 Document Sections")
    sections = doc.get('sections', [])
    
    if not sections:
        st.info("No sections found in this document.")
    else:
        for section in sections:
            with st.expander(f"{section.get('title', 'Untitled Section')}", expanded=False):
                st.markdown(f"**Page {section.get('page_number', 'N/A')}**")
                st.caption(f"Type: {section.get('type', 'text').title()}")
                st.write(section.get("content", "No content"))
                
                # Show section entities if available
                section_entities = section.get('entities', [])
                if section_entities:
                    st.markdown("**Entities in this section:**")
                    for entity in section_entities:
                        entity_type = entity.get("type", entity.get("entity_type", "unknown"))
                        st.caption(f"• {entity['text']} ({entity_type})")

# Page: View Entities
def entities_page():
    """View entities for a specific document."""
    if 'current_document' not in st.session_state or not st.session_state.current_document:
        st.warning("No document selected. Please go to the dashboard to select a document.")
        return
    
    doc = st.session_state.current_document
    st.title(f"🔍 Entities - {doc.get('filename', 'Unknown Document')}")
    
    entities = doc.get('entities', [])
    if not entities:
        st.info("No entities were extracted from this document.")
        return
    
    # Entity type distribution
    st.subheader("📊 Entity Type Distribution")
    entity_counts = {}
    for entity in entities:
        entity_type = entity.get("type", entity.get("entity_type", "other"))
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    if entity_counts:
        import plotly.express as px
        
        # Create pie chart
        fig = px.pie(
            names=list(entity_counts.keys()),
            values=list(entity_counts.values()),
            title="Entity Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Entity list with filters
        st.subheader("🔍 Entity Explorer")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            selected_type = st.selectbox(
                "Filter by type",
                ["All"] + sorted(list(entity_counts.keys()))
            )
        with col2:
            min_confidence = st.slider(
                "Minimum confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                format="%.2f"
            )
        
        # Filter entities
        filtered_entities = [
            e for e in entities 
            if (selected_type == "All" or e.get("type", e.get("entity_type")) == selected_type)
            and e.get("confidence", 1.0) >= min_confidence
        ]
        
        # Display filtered entities
        for entity in filtered_entities:
            entity_type = entity.get("type", entity.get("entity_type", "unknown"))
            with st.expander(f"{entity['text']} ({entity_type})", expanded=False):
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Type", entity_type.title())
                with cols[1]:
                    st.metric("Confidence", f"{entity.get('confidence', 1.0):.0%}")
                with cols[2]:
                    if "start" in entity and "end" in entity:
                        st.metric("Position", f"{entity['start']}-{entity['end']}")
                
                # Show entity context if available
                if "context" in entity:
                    st.caption("Context:")
                    st.write(f"...{entity['context']}...")
                
                # Show metadata if available
                if "metadata" in entity and entity["metadata"]:
                    with st.expander("Metadata"):
                        st.json(entity["metadata"])

# Page: Feedback & Approval
def feedback_page():
    """Human-in-the-loop feedback page for entities and summaries."""
    if 'current_document' not in st.session_state or not st.session_state.current_document:
        st.warning("No document selected. Please go to the dashboard to select a document.")
        return
    
    doc = st.session_state.current_document
    st.title(f"🔄 Feedback & Approval - {doc.get('filename', 'Unknown Document')}")
    
    # Get document data
    doc_id = doc.get('id')
    
    # Show feedback status
    if st.session_state.get(f"feedback_submitted_{doc_id}", False):
        st.success("✅ Feedback has been submitted for this document!")
        submit_time = st.session_state.get(f"feedback_timestamp_{doc_id}", "Unknown")
        st.info(f"📅 Last submitted: {submit_time}")
    elif st.session_state.get(f"feedback_modified_{doc_id}"):
        st.warning("⚠️ You have unsaved changes. Remember to submit your feedback!")
        modified_time = st.session_state.get(f"feedback_modified_{doc_id}", "Unknown")
        st.info(f"📝 Last modified: {modified_time}")
    else:
        st.info("📝 Start reviewing and editing the extracted entities and summary below.")
    
    if not doc_id:
        st.error("No document ID found.")
        return
    
    # Fetch full document data
    status_code, doc_data = make_api_request("GET", f"documents/{doc_id}")
    if status_code != 200:
        st.error(f"Failed to load document: {doc_data.get('detail', 'Unknown error')}")
        return
    
    # Extract entities and summary
    entities = doc_data.get("entities", [])
    summary = doc_data.get("summary", "No summary available.")
    sections = doc_data.get("sections", [])
    
    # Try to get summary from extra_data if main summary is empty
    if not summary or summary == "No summary available.":
        # Try to get summary from extra_data
        if "extra_data" in doc_data and doc_data["extra_data"]:
            alt_summary = doc_data["extra_data"].get("summary")
            if alt_summary:
                summary = alt_summary
    
    # Display the feedback workflow
    try:
        results = display_feedback_workflow(doc_id, entities, summary, sections)
        
        # Store results in session state for potential backend submission
        if results:
            st.session_state[f"feedback_results_{doc_id}"] = results
            # Also store a timestamp for when feedback was last modified
            st.session_state[f"feedback_modified_{doc_id}"] = datetime.now().isoformat()
        
    except Exception as e:
        st.error(f"Error in feedback workflow: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

# Page: QA
def qa_page():
    """Ask questions about documents."""
    st.title("❓ Ask a Question")
    
    # Get document ID from URL or session
    doc_id = (st.session_state.get("last_doc_id") or
              (st.session_state.get("current_document", {}).get("id") if st.session_state.get("current_document") else None))
    
    # If no document ID, show document selection
    if not doc_id:
        st.warning("No document selected. Please select a document to ask questions about.")
        
        # Show available documents for selection
        st.markdown("### Available Documents")
        with st.spinner("Loading documents..."):
            status_code, response = make_api_request("GET", "documents")
            
            if status_code == 200 and response.get("documents"):
                documents = response["documents"]
                
                # Create a selectbox for document selection
                doc_options = {}
                for doc in documents:
                    doc_id_key = doc.get("id")
                    filename = doc.get("filename", "Unknown")
                    status = doc.get("status", "unknown")
                    doc_options[f"{filename} ({status})"] = doc_id_key
                
                if doc_options:
                    selected_doc = st.selectbox(
                        "Select a document to ask questions about:",
                        options=list(doc_options.keys())
                    )
                    
                    if st.button("Select Document", use_container_width=True):
                        selected_doc_id = doc_options[selected_doc]
                        st.session_state["last_doc_id"] = selected_doc_id
                        st.experimental_rerun()
            else:
                st.info("No documents found. Please upload a document first.")
        return
    
    # Show selected document info
    with st.spinner("Loading document info..."):
        status_code, doc_data = make_api_request("GET", f"documents/{doc_id}")
        if status_code == 200:
            st.info(f"📄 Selected document: {doc_data.get('filename', 'Unknown')}")
        else:
            st.error("Failed to load document information.")
            return
    
    # Question form
    question = st.text_area("Enter your question:", "")
    
    if st.button("Submit Question"):
        if not question.strip():
            st.warning("Please enter a question.")
            return
            
        if not doc_id:
            st.error("No document selected.")
            return
            
        with st.spinner("Generating answer..."):
            # Use the correct API endpoint with document ID
            data = {
                "question": question,
                "context": {"include_evidence": True}
            }
            
            status_code, response = make_api_request(
                "POST", f"documents/{doc_id}/qa", json=data
            )
            
            if status_code == 200:
                st.markdown("### Answer")
                
                # Extract answers from response
                answers = response.get("answers", {})
                if answers:
                    for agent_name, answer_data in answers.items():
                        if isinstance(answer_data, dict) and "answer" in answer_data:
                            st.markdown(f"**{agent_name.replace('_', ' ').title()}:**")
                            st.write(answer_data["answer"])
                            
                            # Show confidence if available
                            if "confidence" in answer_data:
                                st.caption(f"Confidence: {answer_data['confidence']}")
                            
                            # Show evidence if available
                            if "evidence" in answer_data and answer_data["evidence"]:
                                with st.expander("Evidence"):
                                    for evidence in answer_data["evidence"]:
                                        st.write(f"- {evidence}")
                            
                            st.markdown("---")
                else:
                    st.info("No answers were generated.")
                
                # Show workflow info
                if "workflow_id" in response:
                    st.caption(f"Workflow ID: {response['workflow_id']}")
                
            else:
                error = response.get("detail", response.get("error", "Failed to get an answer"))
                st.error(f"Error: {error}")
    
    # Add button to select a different document
    if st.button("Select Different Document"):
        st.session_state["last_doc_id"] = None
        st.experimental_rerun()

# Main app
def main():
    """Main app with navigation."""
    # Initialize session state
    if "last_doc_id" not in st.session_state:
        st.session_state["last_doc_id"] = None
    
    if "current_document" not in st.session_state:
        st.session_state["current_document"] = None
    
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "upload"
    
    if "api_logs" not in st.session_state:
        st.session_state["api_logs"] = []
    
    # Get current page from session state or URL params, with session state taking priority
    query_params = st.experimental_get_query_params()
    url_page = query_params.get("page", ["upload"])[0].lower()
    
    # Use session state current_page if it exists, otherwise use URL param
    if "current_page" in st.session_state and st.session_state.current_page:
        current_page = st.session_state.current_page
    else:
        current_page = url_page
        st.session_state.current_page = current_page
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    pages = {
        "upload": "📤 Upload",
        "dashboard": "📊 Dashboard",
        "status": "📊 Status",
        "summary": "📝 Summary",
        "feedback": "🔄 Feedback & Approval",
        "qa": "❓ Ask a Question",
    }
    
    # Display navigation buttons
    for page_id, page_title in pages.items():
        if st.sidebar.button(page_title, key=f"nav_{page_id}"):
            st.session_state.current_page = page_id
            st.experimental_set_query_params(page=page_id)
            st.experimental_rerun()
    
    # Display current page
    if current_page == "upload":
        upload_page()
    elif current_page == "dashboard":
        dashboard_page()
    elif current_page == "view":
        view_page()
    elif current_page == "entities":
        entities_page()
    elif current_page == "status":
        status_page()
    elif current_page == "summary":
        summary_page()
    elif current_page == "feedback":
        feedback_page()
    elif current_page == "qa":
        qa_page()
    else:
        st.warning("Page not found. Redirecting to Upload page...")
        st.experimental_set_query_params(page="upload")
        st.experimental_rerun()

if __name__ == "__main__":
    main()
