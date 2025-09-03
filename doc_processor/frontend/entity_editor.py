"""
Enhanced Entity and Summary Editor UI Component

This module provides interactive UI components for editing and approving
extracted entities and document summaries with human-in-the-loop feedback.
"""

import streamlit as st
import json
import requests
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import pandas as pd

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class EntityEditor:
    """Interactive entity editor with approval workflow."""
    
    def __init__(self):
        self.edited_entities = {}
        self.approved_entities = set()
        self.rejected_entities = set()
        self.new_entities = []
    
    def display_entity_editor(
        self, 
        entities: List[Dict], 
        document_id: str,
        sections: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Display interactive entity editor with approval workflow.
        
        Args:
            entities: List of extracted entities
            document_id: Document ID for tracking
            sections: Document sections for context
            
        Returns:
            Dictionary with editing results
        """
        st.subheader("🔍 Entity Editor & Approval")
        st.markdown("Review, edit, and approve extracted entities. You can modify entity types, text, and confidence scores.")
        
        if not entities:
            st.info("No entities found to edit.")
            return {"approved": [], "rejected": [], "edited": []}
        
        # Initialize session state for this document
        if f"entity_editor_{document_id}" not in st.session_state:
            st.session_state[f"entity_editor_{document_id}"] = {
                "edited_entities": {},
                "approved_entities": set(),
                "rejected_entities": set(),
                "new_entities": []
            }
        
        editor_state = st.session_state[f"entity_editor_{document_id}"]
        
        # Entity statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Entities", len(entities))
        with col2:
            st.metric("Approved", len(editor_state["approved_entities"]))
        with col3:
            st.metric("Rejected", len(editor_state["rejected_entities"]))
        with col4:
            st.metric("Edited", len(editor_state["edited_entities"]))
        
        # Filter and search options
        st.markdown("### Filter & Search")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Get unique entity types from the entities
            entity_types = list(set(e.get("type", e.get("entity_type", "unknown")) for e in entities if e.get("type") or e.get("entity_type")))
            entity_types = [t for t in entity_types if t and t != "unknown"]  # Filter out None and unknown
            selected_type = st.selectbox("Filter by Type", ["All"] + sorted(entity_types))
        
        with col2:
            min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)
        
        with col3:
            search_term = st.text_input("Search Entities", "")
        
        # Filter entities
        filtered_entities = self._filter_entities(entities, selected_type, min_confidence, search_term)
        
        # Bulk actions
        st.markdown("### Bulk Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Approve All Visible", use_container_width=True):
                approved_count = 0
                for idx, entity in enumerate(filtered_entities):
                    entity_id = self._get_entity_id(entity, idx)
                    editor_state["approved_entities"].add(entity_id)
                    if entity_id in editor_state["rejected_entities"]:
                        editor_state["rejected_entities"].remove(entity_id)
                    approved_count += 1
                st.success(f"Approved {approved_count} entities")
                st.experimental_rerun()
        
        with col2:
            if st.button("❌ Reject All Visible", use_container_width=True):
                rejected_count = 0
                for idx, entity in enumerate(filtered_entities):
                    entity_id = self._get_entity_id(entity, idx)
                    editor_state["rejected_entities"].add(entity_id)
                    if entity_id in editor_state["approved_entities"]:
                        editor_state["approved_entities"].remove(entity_id)
                    rejected_count += 1
                st.success(f"Rejected {rejected_count} entities")
                st.experimental_rerun()
        
        with col3:
            if st.button("🔄 Reset All", use_container_width=True):
                editor_state["approved_entities"].clear()
                editor_state["rejected_entities"].clear()
                editor_state["edited_entities"].clear()
                st.success("Reset all approvals")
                st.experimental_rerun()
        
        # Entity editing interface
        st.markdown("### Entity Review & Editing")
        
        for i, entity in enumerate(filtered_entities):
            entity_id = self._get_entity_id(entity, i)
            is_approved = entity_id in editor_state["approved_entities"]
            is_rejected = entity_id in editor_state["rejected_entities"]
            is_edited = entity_id in editor_state["edited_entities"]
            
            # Status indicator with proper emojis
            if is_approved:
                status_icon = "✅"
                status_text = "Approved"
                status_color = "success"
            elif is_rejected:
                status_icon = "❌"
                status_text = "Rejected"
                status_color = "error"
            else:
                status_icon = "⏳"
                status_text = "Pending"
                status_color = "warning"
            
            with st.expander(f"{status_icon} {entity.get('text', 'Unknown')} ({entity.get('type', entity.get('entity_type', 'unknown'))}) - {status_text}", expanded=is_edited):
                # Get current entity data (edited or original)
                current_entity = editor_state["edited_entities"].get(entity_id, entity.copy())
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Editable fields
                    new_text = st.text_input(
                        "Entity Text", 
                        value=current_entity.get("text", ""),
                        key=f"text_{entity_id}"
                    )
                    
                    entity_types = ["PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY", "PERCENT", 
                                  "PRODUCT", "EVENT", "NORP", "FAC", "LOC", "LOCATION", 
                                  "WORK_OF_ART", "LAW", "LANGUAGE", "QUANTITY", "ORDINAL", "CARDINAL"]
                    
                    current_type = current_entity.get("type", current_entity.get("entity_type", "PERSON"))
                    new_type = st.selectbox(
                        "Entity Type",
                        entity_types,
                        index=entity_types.index(current_type) if current_type in entity_types else 0,
                        key=f"type_{entity_id}"
                    )
                    
                    new_confidence = st.slider(
                        "Confidence",
                        0.0, 1.0, 
                        current_entity.get("confidence", 1.0),
                        0.05,
                        key=f"confidence_{entity_id}"
                    )
                
                with col2:
                    # Context and metadata
                    if "context" in entity:
                        st.markdown("**Context:**")
                        st.caption(entity["context"])
                    
                    if "metadata" in entity and entity["metadata"]:
                        st.markdown("**Metadata:**")
                        st.json(entity["metadata"])
                    
                    # Position information
                    if "start" in entity and "end" in entity:
                        st.markdown(f"**Position:** {entity['start']}-{entity['end']}")
                
                # Action buttons
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("💾 Save Changes", key=f"save_{entity_id}"):
                        # Save edited entity
                        edited_entity = {
                            "text": new_text,
                            "type": new_type,
                            "confidence": new_confidence,
                            "start": current_entity.get("start", entity.get("start", 0)),
                            "end": current_entity.get("end", entity.get("end", 0)),
                            "metadata": current_entity.get("metadata", entity.get("metadata", {}))
                        }
                        editor_state["edited_entities"][entity_id] = edited_entity
                        st.success("Changes saved!")
                        st.experimental_rerun()
                
                with col2:
                    if st.button("✅ Approve", key=f"approve_{entity_id}"):
                        editor_state["approved_entities"].add(entity_id)
                        if entity_id in editor_state["rejected_entities"]:
                            editor_state["rejected_entities"].remove(entity_id)
                        st.success("Entity approved!")
                        st.experimental_rerun()
                
                with col3:
                    if st.button("❌ Reject", key=f"reject_{entity_id}"):
                        editor_state["rejected_entities"].add(entity_id)
                        if entity_id in editor_state["approved_entities"]:
                            editor_state["approved_entities"].remove(entity_id)
                        st.success("Entity rejected!")
                        st.experimental_rerun()
                
                with col4:
                    if st.button("🗑️ Delete", key=f"delete_{entity_id}"):
                        editor_state["rejected_entities"].add(entity_id)
                        if entity_id in editor_state["approved_entities"]:
                            editor_state["approved_entities"].remove(entity_id)
                        st.success("Entity marked for deletion!")
                        st.experimental_rerun()
        
        # Add new entity section
        st.markdown("### Add New Entity")
        with st.expander("➕ Add Custom Entity", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                new_entity_text = st.text_input("Entity Text", key="new_entity_text")
                # Use the same entity types as the filter
                available_types = ["PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY", "PERCENT", 
                                  "PRODUCT", "EVENT", "NORP", "FAC", "LOC", "LOCATION", 
                                  "WORK_OF_ART", "LAW", "LANGUAGE", "QUANTITY", "ORDINAL", "CARDINAL"]
                new_entity_type = st.selectbox("Entity Type", available_types, key="new_entity_type")
            
            with col2:
                new_entity_confidence = st.slider("Confidence", 0.0, 1.0, 0.8, 0.05, key="new_entity_confidence")
                new_entity_start = st.number_input("Start Position", 0, 10000, 0, key="new_entity_start")
                new_entity_end = st.number_input("End Position", 0, 10000, 0, key="new_entity_end")
            
            if st.button("Add Entity", key="add_new_entity"):
                if new_entity_text and new_entity_text.strip():
                    if new_entity_start < new_entity_end:
                        new_entity = {
                            "text": new_entity_text.strip(),
                            "type": new_entity_type,
                            "confidence": new_entity_confidence,
                            "start": new_entity_start,
                            "end": new_entity_end,
                            "metadata": {"added_by_user": True}
                        }
                        editor_state["new_entities"].append(new_entity)
                        st.success("New entity added!")
                        st.experimental_rerun()
                    else:
                        st.error("End position must be greater than start position")
                else:
                    st.error("Please provide valid entity text")
        
        # Summary and export
        st.markdown("### Summary & Export")
        
        approved_count = len(editor_state["approved_entities"])
        rejected_count = len(editor_state["rejected_entities"])
        edited_count = len(editor_state["edited_entities"])
        new_count = len(editor_state["new_entities"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Approved Entities", approved_count)
            st.metric("Rejected Entities", rejected_count)
        
        with col2:
            st.metric("Edited Entities", edited_count)
            st.metric("New Entities", new_count)
        
        # Export results
        if st.button("📤 Export Results", use_container_width=True):
            results = self._export_results(entities, editor_state)
            
            # Create downloadable JSON
            json_str = json.dumps(results, indent=2)
            st.download_button(
                label="Download Entity Results (JSON)",
                data=json_str,
                file_name=f"entity_results_{document_id}.json",
                mime="application/json"
            )
            
            # Create downloadable CSV
            df = self._create_entity_dataframe(results)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Entity Results (CSV)",
                data=csv,
                file_name=f"entity_results_{document_id}.csv",
                mime="text/csv"
            )
        
        return {
            "approved": list(editor_state["approved_entities"]),
            "rejected": list(editor_state["rejected_entities"]),
            "edited": editor_state["edited_entities"],
            "new": editor_state["new_entities"]
        }
    
    def _filter_entities(self, entities: List[Dict], entity_type: str, min_confidence: float, search_term: str) -> List[Dict]:
        """Filter entities based on criteria."""
        filtered = entities
        
        # Filter by entity type
        if entity_type != "All":
            filtered = [e for e in filtered if e.get("type", e.get("entity_type", "")) == entity_type]
        
        # Filter by confidence
        filtered = [e for e in filtered if e.get("confidence", 1.0) >= min_confidence]
        
        # Filter by search term
        if search_term and search_term.strip():
            search_lower = search_term.lower().strip()
            filtered = [e for e in filtered if search_lower in e.get("text", "").lower()]
        
        return filtered
    
    def _get_entity_id(self, entity: Dict, index: int = 0) -> str:
        """Generate a unique ID for an entity."""
        # Include entity type and use a hash to ensure uniqueness even for identical text/positions
        import hashlib
        base_string = f"{entity.get('text', '')}_{entity.get('start', 0)}_{entity.get('end', 0)}_{entity.get('type', entity.get('entity_type', ''))}"
        # Add a hash of the full entity dict to handle edge cases
        entity_hash = hashlib.md5(str(sorted(entity.items())).encode()).hexdigest()[:8]
        return f"{base_string}_{entity_hash}_{index}"
    
    def _export_results(self, original_entities: List[Dict], editor_state: Dict) -> Dict[str, Any]:
        """Export editing results."""
        results = {
            "document_id": st.session_state.get("current_document", {}).get("id", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_entities": len(original_entities),
                "approved": len(editor_state["approved_entities"]),
                "rejected": len(editor_state["rejected_entities"]),
                "edited": len(editor_state["edited_entities"]),
                "new": len(editor_state["new_entities"])
            },
            "approved_entities": [],
            "rejected_entities": [],
            "edited_entities": editor_state["edited_entities"],
            "new_entities": editor_state["new_entities"]
        }
        
        # Add approved and rejected entities
        for idx, entity in enumerate(original_entities):
            entity_id = self._get_entity_id(entity, idx)
            if entity_id in editor_state["approved_entities"]:
                # Use edited version if available
                final_entity = editor_state["edited_entities"].get(entity_id, entity)
                results["approved_entities"].append(final_entity)
            elif entity_id in editor_state["rejected_entities"]:
                results["rejected_entities"].append(entity)
        
        return results
    
    def _create_entity_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create a pandas DataFrame from entity results."""
        data = []
        
        for entity in results["approved_entities"]:
            data.append({
                "Status": "Approved",
                "Text": entity.get("text", ""),
                "Type": entity.get("type", entity.get("entity_type", "")),
                "Confidence": entity.get("confidence", 1.0),
                "Start": entity.get("start", 0),
                "End": entity.get("end", 0)
            })
        
        for entity in results["rejected_entities"]:
            data.append({
                "Status": "Rejected",
                "Text": entity.get("text", ""),
                "Type": entity.get("type", entity.get("entity_type", "")),
                "Confidence": entity.get("confidence", 1.0),
                "Start": entity.get("start", 0),
                "End": entity.get("end", 0)
            })
        
        for entity in results["new_entities"]:
            data.append({
                "Status": "New",
                "Text": entity.get("text", ""),
                "Type": entity.get("type", ""),
                "Confidence": entity.get("confidence", 1.0),
                "Start": entity.get("start", 0),
                "End": entity.get("end", 0)
            })
        
        return pd.DataFrame(data)


class SummaryEditor:
    """Interactive summary editor with approval workflow."""
    
    def __init__(self):
        self.edited_summary = None
        self.is_approved = False
    
    def display_summary_editor(
        self, 
        summary: str, 
        document_id: str,
        sections: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Display interactive summary editor.
        
        Args:
            summary: Original document summary
            document_id: Document ID for tracking
            sections: Document sections for context
            
        Returns:
            Dictionary with editing results
        """
        st.subheader("📝 Summary Editor & Approval")
        st.markdown("Review and edit the document summary. You can modify the content and approve it for final use.")
        
        # Debug info
        st.success("✅ Summary Editor is loading correctly!")
        st.write(f"🔍 **Summary Editor Debug**: Document ID: {document_id}")
        st.write(f"🔍 **Summary content**: {summary[:100] + '...' if len(summary) > 100 else summary}")
        
        # Session state debug
        if f"summary_editor_{document_id}" in st.session_state:
            editor_state_debug = st.session_state[f"summary_editor_{document_id}"]
            st.write(f"🔍 **Session State**: Editor initialized = True")
            st.write(f"🔍 **Current edited_summary length**: {len(editor_state_debug.get('edited_summary', ''))}")
        else:
            st.write(f"🔍 **Session State**: Editor not yet initialized")
        
        # Check if summary is empty or placeholder
        if not summary or summary == "No summary available." or len(summary.strip()) == 0:
            st.warning("⚠️ No summary available for this document. You can create one manually or the document may need to be reprocessed.")
            summary = "No summary available. Please create a summary for this document."
            
            # Offer to generate summary from sections if available
            if sections and len(sections) > 0:
                if st.button("🤖 Generate Summary from Sections", key=f"generate_summary_{document_id}"):
                    # Create a basic summary from section titles and content
                    generated_summary = "Document Summary:\n\n"
                    for i, section in enumerate(sections[:5]):  # Limit to first 5 sections
                        title = section.get("title", f"Section {i+1}")
                        content = section.get("content", "")
                        # Take first 100 characters of content
                        content_preview = content[:100] + "..." if len(content) > 100 else content
                        generated_summary += f"• {title}: {content_preview}\n"
                    
                    summary = generated_summary
                    st.success("✅ Generated basic summary from document sections!")
                    st.experimental_rerun()
        
        # Initialize session state for this document
        if f"summary_editor_{document_id}" not in st.session_state:
            st.session_state[f"summary_editor_{document_id}"] = {
                "edited_summary": summary,
                "is_approved": False,
                "original_summary": summary
            }
        
        editor_state = st.session_state[f"summary_editor_{document_id}"]
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Length", len(summary))
        with col2:
            st.metric("Current Length", len(editor_state["edited_summary"]))
        with col3:
            status = "✅ Approved" if editor_state["is_approved"] else "⏳ Pending"
            st.metric("Status", status)
        
        # Summary editing interface
        st.markdown("### Edit Summary")
        
        # Show original summary for reference
        with st.expander("📄 Original Summary", expanded=True if len(summary) < 100 else False):
            st.text_area("Original", value=summary, height=150, disabled=True)
            
        # Quick summary templates
        st.markdown("#### Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🤖 Generate with AI", key=f"ai_summary_{document_id}"):
                with st.spinner("Generating AI summary..."):
                    ai_summary = self._generate_ai_summary(document_id, sections)
                    if ai_summary:
                        st.session_state[f"summary_editor_{document_id}"]["edited_summary"] = ai_summary
                        st.success("✅ AI summary generated!")
                        st.experimental_rerun()
                    else:
                        st.error("❌ Failed to generate AI summary")
        with col2:
            if st.button("📝 Create Template", key=f"new_summary_{document_id}"):
                template = "This document discusses:\n\n• Key Point 1: \n• Key Point 2: \n• Key Point 3: \n\nConclusion: "
                st.session_state[f"summary_editor_{document_id}"]["edited_summary"] = template
                st.experimental_rerun()
        with col3:
            if st.button("🔄 Reset to Original", key=f"reset_original_{document_id}"):
                st.session_state[f"summary_editor_{document_id}"]["edited_summary"] = summary
                st.experimental_rerun()
        with col4:
            if st.button("✨ Enhance Summary", key=f"enhance_summary_{document_id}"):
                current = st.session_state[f"summary_editor_{document_id}"]["edited_summary"]
                enhanced = f"Enhanced Summary:\n\n{current}\n\n[Add more details, context, or analysis here]"
                st.session_state[f"summary_editor_{document_id}"]["edited_summary"] = enhanced
                st.experimental_rerun()
        
        # Editable summary
        st.markdown("### ✏️ Edit Summary")
        st.info("💡 **Tip**: Click in the text area below to start editing the summary. Your changes will be saved when you click 'Save Changes'.")
        
        edited_summary = st.text_area(
            "Summary Content",
            value=editor_state["edited_summary"],
            height=200,
            key=f"summary_edit_{document_id}",
            help="Modify the summary text as needed. You can add, remove, or rephrase content.",
            placeholder="Enter your summary here..."
        )
        
        # Update session state immediately when text changes
        if edited_summary != editor_state["edited_summary"]:
            st.session_state[f"summary_editor_{document_id}"]["edited_summary"] = edited_summary
        
        # Alternative editing method if text area is disabled
        st.markdown("---")
        st.markdown("#### 🔧 Alternative Editing Method")
        st.info("If the text area above appears disabled, use this alternative input method:")
        
        alternative_summary = st.text_input(
            "Enter new summary (alternative method)",
            value="",
            key=f"alt_summary_{document_id}",
            help="Type your summary here if the main text area is not working"
        )
        
        if alternative_summary and st.button("📝 Use This Summary", key=f"use_alt_{document_id}"):
            st.session_state[f"summary_editor_{document_id}"]["edited_summary"] = alternative_summary
            st.success("✅ Summary updated from alternative input!")
            st.experimental_rerun()
        
        # Section context (if available)
        if sections:
            st.markdown("### Document Sections (for reference)")
            section_titles = [s.get("title", f"Section {i+1}") for i, s in enumerate(sections)]
            selected_section = st.selectbox("View Section Content", ["None"] + section_titles)
            
            if selected_section != "None":
                section_idx = section_titles.index(selected_section)
                section_content = sections[section_idx].get("content", "")
                st.text_area("Section Content", value=section_content, height=100, disabled=True)
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💾 Save Changes", key=f"save_summary_{document_id}"):
                editor_state["edited_summary"] = edited_summary
                st.success("Summary saved!")
                st.experimental_rerun()
        
        with col2:
            if st.button("✅ Approve", key=f"approve_summary_{document_id}"):
                editor_state["edited_summary"] = edited_summary
                editor_state["is_approved"] = True
                st.success("Summary approved!")
                st.experimental_rerun()
        
        with col3:
            if st.button("🔄 Reset", key=f"reset_approval_{document_id}"):
                editor_state["edited_summary"] = editor_state["original_summary"]
                editor_state["is_approved"] = False
                st.success("Summary reset to original!")
                st.experimental_rerun()
        
        with col4:
            if st.button("📤 Export", key=f"export_summary_{document_id}"):
                self._export_summary(editor_state, document_id)
        
        # Summary preview
        st.markdown("### Summary Preview")
        st.markdown(editor_state["edited_summary"])
        
        return {
            "original_summary": editor_state["original_summary"],
            "edited_summary": editor_state["edited_summary"],
            "is_approved": editor_state["is_approved"]
        }
    
    def _generate_ai_summary(self, document_id: str, sections: List[Dict] = None) -> Optional[str]:
        """Generate AI summary by calling the backend API."""
        try:
            import requests
            
            # Get API base URL (you might need to adjust this)
            API_BASE_URL = "http://localhost:8000"
            
            # Prepare the request payload
            payload = {
                "document_id": document_id,
                "sections": sections or [],
                "regenerate": True
            }
            
            # Call the backend API to regenerate summary
            response = requests.post(
                f"{API_BASE_URL}/documents/{document_id}/regenerate-summary",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("summary", "")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Error generating AI summary: {str(e)}")
            return None
    
    def _export_summary(self, editor_state: Dict, document_id: str):
        """Export summary results."""
        results = {
            "document_id": document_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_summary": editor_state["original_summary"],
            "edited_summary": editor_state["edited_summary"],
            "is_approved": editor_state["is_approved"],
            "changes_made": editor_state["edited_summary"] != editor_state["original_summary"]
        }
        
        json_str = json.dumps(results, indent=2)
        st.download_button(
            label="Download Summary Results (JSON)",
            data=json_str,
            file_name=f"summary_results_{document_id}.json",
            mime="application/json"
        )


def display_feedback_workflow(document_id: str, entities: List[Dict], summary: str, sections: List[Dict] = None):
    """
    Display the complete feedback workflow for entities and summary.
    
    Args:
        document_id: Document ID
        entities: List of extracted entities
        summary: Document summary
        sections: Document sections for context
    """
    st.title("🔄 Human-in-the-Loop Feedback")
    st.markdown("Review and approve extracted entities and document summary. Your feedback will improve the system's accuracy.")
    
    # Show document summary overview at the top
    st.markdown("### 📄 Document Overview")
    col1, col2 = st.columns([2, 1])
    with col1:
        if summary and summary != "No summary available." and len(summary.strip()) > 0:
            st.info(f"**Summary Preview:** {summary[:200] + '...' if len(summary) > 200 else summary}")
        else:
            st.warning("**No summary available** - You can create one in the Summary Editor tab")
    with col2:
        st.metric("Entities Found", len(entities))
        st.metric("Sections Found", len(sections) if sections else 0)
    
    # Check for existing feedback
    existing_feedback = load_existing_feedback(document_id)
    if existing_feedback:
        st.info("📋 Previous feedback found for this document. You can review and modify it.")
        with st.expander("View Previous Feedback", expanded=False):
            st.json(existing_feedback)
    
    # Debug information
    st.write("🔍 **Debug Info:**")
    st.write(f"Document ID: {document_id}")
    st.write(f"Summary length: {len(summary) if summary else 0}")
    st.write(f"Entities count: {len(entities) if entities else 0}")
    st.write(f"Sections count: {len(sections) if sections else 0}")
    
    # Create tabs for different editing modes
    tab1, tab2, tab3 = st.tabs(["🔍 Entity Editor", "📝 Summary Editor", "📊 Review & Export"])
    
    with tab1:
        entity_editor = EntityEditor()
        entity_results = entity_editor.display_entity_editor(entities, document_id, sections)
    
    with tab2:
        st.write("📝 **Summary Editor Debug:**")
        st.write(f"About to create SummaryEditor for document: {document_id}")
        summary_editor = SummaryEditor()
        summary_results = summary_editor.display_summary_editor(summary, document_id, sections)
    
    with tab3:
        display_review_and_export(document_id, entity_results, summary_results)
    
    return {
        "entities": entity_results,
        "summary": summary_results
    }


def display_review_and_export(document_id: str, entity_results: Dict, summary_results: Dict):
    """Display final review and export options."""
    st.subheader("📊 Final Review & Export")
    
    # Show current summary prominently
    st.markdown("### 📄 Current Summary")
    if summary_results and summary_results.get("edited_summary"):
        current_summary = summary_results["edited_summary"]
        st.info(f"**Final Summary:** {current_summary}")
        
        # Show if it was edited
        if summary_results.get("original_summary") != current_summary:
            st.success("✏️ Summary has been edited")
        else:
            st.info("📝 Summary unchanged from original")
    else:
        st.warning("⚠️ No summary available")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Approved Entities", len(entity_results.get("approved", [])))
    
    with col2:
        st.metric("Rejected Entities", len(entity_results.get("rejected", [])))
    
    with col3:
        st.metric("Edited Entities", len(entity_results.get("edited", {})))
    
    with col4:
        st.metric("Summary Approved", "✅" if summary_results.get("is_approved", False) else "❌")
    
    # Final approval
    st.markdown("### Final Approval")
    
    # Check if feedback has already been submitted
    feedback_submitted = st.session_state.get(f"feedback_submitted_{document_id}", False)
    
    if feedback_submitted:
        st.success("✅ Feedback already submitted!")
        submit_timestamp = st.session_state.get(f"feedback_timestamp_{document_id}", "Unknown")
        st.info(f"📅 Submitted at: {submit_timestamp}")
        
        # Option to resubmit
        if st.button("🔄 Resubmit Feedback", use_container_width=True):
            submit_feedback_to_backend(document_id, entity_results, summary_results)
    else:
        if st.button("✅ Submit Feedback to Backend", use_container_width=True, type="primary"):
            # Submit feedback to backend API
            submit_feedback_to_backend(document_id, entity_results, summary_results)
    
    # Export all results
    st.markdown("### Export All Results")
    
    if st.button("📤 Export Complete Results", use_container_width=True):
        complete_results = {
            "document_id": document_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "entities": entity_results,
            "summary": summary_results,
            "workflow_completed": True
        }
        
        json_str = json.dumps(complete_results, indent=2)
        st.download_button(
            label="Download Complete Results (JSON)",
            data=json_str,
            file_name=f"complete_feedback_{document_id}.json",
            mime="application/json"
        )


def submit_feedback_to_backend(document_id: str, entity_results: Dict, summary_results: Dict):
    """Submit feedback to the backend API."""
    try:
        # Prepare feedback data
        feedback_data = {
            "document_id": document_id,
            "entities": entity_results,
            "summary": summary_results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Submit to backend
        url = f"{API_BASE_URL}/documents/{document_id}/feedback"
        response = requests.post(
            url,
            json=feedback_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Feedback submitted successfully! Feedback ID: {result.get('feedback_id', 'N/A')}")
            
            # Mark feedback as submitted in session state to prevent re-submission
            st.session_state[f"feedback_submitted_{document_id}"] = True
            st.session_state[f"feedback_timestamp_{document_id}"] = datetime.now().isoformat()
            
            # Show what was submitted
            with st.expander("Submitted Feedback Summary", expanded=True):
                st.markdown(f"**Entities:** {len(entity_results.get('approved', []))} approved, {len(entity_results.get('rejected', []))} rejected")
                st.markdown(f"**Summary:** {'Approved' if summary_results.get('is_approved', False) else 'Pending approval'}")
                st.markdown(f"**Edits:** {len(entity_results.get('edited', {}))} entities modified")
                st.markdown(f"**New Entities:** {len(entity_results.get('new', []))} entities added")
                
            # Add option to continue or go to dashboard, but don't force redirect
            st.info("💡 Feedback has been saved successfully. You can continue editing or navigate to another page using the sidebar.")
        else:
            error_msg = response.json().get('detail', 'Unknown error')
            st.error(f"❌ Failed to submit feedback: {error_msg}")
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error submitting feedback: {str(e)}")
    except Exception as e:
        st.error(f"❌ Error submitting feedback: {str(e)}")


def load_existing_feedback(document_id: str) -> Optional[Dict]:
    """Load existing feedback for a document."""
    try:
        url = f"{API_BASE_URL}/documents/{document_id}/feedback"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('feedback_exists', False):
                return result.get('feedback')
        return None
        
    except requests.exceptions.RequestException:
        return None
    except Exception:
        return None
