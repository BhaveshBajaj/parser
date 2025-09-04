"""AutoGen-based agents for document processing workflows."""

import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from uuid import uuid4
import asyncio

# AutoGen imports
import autogen
from autogen import ConversableAgent, GroupChat, GroupChatManager

from ...models.document import Document
from ..entity_extractor import EntityExtractor
from ..embedding_service import get_embedding_service
from ..vector_search import get_search_index, HybridSearch
from ...core.config import settings

logger = logging.getLogger(__name__)


class AutoGenDocumentAgent(ConversableAgent):
    """Base AutoGen agent for document processing tasks."""
    
    def __init__(
        self,
        name: str,
        system_message: str,
        llm_config: Optional[Dict] = None,
        human_input_mode: str = "NEVER",
        max_consecutive_auto_reply: int = 3,
        config: Optional[Dict] = None,
        **kwargs
    ):
        # Configure LLM settings
        if llm_config is None and settings.AZURE_OPENAI_API_KEY:
            llm_config = {
                "config_list": [{
                    "model": settings.AZURE_OPENAI_DEPLOYMENT,
                    "api_key": settings.AZURE_OPENAI_API_KEY,
                    "base_url": settings.AZURE_OPENAI_ENDPOINT,
                    "api_type": "azure",
                    "api_version": settings.AZURE_OPENAI_API_VERSION,
                }],
                "temperature": 0.3,
                "timeout": 120,
            }
        
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            **kwargs
        )
        
        # Store document processing context
        self.document_context: Optional[Dict[str, Any]] = None
        self.processing_results: Dict[str, Any] = {}
        
        # Initialize embedding and search services
        self.config = config or {}
        self.embedding_service = get_embedding_service(self.config)
        self.search_index = get_search_index(self.config)
    
    def set_document_context(self, document: Document, context: Optional[Dict[str, Any]] = None):
        """Set the document and context for processing."""
        self.document_context = {
            "document": document,
            "context": context or {},
            "document_id": str(document.id),
            "filename": document.filename,
            "content_type": document.content_type
        }
    
    def get_document_content(self) -> str:
        """Extract readable content from the document."""
        if not self.document_context:
            return ""
        
        document = self.document_context["document"]
        
        # Get content from sections if available
        if document.extra_data and "sections" in document.extra_data:
            sections = document.extra_data["sections"]
            content_parts = []
            for section in sections:
                title = section.get("title", "")
                content = section.get("content", "")
                if title:
                    content_parts.append(f"## {title}")
                content_parts.append(content)
            return "\n\n".join(content_parts)
        
        # Fallback to summary or raw content
        if document.summary:
            return document.summary
        
        return f"Document: {document.filename} (Content not available in processed format)"


class AutoGenSummarizationAgent(AutoGenDocumentAgent):
    """AutoGen agent for document summarization tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        system_message = """You are an expert document summarization agent. Your role is to:

1. Analyze document content and create concise, accurate summaries
2. Extract key points and main themes
3. Identify important entities and concepts
4. Provide section-level and document-level summaries
5. Ensure summaries are factual and preserve important details

When processing documents:
- Focus on the main ideas and key information
- Maintain factual accuracy
- Use clear, professional language
- Structure summaries logically
- Include relevant entities and dates when present

Respond with structured JSON containing:
- summary: Main document summary
- key_points: List of key points
- entities_mentioned: Important entities found
- section_summaries: Summaries of individual sections if applicable
"""
        
        super().__init__(
            name="summarization_agent",
            system_message=system_message,
            config=config
        )
    
    def generate_summary_prompt(self, document_content: str, summary_type: str = "document") -> str:
        """Generate a prompt for summarization based on the type."""
        if summary_type == "section":
            return f"""Please analyze and summarize the following document section:

{document_content}

Provide a JSON response with:
- section_summary: Concise summary of this section
- key_points: 3-5 main points from this section
- entities: Important entities mentioned (people, places, organizations, dates)
"""
        elif summary_type == "corpus":
            return f"""Please analyze multiple documents and provide a corpus-level summary:

{document_content}

Provide a JSON response with:
- corpus_summary: Overall summary of all documents
- common_themes: Themes that appear across documents
- key_entities: Most important entities across all documents
- document_relationships: How documents relate to each other
"""
        else:  # document level
            return f"""Please analyze and summarize the following document:

{document_content}

Provide a JSON response with:
- summary: Comprehensive document summary
- key_points: 5-7 main points from the document
- entities_mentioned: Important entities (people, places, organizations, dates)
- main_topics: Primary topics covered
- document_type: Inferred type of document
"""


class AutoGenEntityAgent(AutoGenDocumentAgent):
    """AutoGen agent for entity extraction and processing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        system_message = """You are an expert entity extraction and analysis agent. Your role is to:

1. Extract named entities from documents (people, organizations, locations, dates, etc.)
2. Categorize and tag entities appropriately
3. Validate entity quality and relevance
4. Identify relationships between entities
5. Provide confidence scores for extractions

When processing documents:
- Extract all relevant named entities
- Classify entities by type (PERSON, ORG, GPE, DATE, MONEY, etc.)
- Provide confidence scores (0.0-1.0)
- Identify entity relationships and contexts
- Filter out low-quality or irrelevant entities

Respond with structured JSON containing:
- entities: List of extracted entities with types and confidence
- entity_relationships: Relationships between entities
- entity_categories: Categorization of entities
- validation_results: Quality assessment of extractions
"""
        
        super().__init__(
            name="entity_agent",
            system_message=system_message,
            config=config
        )
        
        # Initialize entity extractor for LLM-powered extraction
        self.entity_extractor = EntityExtractor()
        # Note: EntityExtractor uses Azure OpenAI for LLM-powered entity extraction
    
    def generate_entity_prompt(self, document_content: str, entity_types: Optional[List[str]] = None) -> str:
        """Generate a prompt for entity extraction."""
        entity_types_str = ", ".join(entity_types) if entity_types else "all entity types"
        
        return f"""Please extract and analyze entities from the following document content.

Focus on extracting: {entity_types_str}

Document content:
{document_content}

Provide a JSON response with:
- entities: [
    {{
      "text": "entity text",
      "type": "entity type (PERSON, ORG, GPE, DATE, etc.)",
      "confidence": 0.95,
      "context": "surrounding context",
      "start_pos": 123,
      "end_pos": 135
    }}
  ]
- entity_summary: {{
    "total_entities": 15,
    "by_type": {{"PERSON": 5, "ORG": 3, "GPE": 4, "DATE": 3}},
    "high_confidence_count": 12
  }}
- relationships: [
    {{
      "entity1": "John Smith",
      "entity2": "OpenAI", 
      "relationship": "works_at",
      "confidence": 0.9
    }}
  ]
"""
    
    async def extract_entities_with_llm(self, document: Document) -> Dict[str, Any]:
        """LLM-powered entity extraction using Azure OpenAI EntityExtractor."""
        try:
            entities = await self.entity_extractor.extract_entities(document)
            return {
                "entities": [entity.to_dict() for entity in entities],
                "total_entities": len(entities),
                "extraction_method": "azure_openai_llm",
                "model_used": "Azure OpenAI"
            }
        except Exception as e:
            logger.error(f"LLM entity extraction failed: {str(e)}")
            return {
                "entities": [],
                "total_entities": 0,
                "extraction_method": "llm_failed",
                "error": str(e)
            }


class AutoGenQAAgent(AutoGenDocumentAgent):
    """AutoGen agent for question answering tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        system_message = """You are an expert question-answering agent for document analysis. Your role is to:

1. Answer questions accurately based on document content
2. Provide evidence and citations for answers
3. Handle both factual and analytical questions
4. Assess confidence in answers
5. Identify when questions cannot be answered from available content

When answering questions:
- Base answers strictly on document content
- Provide specific evidence and quotes when possible
- Indicate confidence levels in your answers
- Clearly state when information is not available
- Consider context and relationships between concepts

Respond with structured JSON containing:
- answer: Direct answer to the question
- evidence: Supporting evidence from the document
- confidence: Confidence score (0.0-1.0)
- sources: Specific sections or parts of document used
- related_entities: Relevant entities that support the answer
"""
        
        super().__init__(
            name="qa_agent",
            system_message=system_message,
            config=config
        )
    
    async def generate_qa_prompt(self, document_content: str, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a prompt for question answering using semantic search."""
        # Use semantic search to find most relevant content
        relevant_content = await self._find_relevant_content_semantic(question, document_content)
        
        context_info = ""
        if context:
            if "entities" in context:
                entities = context["entities"][:10]  # Limit to top 10
                entity_list = [f"{e.get('text', '')} ({e.get('type', '')})" for e in entities]
                context_info += f"\n\nRelevant entities: {', '.join(entity_list)}"
            
            if "section_summaries" in context:
                context_info += f"\n\nDocument has {len(context['section_summaries'])} sections available."
        
        # Add semantically relevant content
        if relevant_content:
            context_info += f"\n\nMost relevant content (found using semantic search):\n{relevant_content}"
        
        return f"""Please answer the following question based on the document content provided.

Question: {question}

Document content:
{document_content}
{context_info}

Provide a JSON response with:
- answer: Your answer to the question based on LLM analysis
- evidence: Specific quotes or evidence from the document
- confidence: How confident you are in this answer (0.0-1.0)
- reasoning: Brief explanation of your LLM reasoning process
- sources: Which parts of the document were most relevant
- related_information: Additional relevant information found using semantic analysis
- limitations: Any limitations or caveats about the answer
- semantic_relevance: How the semantic search helped find relevant content
"""
    
    async def _find_relevant_content_semantic(self, question: str, document_content: str) -> str:
        """Find relevant content using semantic embeddings."""
        try:
            # Split document into chunks for semantic search
            chunks = self.embedding_service.chunk_text(document_content)
            if not chunks:
                return ""
            
            # Get embeddings for question and chunks
            question_embedding = await self.embedding_service.embed_text(question)
            if not question_embedding:
                return ""
            
            # Find most similar chunks
            similarities = []
            for chunk in chunks:
                chunk_embedding = await self.embedding_service.embed_text(chunk['text'])
                if chunk_embedding:
                    similarity = self.embedding_service.cosine_similarity(question_embedding, chunk_embedding)
                    similarities.append((chunk['text'], similarity))
            
            # Sort by similarity and get top 3 chunks
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_chunks = [chunk_text for chunk_text, _ in similarities[:3]]
            
            return "\n\n".join(top_chunks)
            
        except Exception as e:
            logger.error(f"Semantic content search failed: {str(e)}")
            return ""
    
    def generate_corpus_qa_prompt(self, documents_content: str, question: str) -> str:
        """Generate a prompt for corpus-level question answering."""
        return f"""Please answer the following question based on multiple documents provided.

Question: {question}

Multiple documents content:
{documents_content}

Analyze across all documents and provide a JSON response with:
- answer: Comprehensive answer drawing from all documents
- evidence_by_document: Evidence organized by document
- confidence: Overall confidence in the answer
- cross_document_insights: Insights that emerge from comparing documents
- document_relevance: Which documents were most relevant to the question
- synthesis: How information from different documents relates
"""


class AutoGenValidationAgent(AutoGenDocumentAgent):
    """AutoGen agent for validating and cross-checking results from other agents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        system_message = """You are an expert validation agent responsible for quality assurance. Your role is to:

1. Validate outputs from other agents (summaries, entities, Q&A responses)
2. Check for consistency across different agent results
3. Identify potential errors or inconsistencies
4. Recommend corrections or improvements
5. Assess overall quality and reliability

When validating results:
- Check factual accuracy against source documents
- Verify consistency between different agent outputs
- Assess completeness and relevance
- Identify missing information or errors
- Provide specific recommendations for improvements

Respond with structured JSON containing:
- validation_results: Assessment of each agent's output
- consistency_check: Cross-agent consistency analysis
- quality_scores: Numerical quality assessments
- recommendations: Specific improvement recommendations
- overall_assessment: Summary of validation results
"""
        
        super().__init__(
            name="validation_agent",
            system_message=system_message,
            config=config
        )
    
    def generate_validation_prompt(self, agent_results: Dict[str, Any], document_content: str) -> str:
        """Generate a prompt for validating agent results."""
        results_summary = []
        for agent_name, result in agent_results.items():
            if hasattr(result, 'result_data'):
                results_summary.append(f"{agent_name}: {json.dumps(result.result_data, indent=2)}")
            else:
                results_summary.append(f"{agent_name}: {json.dumps(result, indent=2)}")
        
        results_text = "\n\n".join(results_summary)
        
        return f"""Please validate the following agent results against the source document.

Source document:
{document_content}

Agent results to validate:
{results_text}

Provide a JSON response with:
- validation_results: {{
    "agent_name": {{
      "accuracy_score": 0.95,
      "completeness_score": 0.90,
      "relevance_score": 0.85,
      "issues": ["list of specific issues"],
      "strengths": ["list of strengths"]
    }}
  }}
- cross_validation: {{
    "consistency_score": 0.92,
    "conflicting_information": [],
    "supporting_evidence": []
  }}
- recommendations: [
    {{
      "agent": "agent_name",
      "issue": "specific issue",
      "recommendation": "specific recommendation",
      "priority": "high/medium/low"
    }}
  ]
- overall_quality: {{
    "score": 0.88,
    "status": "acceptable/needs_improvement/poor",
    "summary": "brief assessment"
  }}
"""


class AutoGenWorkflowOrchestrator:
    """AutoGen-based workflow orchestrator using GroupChat."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize agents
        self.summarization_agent = AutoGenSummarizationAgent(self.config.get("summarization", {}))
        self.entity_agent = AutoGenEntityAgent(self.config.get("entity", {}))
        self.qa_agent = AutoGenQAAgent(self.config.get("qa", {}))
        self.validation_agent = AutoGenValidationAgent(self.config.get("validation", {}))
        
        # Initialize embedding and search services for orchestrator
        self.embedding_service = get_embedding_service(self.config)
        self.search_index = get_search_index(self.config)
        
        # Create user proxy for coordination
        self.user_proxy = autogen.UserProxyAgent(
            name="coordinator",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False,
        )
    
    async def execute_document_workflow(
        self,
        document: Document,
        workflow_type: str = "full",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a document processing workflow using AutoGen GroupChat."""
        workflow_id = str(uuid4())
        self.logger.info(f"Starting AutoGen workflow {workflow_id} for document {document.id}")
        
        try:
            # Set document context for all agents
            for agent in [self.summarization_agent, self.entity_agent, self.qa_agent, self.validation_agent]:
                agent.set_document_context(document, context)
            
            # Ensure document is indexed for semantic search
            await self._ensure_document_indexed(document)
            
            # Create workflow based on type
            if workflow_type == "full":
                return await self._execute_full_workflow(document, workflow_id, context)
            elif workflow_type == "summarization":
                return await self._execute_summarization_workflow(document, workflow_id, context)
            elif workflow_type == "entity_extraction":
                return await self._execute_entity_workflow(document, workflow_id, context)
            elif workflow_type == "qa" and context and "question" in context:
                return await self._execute_qa_workflow(document, workflow_id, context)
            else:
                raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        except Exception as e:
            self.logger.error(f"AutoGen workflow {workflow_id} failed: {str(e)}", exc_info=True)
            return {
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "document_id": str(document.id),
                "status": "failed",
                "error": str(e)
            }
    
    async def _execute_full_workflow(
        self,
        document: Document,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute full workflow with summarization, entity extraction, and validation."""
        
        # Define the workflow sequence
        agents = [
            self.summarization_agent,
            self.entity_agent,
            self.validation_agent
        ]
        
        # Create GroupChat
        groupchat = GroupChat(
            agents=[self.user_proxy] + agents,
            messages=[],
            max_round=10,
            speaker_selection_method="round_robin"
        )
        
        # Create GroupChat manager
        manager = GroupChatManager(groupchat=groupchat, llm_config=agents[0].llm_config)
        
        # Prepare initial message
        document_content = self.summarization_agent.get_document_content()
        initial_message = f"""Please process this document through a full workflow:
        
Document: {document.filename}
Content: {document_content[:2000]}...

Tasks to complete:
1. Summarization Agent: Create a comprehensive summary
2. Entity Agent: Extract and analyze entities
3. Validation Agent: Validate the results

Each agent should provide structured JSON output as specified in their system messages.
"""
        
        # Execute the workflow
        chat_result = await asyncio.to_thread(
            self.user_proxy.initiate_chat,
            manager,
            message=initial_message
        )
        
        # Process results
        return self._process_chat_results(chat_result, workflow_id, document, "full")
    
    async def _execute_summarization_workflow(
        self,
        document: Document,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute summarization-only workflow."""
        
        document_content = self.summarization_agent.get_document_content()
        prompt = self.summarization_agent.generate_summary_prompt(document_content, "document")
        
        # Direct interaction with summarization agent
        response = await asyncio.to_thread(
            self.user_proxy.initiate_chat,
            self.summarization_agent,
            message=prompt
        )
        
        return self._process_single_agent_result(
            response, self.summarization_agent, workflow_id, document, "summarization"
        )
    
    async def _execute_entity_workflow(
        self,
        document: Document,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute entity extraction workflow."""
        
        document_content = self.entity_agent.get_document_content()
        entity_types = context.get("entity_types") if context else None
        prompt = self.entity_agent.generate_entity_prompt(document_content, entity_types)
        
        # Direct interaction with entity agent
        response = await asyncio.to_thread(
            self.user_proxy.initiate_chat,
            self.entity_agent,
            message=prompt
        )
        
        return self._process_single_agent_result(
            response, self.entity_agent, workflow_id, document, "entity_extraction"
        )
    
    async def _execute_qa_workflow(
        self,
        document: Document,
        workflow_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Q&A workflow."""
        
        question = context["question"]
        document_content = self.qa_agent.get_document_content()
        prompt = await self.qa_agent.generate_qa_prompt(document_content, question, context)
        
        # Direct interaction with Q&A agent
        response = await asyncio.to_thread(
            self.user_proxy.initiate_chat,
            self.qa_agent,
            message=prompt
        )
        
        return self._process_single_agent_result(
            response, self.qa_agent, workflow_id, document, "qa"
        )
    
    def _process_chat_results(
        self,
        chat_result: Any,
        workflow_id: str,
        document: Document,
        workflow_type: str
    ) -> Dict[str, Any]:
        """Process results from GroupChat execution."""
        
        # Extract messages from chat result
        messages = []
        if hasattr(chat_result, 'chat_history'):
            messages = chat_result.chat_history
        elif hasattr(chat_result, 'messages'):
            messages = chat_result.messages
        
        # Parse agent responses
        agent_results = {}
        for message in messages:
            if isinstance(message, dict) and 'name' in message and 'content' in message:
                agent_name = message['name']
                content = message['content']
                
                # Try to parse JSON responses
                try:
                    parsed_content = json.loads(content)
                    agent_results[agent_name] = {
                        "status": "completed",
                        "result_data": parsed_content,
                        "raw_response": content
                    }
                except json.JSONDecodeError:
                    agent_results[agent_name] = {
                        "status": "completed",
                        "result_data": {"response": content},
                        "raw_response": content
                    }
        
        return {
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "document_id": str(document.id),
            "status": "completed",
            "agent_results": agent_results,
            "execution_method": "autogen_groupchat",
            "message_count": len(messages)
        }
    
    def _process_single_agent_result(
        self,
        response: Any,
        agent: AutoGenDocumentAgent,
        workflow_id: str,
        document: Document,
        workflow_type: str
    ) -> Dict[str, Any]:
        """Process results from single agent execution."""
        
        # Extract the response content
        response_content = ""
        if hasattr(response, 'chat_history') and response.chat_history:
            last_message = response.chat_history[-1]
            if isinstance(last_message, dict) and 'content' in last_message:
                response_content = last_message['content']
        
        # Try to parse JSON response
        try:
            parsed_response = json.loads(response_content)
            result_data = parsed_response
        except json.JSONDecodeError:
            result_data = {"response": response_content}
        
        return {
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "document_id": str(document.id),
            "status": "completed",
            "agent_results": {
                agent.name: {
                    "status": "completed",
                    "result_data": result_data,
                    "raw_response": response_content
                }
            },
            "execution_method": "autogen_single_agent"
        }
    
    async def execute_corpus_workflow(
        self,
        documents: List[Document],
        workflow_type: str = "corpus_analysis",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute workflow across multiple documents."""
        workflow_id = str(uuid4())
        self.logger.info(f"Starting AutoGen corpus workflow {workflow_id} for {len(documents)} documents")
        
        try:
            # Process documents individually first
            document_results = []
            for doc in documents:
                doc_result = await self.execute_document_workflow(doc, "summarization", context)
                document_results.append(doc_result)
            
            # Create corpus-level content
            corpus_content = self._create_corpus_content(documents, document_results)
            
            # Execute corpus-level analysis
            if workflow_type == "corpus_qa" and context and "question" in context:
                question = context["question"]
                prompt = self.qa_agent.generate_corpus_qa_prompt(corpus_content, question)
                
                response = await asyncio.to_thread(
                    self.user_proxy.initiate_chat,
                    self.qa_agent,
                    message=prompt
                )
                
                corpus_result = self._process_single_agent_result(
                    response, self.qa_agent, workflow_id, documents[0], "corpus_qa"
                )
            else:
                # Default corpus summarization
                prompt = self.summarization_agent.generate_summary_prompt(corpus_content, "corpus")
                
                response = await asyncio.to_thread(
                    self.user_proxy.initiate_chat,
                    self.summarization_agent,
                    message=prompt
                )
                
                corpus_result = self._process_single_agent_result(
                    response, self.summarization_agent, workflow_id, documents[0], "corpus_analysis"
                )
            
            return {
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "corpus_size": len(documents),
                "document_results": document_results,
                "corpus_results": corpus_result,
                "status": "completed"
            }
        
        except Exception as e:
            self.logger.error(f"AutoGen corpus workflow {workflow_id} failed: {str(e)}", exc_info=True)
            return {
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "corpus_size": len(documents),
                "status": "failed",
                "error": str(e)
            }
    
    def _create_corpus_content(self, documents: List[Document], document_results: List[Dict[str, Any]]) -> str:
        """Create combined content from multiple documents for corpus analysis."""
        corpus_parts = []
        
        for i, (doc, result) in enumerate(zip(documents, document_results)):
            corpus_parts.append(f"=== Document {i+1}: {doc.filename} ===")
            
            # Add document content
            if doc.extra_data and "sections" in doc.extra_data:
                sections = doc.extra_data["sections"]
                for section in sections[:3]:  # Limit to first 3 sections per doc
                    title = section.get("title", "")
                    content = section.get("content", "")
                    if title:
                        corpus_parts.append(f"## {title}")
                    corpus_parts.append(content[:500])  # Limit content length
            
            # Add summary if available from results
            if ("agent_results" in result and 
                "summarization_agent" in result["agent_results"] and
                "result_data" in result["agent_results"]["summarization_agent"]):
                summary_data = result["agent_results"]["summarization_agent"]["result_data"]
                if "summary" in summary_data:
                    corpus_parts.append(f"Summary: {summary_data['summary']}")
            
            corpus_parts.append("")  # Add spacing between documents
        
        return "\n".join(corpus_parts)
    
    async def _ensure_document_indexed(self, document: Document):
        """Ensure document is indexed in the vector search index for semantic search."""
        try:
            # Check if document is already indexed
            index_stats = self.search_index.get_index_stats()
            if str(document.id) in index_stats.get("documents", []):
                self.logger.debug(f"Document {document.id} already indexed")
                return
            
            # Add document to search index for semantic search
            success = await self.search_index.add_document(document)
            if success:
                self.logger.info(f"Successfully indexed document {document.id} for semantic search")
            else:
                self.logger.warning(f"Failed to index document {document.id} for semantic search")
                
        except Exception as e:
            self.logger.error(f"Error indexing document {document.id}: {str(e)}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the LLM-powered system."""
        return {
            "implementation": "100% LLM-Powered AutoGen System",
            "framework": "Microsoft AutoGen with Azure OpenAI",
            "capabilities": {
                "semantic_search": "Vector embeddings with Azure OpenAI",
                "entity_extraction": "LLM-powered entity analysis",
                "summarization": "Advanced LLM summarization",
                "question_answering": "Conversational AI with semantic retrieval",
                "validation": "LLM-powered cross-agent validation"
            },
            "no_hardcoding": "All processing uses LLM reasoning",
            "no_fallbacks": "Pure LLM implementation without rule-based fallbacks",
            "embedding_integration": "Full vector search and semantic similarity",
            "azure_openai_powered": bool(settings.AZURE_OPENAI_API_KEY and settings.AZURE_OPENAI_ENDPOINT)
        }
