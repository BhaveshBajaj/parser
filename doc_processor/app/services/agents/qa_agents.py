"""Question & Answer agents for document corpus querying."""

import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from .base_agent import BaseAgent, AgentResult, AgentStatus, AgentError
from ...models.document import Document
from ..entity_extractor import Entity, EntityType

logger = logging.getLogger(__name__)


class DocumentQAAgent(BaseAgent):
    """Agent for answering questions about a single document."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="document_qa",
            description="Answers questions about individual documents",
            config=config
        )
        self.max_context_length = self.config.get("max_context_length", 4000)
        self.answer_length = self.config.get("answer_length", 200)
        self.use_entities = self.config.get("use_entities", True)
    
    async def validate_input(self, document: Document, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate that we have a question and document content."""
        if not context or "question" not in context:
            self.logger.warning("No question provided in context")
            return False
        
        question = context["question"]
        if not question or not question.strip():
            self.logger.warning("Empty question provided")
            return False
        
        # Check if document has content in sections or summary
        has_sections = (document.extra_data and 
                       "sections" in document.extra_data and 
                       len(document.extra_data["sections"]) > 0)
        has_summary = document.summary and len(document.summary.strip()) > 0
        
        if not (has_sections or has_summary):
            self.logger.warning(f"Document {document.id} has no content for Q&A")
            return False
        
        return True
    
    async def process(
        self, 
        document: Document, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """Answer a question about the document."""
        try:
            question = context["question"]
            
            # Get content from sections or summary
            sections = []
            if document.extra_data and "sections" in document.extra_data:
                sections = document.extra_data["sections"]
            elif document.summary and document.summary.strip():
                # Create a virtual section from summary
                sections = [{
                    "title": "Document Summary",
                    "content": document.summary,
                    "type": "summary"
                }]
            
            entities = []
            if self.use_entities and document.extra_data and "all_entities" in document.extra_data:
                entities = document.extra_data["all_entities"]
            
            # Find relevant sections
            relevant_sections = await self._find_relevant_sections(question, sections)
            
            # Extract relevant context
            context_text = self._extract_context(relevant_sections, entities)
            
            # Generate answer
            answer = await self._generate_answer(question, context_text, document)
            
            # Find supporting evidence
            evidence = self._find_evidence(answer, relevant_sections)
            
            return self._create_result(
                status=AgentStatus.COMPLETED,
                result_data={
                    "question": question,
                    "answer": answer,
                    "relevant_sections": relevant_sections,
                    "evidence": evidence,
                    "confidence": self._calculate_confidence(question, answer, relevant_sections),
                    "context_length": len(context_text)
                },
                metadata={
                    "agent_version": "1.0",
                    "processing_type": "document_qa",
                    "sections_analyzed": len(sections),
                    "relevant_sections_found": len(relevant_sections),
                    "entities_used": len(entities) if self.use_entities else 0
                }
            )
            
        except Exception as e:
            raise AgentError(f"Failed to answer question: {str(e)}", self.name)
    
    async def _find_relevant_sections(self, question: str, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find sections most relevant to the question."""
        relevant_sections = []
        question_words = set(question.lower().split())
        
        for section in sections:
            title = section.get("title", "").lower()
            content = section.get("content", "").lower()
            
            # Simple keyword matching (in real implementation, use semantic similarity)
            title_matches = len(question_words.intersection(set(title.split())))
            content_matches = len(question_words.intersection(set(content.split())))
            
            relevance_score = title_matches * 2 + content_matches * 0.1  # Title matches weighted higher
            
            if relevance_score > 0:
                section_with_score = {**section, "relevance_score": relevance_score}
                relevant_sections.append(section_with_score)
        
        # Sort by relevance and return top sections
        relevant_sections.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant_sections[:5]  # Top 5 most relevant sections
    
    def _extract_context(self, sections: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> str:
        """Extract context text from relevant sections and entities."""
        context_parts = []
        
        # Add section content
        for section in sections:
            title = section.get("title", "")
            content = section.get("content", "")
            if title:
                context_parts.append(f"## {title}")
            context_parts.append(content)
        
        # Add relevant entities
        if entities:
            entity_text = "Relevant entities: " + ", ".join([
                f"{e.get('text', '')} ({e.get('type', '')})" 
                for e in entities[:10]  # Limit to 10 entities
            ])
            context_parts.append(entity_text)
        
        full_context = "\n\n".join(context_parts)
        
        # Truncate if too long
        if len(full_context) > self.max_context_length:
            full_context = full_context[:self.max_context_length] + "..."
        
        return full_context
    
    async def _generate_answer(self, question: str, context: str, document: Document) -> str:
        """Generate an answer to the question based on the context using Azure OpenAI."""
        try:
            from ...core.config import settings
            from openai import AzureOpenAI
            
            # Check if Azure OpenAI is configured
            if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
                self.logger.info("Azure OpenAI not configured, using extractive fallback")
                return await self._generate_extractive_answer(question, context)
            
            # Initialize Azure OpenAI client
            client = AzureOpenAI(
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
            )
            deployment = settings.AZURE_OPENAI_DEPLOYMENT
            
            # Create the prompt for Q&A
            system_message = """You are a helpful assistant that answers questions based on document content. 
            Provide accurate, concise answers based only on the information provided in the context. 
            If the answer is not in the context, say so clearly. 
            Keep answers under 200 words."""
            
            user_message = f"""Context from document:
{context}

Question: {question}

Please answer the question based on the context provided."""
            
            # Call Azure OpenAI API
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Log the successful LLM call
            self.logger.info(f"Generated LLM answer for question: {question[:50]}...")
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error generating LLM answer: {str(e)}")
            # Fallback to extractive approach
            return await self._generate_extractive_answer(question, context)
    
    async def _generate_extractive_answer(self, question: str, context: str) -> str:
        """Generate an answer using simple extractive approach (fallback)."""
        question_words = set(question.lower().split())
        context_sentences = context.split('. ')
        
        # Find sentences that contain question keywords
        relevant_sentences = []
        for sentence in context_sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words.intersection(sentence_words))
            if overlap > 0:
                relevant_sentences.append((sentence, overlap))
        
        if not relevant_sentences:
            return "I couldn't find a specific answer to your question in this document."
        
        # Sort by relevance and take top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        answer_sentences = [sent[0] for sent in relevant_sentences[:3]]
        
        answer = '. '.join(answer_sentences)
        
        # Truncate if too long
        if len(answer) > self.answer_length:
            answer = answer[:self.answer_length] + "..."
        
        return answer
    
    def _find_evidence(self, answer: str, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find evidence supporting the answer."""
        evidence = []
        answer_words = set(answer.lower().split())
        
        for section in sections:
            content = section.get("content", "")
            content_words = set(content.lower().split())
            
            # Calculate overlap between answer and section content
            overlap = len(answer_words.intersection(content_words))
            if overlap > 2:  # At least 3 words in common
                evidence.append({
                    "section_title": section.get("title", ""),
                    "section_content": content[:200] + "..." if len(content) > 200 else content,
                    "relevance_score": overlap
                })
        
        return sorted(evidence, key=lambda x: x["relevance_score"], reverse=True)[:3]
    
    def _calculate_confidence(self, question: str, answer: str, sections: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the answer."""
        if not answer or "couldn't find" in answer.lower():
            return 0.1
        
        # Simple heuristics for confidence
        confidence = 0.5  # Base confidence
        
        # Boost confidence if we found relevant sections
        if sections:
            confidence += 0.2
        
        # Boost confidence based on answer length (longer = more confident)
        if len(answer) > 50:
            confidence += 0.2
        
        # Boost confidence if answer contains specific details
        if any(char.isdigit() for char in answer):
            confidence += 0.1
        
        return min(1.0, confidence)


class CorpusQAAgent(BaseAgent):
    """Agent for answering questions across multiple documents in a corpus."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="corpus_qa",
            description="Answers questions across multiple documents",
            config=config
        )
        self.max_documents = self.config.get("max_documents", 20)
        self.answer_length = self.config.get("answer_length", 300)
        self.combine_answers = self.config.get("combine_answers", True)
    
    async def validate_input(self, document: Document, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate that we have a question and corpus documents."""
        if not context or "question" not in context:
            self.logger.warning("No question provided in context")
            return False
        
        if not context or "corpus_documents" not in context:
            self.logger.warning("No corpus documents provided in context")
            return False
        
        corpus_documents = context["corpus_documents"]
        if not corpus_documents or len(corpus_documents) == 0:
            self.logger.warning("Empty corpus provided")
            return False
        
        return True
    
    async def process(
        self, 
        document: Document, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """Answer a question across multiple documents."""
        try:
            question = context["question"]
            corpus_documents = context["corpus_documents"]
            
            # Limit corpus size if needed
            if len(corpus_documents) > self.max_documents:
                corpus_documents = corpus_documents[:self.max_documents]
                self.logger.warning(f"Limited corpus to {self.max_documents} documents")
            
            # Get answers from individual documents
            document_answers = []
            for doc in corpus_documents:
                try:
                    doc_answer = await self._get_document_answer(question, doc)
                    if doc_answer:
                        document_answers.append(doc_answer)
                except Exception as e:
                    self.logger.warning(f"Failed to get answer from document {doc.id}: {str(e)}")
                    continue
            
            # Combine and synthesize answers
            final_answer = await self._synthesize_answers(question, document_answers)
            
            # Rank document relevance
            document_relevance = self._rank_document_relevance(document_answers)
            
            return self._create_result(
                status=AgentStatus.COMPLETED,
                result_data={
                    "question": question,
                    "answer": final_answer,
                    "document_answers": document_answers,
                    "document_relevance": document_relevance,
                    "total_documents_searched": len(corpus_documents),
                    "documents_with_answers": len(document_answers)
                },
                metadata={
                    "agent_version": "1.0",
                    "processing_type": "corpus_qa",
                    "corpus_size": len(corpus_documents),
                    "successful_extractions": len(document_answers)
                }
            )
            
        except Exception as e:
            raise AgentError(f"Failed to answer corpus question: {str(e)}", self.name)
    
    async def _get_document_answer(self, question: str, document: Document) -> Optional[Dict[str, Any]]:
        """Get an answer from a single document."""
        if not document.extra_data or "sections" not in document.extra_data:
            return None
        
        sections = document.extra_data["sections"]
        if not sections:
            return None
        
        # Use the DocumentQAAgent logic
        doc_qa_agent = DocumentQAAgent()
        try:
            result = await doc_qa_agent.process(
                document,
                context={"question": question}
            )
            
            if result.is_successful():
                return {
                    "document_id": str(document.id),
                    "document_filename": document.filename,
                    "answer": result.result_data["answer"],
                    "confidence": result.result_data["confidence"],
                    "evidence": result.result_data["evidence"]
                }
        except Exception:
            pass
        
        return None
    
    async def _synthesize_answers(self, question: str, document_answers: List[Dict[str, Any]]) -> str:
        """Synthesize multiple document answers into a coherent response."""
        if not document_answers:
            return "I couldn't find an answer to your question in the provided documents."
        
        if len(document_answers) == 1:
            return document_answers[0]["answer"]
        
        if not self.combine_answers:
            # Return the highest confidence answer
            best_answer = max(document_answers, key=lambda x: x["confidence"])
            return f"From {best_answer['document_filename']}: {best_answer['answer']}"
        
        # Combine answers from multiple documents
        combined_parts = []
        for i, doc_answer in enumerate(document_answers[:3]):  # Limit to top 3
            filename = doc_answer["document_filename"]
            answer = doc_answer["answer"]
            combined_parts.append(f"From {filename}: {answer}")
        
        combined_answer = "\n\n".join(combined_parts)
        
        # Truncate if too long
        if len(combined_answer) > self.answer_length:
            combined_answer = combined_answer[:self.answer_length] + "..."
        
        return combined_answer
    
    def _rank_document_relevance(self, document_answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank documents by their relevance to the question."""
        ranked_docs = []
        
        for doc_answer in document_answers:
            relevance_score = doc_answer["confidence"]
            
            # Boost score for longer answers (more detailed)
            answer_length = len(doc_answer["answer"])
            if answer_length > 100:
                relevance_score += 0.1
            
            # Boost score for having evidence
            if doc_answer.get("evidence"):
                relevance_score += 0.1
            
            ranked_docs.append({
                "document_id": doc_answer["document_id"],
                "document_filename": doc_answer["document_filename"],
                "relevance_score": relevance_score
            })
        
        return sorted(ranked_docs, key=lambda x: x["relevance_score"], reverse=True)


class ContextualQAAgent(BaseAgent):
    """Agent for answering questions with enhanced contextual understanding."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="contextual_qa",
            description="Answers questions with enhanced contextual understanding",
            config=config
        )
        self.context_window = self.config.get("context_window", 5)
        self.use_entity_context = self.config.get("use_entity_context", True)
        self.use_cross_references = self.config.get("use_cross_references", True)
    
    async def validate_input(self, document: Document, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate input for contextual Q&A."""
        if not context or "question" not in context:
            self.logger.warning("No question provided in context")
            return False
        
        return True
    
    async def process(
        self, 
        document: Document, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """Answer a question with enhanced contextual understanding."""
        try:
            question = context["question"]
            
            # Build enhanced context
            enhanced_context = await self._build_enhanced_context(question, document, context)
            
            # Generate contextual answer
            answer = await self._generate_contextual_answer(question, enhanced_context)
            
            # Extract contextual insights
            insights = self._extract_contextual_insights(enhanced_context)
            
            return self._create_result(
                status=AgentStatus.COMPLETED,
                result_data={
                    "question": question,
                    "answer": answer,
                    "enhanced_context": enhanced_context,
                    "contextual_insights": insights,
                    "context_quality": self._assess_context_quality(enhanced_context)
                },
                metadata={
                    "agent_version": "1.0",
                    "processing_type": "contextual_qa",
                    "context_enhancements_used": list(enhanced_context.keys())
                }
            )
            
        except Exception as e:
            raise AgentError(f"Failed contextual Q&A: {str(e)}", self.name)
    
    async def _build_enhanced_context(
        self, 
        question: str, 
        document: Document, 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build enhanced context for better question answering."""
        enhanced_context = {
            "question": question,
            "document_context": {},
            "entity_context": {},
            "cross_references": {},
            "temporal_context": {},
            "semantic_context": {}
        }
        
        # Document context
        if document.extra_data and "sections" in document.extra_data:
            sections = document.extra_data["sections"]
            enhanced_context["document_context"] = {
                "total_sections": len(sections),
                "section_titles": [s.get("title", "") for s in sections],
                "content_summary": self._summarize_content(sections)
            }
        
        # Entity context
        if self.use_entity_context and document.extra_data and "all_entities" in document.extra_data:
            entities = document.extra_data["all_entities"]
            enhanced_context["entity_context"] = self._build_entity_context(entities, question)
        
        # Cross-references
        if self.use_cross_references and context and "corpus_documents" in context:
            corpus_docs = context["corpus_documents"]
            enhanced_context["cross_references"] = await self._find_cross_references(question, corpus_docs)
        
        # Temporal context
        enhanced_context["temporal_context"] = self._extract_temporal_context(question, document)
        
        return enhanced_context
    
    def _summarize_content(self, sections: List[Dict[str, Any]]) -> str:
        """Create a brief summary of document content."""
        if not sections:
            return ""
        
        # Take first sentence from each section
        summary_parts = []
        for section in sections[:5]:  # Limit to first 5 sections
            content = section.get("content", "")
            sentences = content.split('. ')
            if sentences:
                summary_parts.append(sentences[0])
        
        return ". ".join(summary_parts)
    
    def _build_entity_context(self, entities: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
        """Build context from relevant entities."""
        question_words = set(question.lower().split())
        relevant_entities = []
        
        for entity in entities:
            entity_text = entity.get("text", "").lower()
            # Check if entity text contains any question word OR if any question word is in entity text
            if (any(word in entity_text for word in question_words) or 
                any(entity_text in word or word in entity_text for word in question_words)):
                relevant_entities.append(entity)
        
        # Also include entities that might be relevant by type
        if not relevant_entities:
            # If asking about "who", look for PERSON entities
            if any(word in question.lower() for word in ["who", "person", "people"]):
                relevant_entities.extend([e for e in entities if e.get("type") == "PERSON"])
            # If asking about "where", look for location entities
            elif any(word in question.lower() for word in ["where", "location", "place"]):
                relevant_entities.extend([e for e in entities if e.get("type") in ["GPE", "LOC", "LOCATION"]])
            # If asking about "when", look for time entities
            elif any(word in question.lower() for word in ["when", "time", "date"]):
                relevant_entities.extend([e for e in entities if e.get("type") in ["DATE", "TIME"]])
        
        entity_types = {}
        for entity in relevant_entities:
            entity_type = entity.get("type", "UNKNOWN")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        return {
            "relevant_entities": relevant_entities[:10],  # Limit to 10
            "entity_type_distribution": entity_types,
            "total_relevant": len(relevant_entities)
        }
    
    async def _find_cross_references(self, question: str, corpus_docs: List[Document]) -> Dict[str, Any]:
        """Find cross-references in other documents."""
        cross_refs = []
        question_words = set(question.lower().split())
        
        for doc in corpus_docs[:5]:  # Limit to 5 documents
            if doc.extra_data and "sections" in doc.extra_data:
                for section in doc.extra_data["sections"]:
                    content = section.get("content", "").lower()
                    content_words = set(content.split())
                    
                    overlap = len(question_words.intersection(content_words))
                    if overlap > 1:
                        cross_refs.append({
                            "document_id": str(doc.id),
                            "document_filename": doc.filename,
                            "section_title": section.get("title", ""),
                            "relevance_score": overlap
                        })
        
        return {
            "references": sorted(cross_refs, key=lambda x: x["relevance_score"], reverse=True)[:3],
            "total_references_found": len(cross_refs)
        }
    
    def _extract_temporal_context(self, question: str, document: Document) -> Dict[str, Any]:
        """Extract temporal context from the question and document."""
        temporal_words = ["when", "before", "after", "during", "since", "until", "date", "time", "year"]
        has_temporal = any(word in question.lower() for word in temporal_words)
        
        temporal_entities = []
        if document.extra_data and "all_entities" in document.extra_data:
            for entity in document.extra_data["all_entities"]:
                if entity.get("type") in ["DATE", "TIME"]:
                    temporal_entities.append(entity)
        
        return {
            "has_temporal_aspect": has_temporal,
            "temporal_entities": temporal_entities[:5],
            "temporal_entity_count": len(temporal_entities)
        }
    
    async def _generate_contextual_answer(self, question: str, enhanced_context: Dict[str, Any]) -> str:
        """Generate an answer using enhanced context and LLM."""
        try:
            from ...core.config import settings
            from openai import AzureOpenAI
            
            # Check if Azure OpenAI is configured
            if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
                self.logger.info("Azure OpenAI not configured, using contextual fallback")
                return await self._generate_contextual_fallback(question, enhanced_context)
            
            # Initialize Azure OpenAI client
            client = AzureOpenAI(
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
            )
            deployment = settings.AZURE_OPENAI_DEPLOYMENT
            
            # Build comprehensive context for LLM
            context_parts = []
            
            # Add document content
            doc_context = enhanced_context.get("document_context", {})
            content_summary = doc_context.get("content_summary", "")
            if content_summary:
                context_parts.append(f"Document Content:\n{content_summary}")
            
            # Add entity information
            entity_context = enhanced_context.get("entity_context", {})
            if entity_context.get("relevant_entities"):
                entities = entity_context["relevant_entities"][:10]  # Top 10 entities
                entity_info = []
                for entity in entities:
                    text = entity.get("text", "")
                    entity_type = entity.get("type", "")
                    if text and entity_type:
                        entity_info.append(f"{text} ({entity_type})")
                
                if entity_info:
                    context_parts.append(f"Relevant Entities:\n{', '.join(entity_info)}")
            
            # Add temporal context if relevant
            temporal_context = enhanced_context.get("temporal_context", {})
            if temporal_context.get("temporal_entities"):
                temporal_entities = temporal_context["temporal_entities"][:5]
                temporal_info = []
                for entity in temporal_entities:
                    text = entity.get("text", "")
                    entity_type = entity.get("type", "")
                    if text and entity_type:
                        temporal_info.append(f"{text} ({entity_type})")
                
                if temporal_info:
                    context_parts.append(f"Time/Date Information:\n{', '.join(temporal_info)}")
            
            full_context = "\n\n".join(context_parts)
            
            if not full_context.strip():
                return "I don't have enough context from the document to answer your question."
            
            # Create the prompt for contextual Q&A
            system_message = """You are an expert document analyst that provides comprehensive answers using enhanced contextual understanding. 
            Analyze the provided document content, entities, and temporal information to give detailed, accurate answers.
            If the answer requires information not in the context, state that clearly.
            Provide specific details and references when possible."""
            
            user_message = f"""Enhanced Document Context:
{full_context}

Question: {question}

Please provide a comprehensive answer based on the enhanced context provided. Include specific details and entity references where relevant."""
            
            # Call Azure OpenAI API
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2,
                max_tokens=400
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Log the successful LLM call
            self.logger.info(f"Generated contextual LLM answer for question: {question[:50]}...")
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error generating contextual LLM answer: {str(e)}")
            # Fallback to non-LLM approach
            return await self._generate_contextual_fallback(question, enhanced_context)
    
    async def _generate_contextual_fallback(self, question: str, enhanced_context: Dict[str, Any]) -> str:
        """Generate contextual answer without LLM (fallback)."""
        # Extract content from document context
        doc_context = enhanced_context.get("document_context", {})
        content_summary = doc_context.get("content_summary", "")
        
        # If we have content summary, try to answer from it
        if content_summary:
            question_words = set(question.lower().split())
            
            # Find sentences that might contain answers
            sentences = content_summary.split('. ')
            relevant_sentences = []
            
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                overlap = len(question_words.intersection(sentence_words))
                
                # Also check for entity matches in the sentence
                entity_context = enhanced_context.get("entity_context", {})
                entity_bonus = 0
                if entity_context.get("relevant_entities"):
                    for entity in entity_context["relevant_entities"]:
                        entity_text = entity.get("text", "").lower()
                        if entity_text in sentence.lower():
                            entity_bonus += 2  # Bonus for sentences containing relevant entities
                
                total_score = overlap + entity_bonus
                if total_score > 0:
                    relevant_sentences.append((sentence, total_score))
            
            if relevant_sentences:
                # Sort by relevance and take the best match
                relevant_sentences.sort(key=lambda x: x[1], reverse=True)
                best_sentence = relevant_sentences[0][0]
                
                # Build answer with context
                base_answer = f"Based on the document content: {best_sentence}"
                
                # Add entity information if available
                entity_context = enhanced_context.get("entity_context", {})
                if entity_context.get("relevant_entities"):
                    entities = entity_context["relevant_entities"][:3]  # Top 3 entities
                    entity_names = [e.get("text", "") for e in entities]
                    if entity_names:
                        base_answer += f" Key entities mentioned include: {', '.join(entity_names)}."
                
                return base_answer
        
        # Fallback: try to answer from entities
        entity_context = enhanced_context.get("entity_context", {})
        if entity_context.get("relevant_entities"):
            entities = entity_context["relevant_entities"]
            entity_info = []
            for entity in entities[:5]:  # Top 5 entities
                text = entity.get("text", "")
                entity_type = entity.get("type", "")
                if text and entity_type:
                    entity_info.append(f"{text} ({entity_type})")
            
            if entity_info:
                return f"Based on the document analysis, I found these relevant entities: {', '.join(entity_info)}. This suggests the document contains information related to your question."
        
        # Final fallback
        return "I found some relevant information in the document, but I need more context to provide a specific answer to your question."
    
    def _extract_contextual_insights(self, enhanced_context: Dict[str, Any]) -> List[str]:
        """Extract insights from the enhanced context."""
        insights = []
        
        # Document insights
        doc_context = enhanced_context.get("document_context", {})
        if doc_context.get("total_sections", 0) > 10:
            insights.append("This is a comprehensive document with extensive content")
        
        # Entity insights
        entity_context = enhanced_context.get("entity_context", {})
        if entity_context.get("total_relevant", 0) > 5:
            insights.append("Multiple relevant entities found, suggesting rich contextual information")
        
        # Cross-reference insights
        cross_refs = enhanced_context.get("cross_references", {})
        if cross_refs.get("total_references_found", 0) > 0:
            insights.append("Cross-references found in related documents")
        
        # Temporal insights
        temporal_context = enhanced_context.get("temporal_context", {})
        if temporal_context.get("has_temporal_aspect"):
            insights.append("Question has temporal aspects that may require chronological analysis")
        
        return insights
    
    def _assess_context_quality(self, enhanced_context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of the enhanced context."""
        quality_score = 0.5  # Base score
        
        # Document context quality
        doc_context = enhanced_context.get("document_context", {})
        if doc_context.get("total_sections", 0) > 0:
            quality_score += 0.1
        
        # Entity context quality
        entity_context = enhanced_context.get("entity_context", {})
        if entity_context.get("total_relevant", 0) > 0:
            quality_score += 0.2
        
        # Cross-reference quality
        cross_refs = enhanced_context.get("cross_references", {})
        if cross_refs.get("total_references_found", 0) > 0:
            quality_score += 0.2
        
        return {
            "overall_score": min(1.0, quality_score),
            "has_document_context": bool(doc_context),
            "has_entity_context": bool(entity_context.get("relevant_entities")),
            "has_cross_references": bool(cross_refs.get("references")),
            "context_completeness": min(1.0, quality_score)
        }
