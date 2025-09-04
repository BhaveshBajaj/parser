"""AutoGen agent for content review including bias detection and completeness checks."""

import json
import re
from typing import Any, Dict, List, Optional

from .base_agent import BaseAutoGenAgent


class ContentReviewAgent(BaseAutoGenAgent):
    """AutoGen agent for content review, bias detection, and completeness analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        system_message = """You are a content review agent. Analyze content for bias, completeness, and quality.

CRITICAL: Respond with ONLY valid JSON.

Detect bias (gender, racial, cultural, political), assess completeness, check factual consistency, evaluate readability, and provide improvement recommendations.

JSON format:
{
  "bias_analysis": {"gender_bias": {"detected": false, "confidence": 0.95, "examples": [], "severity": "none"}, ...},
  "completeness_assessment": {"coverage_score": 0.85, "missing_topics": [], "depth_analysis": "adequate"},
  "quality_metrics": {"factual_accuracy": 0.90, "readability": 0.82, "structure": 0.87},
  "factual_consistency": {"internal_consistency": 0.90, "contradictions": []},
  "misinformation_risk": {"risk_level": "low", "confidence": 0.85, "potential_issues": []},
  "readability_score": {"overall_score": 0.82, "complexity_level": "intermediate"},
  "recommendations": [{"category": "bias", "issue": "description", "recommendation": "fix", "priority": "medium"}],
  "overall_assessment": {"quality_score": 0.87, "status": "acceptable", "summary": "brief"}
}

ONLY JSON response."""
        
        super().__init__(
            name="content_review_agent",
            system_message=system_message,
            config=config
        )
        
        # Bias detection patterns and keywords
        self.bias_patterns = {
            "gender_bias": [
                r"\b(he|him|his)\b.*\b(engineer|doctor|scientist|leader)\b",
                r"\b(she|her|hers)\b.*\b(nurse|teacher|secretary|assistant)\b",
                r"\b(man|men)\b.*\b(strong|aggressive|assertive)\b",
                r"\b(woman|women)\b.*\b(emotional|nurturing|gentle)\b"
            ],
            "racial_bias": [
                r"\b(black|white|asian|hispanic)\b.*\b(criminal|thug|terrorist)\b",
                r"\b(immigrant|foreigner)\b.*\b(illegal|criminal|threat)\b"
            ],
            "cultural_bias": [
                r"\b(western|eastern)\b.*\b(superior|inferior|advanced|backward)\b",
                r"\b(developed|developing)\b.*\b(country|nation)\b"
            ],
            "political_bias": [
                r"\b(liberal|conservative|left|right)\b.*\b(extreme|radical|fanatic)\b",
                r"\b(democrat|republican)\b.*\b(corrupt|incompetent|evil)\b"
            ]
        }
        
        # Quality assessment criteria
        self.quality_criteria = {
            "completeness": ["coverage", "depth", "context", "examples"],
            "accuracy": ["factual_correctness", "source_reliability", "consistency"],
            "clarity": ["readability", "structure", "terminology", "flow"],
            "objectivity": ["neutral_tone", "balanced_perspective", "evidence_based"]
        }
    
    def generate_content_review_prompt(self, content: str, content_type: str = "general") -> str:
        """Generate a prompt for content review analysis."""
        return f"""Review content for bias, completeness, quality. Return ONLY JSON:

Content: {content}

JSON: {{
  "bias_analysis": {{"gender_bias": {{"detected": false, "confidence": 0.95, "examples": [], "severity": "none"}}, "racial_bias": {{"detected": false, "confidence": 0.90, "examples": [], "severity": "none"}}, "cultural_bias": {{"detected": false, "confidence": 0.85, "examples": [], "severity": "none"}}, "political_bias": {{"detected": false, "confidence": 0.88, "examples": [], "severity": "none"}}, "overall_bias_score": 0.92}},
  "completeness_assessment": {{"coverage_score": 0.85, "missing_topics": [], "depth_analysis": "adequate", "context_provided": true, "examples_included": true}},
  "quality_metrics": {{"factual_accuracy": 0.90, "source_reliability": 0.85, "consistency": 0.88, "readability": 0.82, "structure": 0.87}},
  "factual_consistency": {{"internal_consistency": 0.90, "contradictions": [], "verification_needed": []}},
  "misinformation_risk": {{"risk_level": "low", "confidence": 0.85, "potential_issues": [], "verification_recommendations": []}},
  "readability_score": {{"overall_score": 0.82, "complexity_level": "intermediate", "clarity_issues": [], "improvement_suggestions": []}},
  "recommendations": [{{"category": "bias", "issue": "issue_description", "recommendation": "improvement_suggestion", "priority": "medium"}}],
  "overall_assessment": {{"quality_score": 0.87, "status": "acceptable", "summary": "Content quality assessment summary"}}
}}

ONLY JSON."""
    
    def detect_bias_patterns(self, content: str) -> Dict[str, Any]:
        """Detect bias patterns in content using regex patterns."""
        bias_results = {}
        
        for bias_type, patterns in self.bias_patterns.items():
            detected_examples = []
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    detected_examples.extend(matches)
            
            bias_results[bias_type] = {
                "detected": len(detected_examples) > 0,
                "examples": detected_examples[:5],  # Limit to 5 examples
                "count": len(detected_examples)
            }
        
        return bias_results
    
    def assess_content_completeness(self, content: str, expected_elements: Optional[List[str]] = None) -> Dict[str, Any]:
        """Assess content completeness based on expected elements."""
        if not expected_elements:
            expected_elements = ["introduction", "main_content", "conclusion", "examples", "context"]
        
        completeness_scores = {}
        for element in expected_elements:
            # Simple keyword-based detection (can be enhanced with more sophisticated NLP)
            if element.lower() in content.lower():
                completeness_scores[element] = 1.0
            else:
                completeness_scores[element] = 0.0
        
        return {
            "expected_elements": expected_elements,
            "completeness_scores": completeness_scores,
            "overall_completeness": sum(completeness_scores.values()) / len(completeness_scores)
        }
    
    def calculate_readability_score(self, content: str) -> Dict[str, Any]:
        """Calculate basic readability metrics."""
        sentences = re.split(r'[.!?]+', content)
        words = re.findall(r'\b\w+\b', content)
        
        if not sentences or not words:
            return {"score": 0.0, "level": "unknown"}
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple readability score (0-1, higher is more readable)
        readability_score = max(0, min(1, 1 - (avg_sentence_length - 10) / 20 - (avg_word_length - 4) / 10))
        
        if readability_score > 0.8:
            level = "easy"
        elif readability_score > 0.6:
            level = "intermediate"
        elif readability_score > 0.4:
            level = "difficult"
        else:
            level = "very_difficult"
        
        return {
            "score": readability_score,
            "level": level,
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length
        }
