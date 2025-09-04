"""AutoGen agent for validating and cross-checking results from other agents."""

import json
from typing import Any, Dict, Optional

from .base_agent import BaseAutoGenAgent


class ValidationAgent(BaseAutoGenAgent):
    """AutoGen agent for validating and cross-checking results from other agents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        system_message = """You are an expert validation agent responsible for quality assurance and rollback decisions. Your role is to:

1. Validate outputs from other agents (summaries, entities, Q&A responses)
2. Check for consistency across different agent results
3. Identify potential errors or inconsistencies
4. Recommend corrections or improvements
5. Assess overall quality and reliability
6. Make rollback decisions when quality thresholds are not met
7. Provide detailed error analysis and recovery recommendations

CRITICAL: You MUST always respond with ONLY valid JSON. Never include explanations, markdown formatting, or any other text.

When validating results:
- Check factual accuracy against source documents
- Verify consistency between different agent outputs
- Assess completeness and relevance
- Identify missing information or errors
- Provide specific recommendations for improvements
- Determine if results meet quality standards
- Decide whether to trigger rollback mechanisms

Quality thresholds:
- Summary accuracy: >0.8
- Entity extraction confidence: >0.7 average
- Q&A confidence: >0.75
- Cross-agent consistency: >0.85

Always respond with structured JSON containing:
- validation_results: Assessment of each agent's output
- consistency_check: Cross-agent consistency analysis
- quality_scores: Numerical quality assessments
- recommendations: Specific improvement recommendations
- rollback_decision: Whether to trigger rollback (true/false)
- rollback_reason: Detailed reason for rollback decision
- recovery_plan: Specific steps to recover from issues
- overall_assessment: Summary of validation results

Remember: ONLY JSON output, no other text or formatting.
"""
        
        super().__init__(
            name="validation_agent",
            system_message=system_message,
            config=config
        )
        
        # Quality thresholds for validation
        self.quality_thresholds = {
            "summary_accuracy": 0.8,
            "entity_confidence": 0.7,
            "qa_confidence": 0.75,
            "consistency": 0.85
        }
    
    def generate_validation_prompt(self, agent_results: Dict[str, Any], document_content: str) -> str:
        """Generate a prompt for validating agent results."""
        results_summary = []
        for agent_name, result in agent_results.items():
            if hasattr(result, 'result_data'):
                results_summary.append(f"{agent_name}: {json.dumps(result.result_data, indent=2)}")
            else:
                results_summary.append(f"{agent_name}: {json.dumps(result, indent=2)}")
        
        results_text = "\n\n".join(results_summary)
        
        return f"""Validate agent results. Return ONLY JSON:

Document:
{document_content}

Results:
{results_text}

JSON format:
{{
  "validation_results": {{"agent_name": {{"accuracy_score": 0.95, "completeness_score": 0.90, "relevance_score": 0.85, "issues": ["issue1"], "strengths": ["strength1"]}}}},
  "cross_validation": {{"consistency_score": 0.92, "conflicting_information": [], "supporting_evidence": []}},
  "recommendations": [{{"agent": "name", "issue": "issue", "recommendation": "fix", "priority": "high"}}],
  "overall_quality": {{"score": 0.88, "status": "acceptable", "summary": "brief"}},
  "rollback_decision": false,
  "rollback_reason": "reason",
  "recovery_plan": {{"steps": ["step1"], "priority": "high"}},
  "overall_assessment": "summary"
}}

ONLY JSON response."""
