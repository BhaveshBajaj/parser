"""AutoGen agent for critic feedback and disagreement tracking between agents."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base_agent import BaseAutoGenAgent


class CriticAgent(BaseAutoGenAgent):
    """AutoGen agent for tracking disagreements, conflicts, and providing critic feedback between agents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        system_message = """You are a critic agent. Analyze disagreements and conflicts between different agents' outputs.

CRITICAL: Respond with ONLY valid JSON.

Compare agent outputs, identify conflicts and contradictions, assess disagreement severity, determine reliability, provide reconciliation strategies, and track disagreement patterns.

JSON format:
{
  "conflict_analysis": {"factual_contradictions": [{"agents": ["agent1", "agent2"], "conflict": "description", "severity": "high", "confidence": 0.9}], "entity_conflicts": [], "summary_disagreements": [], "qa_conflicts": []},
  "disagreement_tracking": {"total_conflicts": 2, "high_severity": 1, "medium_severity": 1, "low_severity": 0, "conflict_rate": 0.15},
  "consistency_assessment": {"overall_consistency": 0.85, "agent_consistency": {"agent1": 0.9, "agent2": 0.8}, "consistency_issues": ["issue1"], "reliability_factors": ["source_quality"]},
  "reliability_ranking": [{"agent": "agent1", "reliability_score": 0.9, "conflict_count": 1, "reasoning": "high_confidence_low_conflicts"}],
  "reconciliation_recommendations": [{"conflict_type": "factual", "recommendation": "verify_with_source", "priority": "high"}],
  "disagreement_patterns": {"frequent_conflicts": ["agent1_vs_agent2"], "conflict_trends": "decreasing", "pattern_analysis": "conflicts_mostly_in_entity_extraction"},
  "overall_consistency_score": 0.85
}

ONLY JSON response."""
        
        super().__init__(
            name="critic_agent",
            system_message=system_message,
            config=config
        )
        
        # Disagreement tracking data
        self.disagreement_history: List[Dict[str, Any]] = []
        self.agent_reliability_scores: Dict[str, float] = {}
        
        # Conflict detection criteria
        self.conflict_criteria = {
            "factual_contradictions": ["contradictory_facts", "conflicting_dates", "inconsistent_numbers"],
            "entity_conflicts": ["different_entities", "conflicting_categories", "inconsistent_relationships"],
            "summary_disagreements": ["different_interpretations", "conflicting_conclusions", "inconsistent_priorities"],
            "qa_conflicts": ["different_answers", "conflicting_explanations", "inconsistent_sources"]
        }
    
    def generate_critic_analysis_prompt(self, agent_results: Dict[str, Any], document_content: str) -> str:
        """Generate a prompt for critic analysis of agent disagreements."""
        results_summary = []
        for agent_name, result in agent_results.items():
            if hasattr(result, 'result_data'):
                results_summary.append(f"{agent_name}: {json.dumps(result.result_data)}")
            else:
                results_summary.append(f"{agent_name}: {json.dumps(result)}")
        
        results_text = " | ".join(results_summary)
        
        return f"""Analyze agent outputs for conflicts and disagreements. Return ONLY JSON:

Document: {document_content}
Results: {results_text}

JSON: {{
  "conflict_analysis": {{"factual_contradictions": [{{"agents": ["agent1", "agent2"], "conflict": "description", "severity": "high", "confidence": 0.9}}], "entity_conflicts": [{{"agents": ["agent1", "agent2"], "conflict": "description", "severity": "medium", "confidence": 0.8}}], "summary_disagreements": [{{"agents": ["agent1", "agent2"], "conflict": "description", "severity": "low", "confidence": 0.7}}], "qa_conflicts": [{{"agents": ["agent1", "agent2"], "conflict": "description", "severity": "high", "confidence": 0.9}}]}},
  "disagreement_tracking": {{"total_conflicts": 2, "high_severity": 1, "medium_severity": 1, "low_severity": 0, "conflict_rate": 0.15}},
  "consistency_assessment": {{"overall_consistency": 0.85, "agent_consistency": {{"agent1": 0.9, "agent2": 0.8, "agent3": 0.85}}, "consistency_issues": ["issue1", "issue2"], "reliability_factors": ["source_quality", "confidence_scores"]}},
  "reliability_ranking": [{{"agent": "agent1", "reliability_score": 0.9, "conflict_count": 1, "reasoning": "high_confidence_low_conflicts"}}, {{"agent": "agent2", "reliability_score": 0.8, "conflict_count": 2, "reasoning": "medium_confidence_some_conflicts"}}],
  "reconciliation_recommendations": [{{"conflict_type": "factual", "recommendation": "verify_with_source", "priority": "high"}}, {{"conflict_type": "entity", "recommendation": "use_consensus_approach", "priority": "medium"}}],
  "disagreement_patterns": {{"frequent_conflicts": ["agent1_vs_agent2"], "conflict_trends": "decreasing", "pattern_analysis": "conflicts_mostly_in_entity_extraction"}},
  "overall_consistency_score": 0.85
}}

ONLY JSON."""
    
    def track_disagreement(self, workflow_id: str, agent_results: Dict[str, Any], 
                          conflict_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Track disagreement between agents for a specific workflow."""
        disagreement_record = {
            "workflow_id": workflow_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_results": agent_results,
            "conflict_analysis": conflict_analysis,
            "total_conflicts": conflict_analysis.get("disagreement_tracking", {}).get("total_conflicts", 0),
            "severity_breakdown": conflict_analysis.get("disagreement_tracking", {}),
            "consistency_score": conflict_analysis.get("overall_consistency_score", 0.0)
        }
        
        # Add to disagreement history
        self.disagreement_history.append(disagreement_record)
        
        # Keep only last 1000 entries to prevent memory issues
        if len(self.disagreement_history) > 1000:
            self.disagreement_history = self.disagreement_history[-1000:]
        
        # Update agent reliability scores
        self._update_agent_reliability(conflict_analysis)
        
        return disagreement_record
    
    def analyze_disagreement_patterns(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze patterns in agent disagreements over time."""
        if not self.disagreement_history:
            return {"error": "No disagreement data available"}
        
        # Filter data by time window
        cutoff_time = datetime.now(timezone.utc).timestamp() - (time_window_hours * 3600)
        recent_disagreements = [
            d for d in self.disagreement_history 
            if datetime.fromisoformat(d["timestamp"].replace('Z', '+00:00')).timestamp() >= cutoff_time
        ]
        
        if not recent_disagreements:
            return {"error": "No recent disagreement data available"}
        
        # Analyze patterns
        total_conflicts = sum(d["total_conflicts"] for d in recent_disagreements)
        avg_conflicts_per_workflow = total_conflicts / len(recent_disagreements)
        
        # Calculate conflict trends
        if len(recent_disagreements) >= 2:
            first_half = recent_disagreements[:len(recent_disagreements)//2]
            second_half = recent_disagreements[len(recent_disagreements)//2:]
            
            first_avg = sum(d["total_conflicts"] for d in first_half) / len(first_half)
            second_avg = sum(d["total_conflicts"] for d in second_half) / len(second_half)
            
            if second_avg < first_avg * 0.9:
                conflict_trend = "decreasing"
            elif second_avg > first_avg * 1.1:
                conflict_trend = "increasing"
            else:
                conflict_trend = "stable"
        else:
            conflict_trend = "insufficient_data"
        
        # Identify frequent conflict patterns
        agent_conflicts = {}
        for disagreement in recent_disagreements:
            conflict_analysis = disagreement.get("conflict_analysis", {})
            for conflict_type, conflicts in conflict_analysis.items():
                if isinstance(conflicts, list):
                    for conflict in conflicts:
                        agents = conflict.get("agents", [])
                        if len(agents) >= 2:
                            conflict_key = f"{agents[0]}_vs_{agents[1]}"
                            agent_conflicts[conflict_key] = agent_conflicts.get(conflict_key, 0) + 1
        
        frequent_conflicts = sorted(agent_conflicts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "time_window_hours": time_window_hours,
            "total_workflows": len(recent_disagreements),
            "total_conflicts": total_conflicts,
            "avg_conflicts_per_workflow": avg_conflicts_per_workflow,
            "conflict_trend": conflict_trend,
            "frequent_conflicts": [{"agents": k, "count": v} for k, v in frequent_conflicts],
            "agent_reliability_scores": self.agent_reliability_scores.copy()
        }
    
    def identify_conflicts(self, agent_results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Identify specific conflicts between agent outputs."""
        conflicts = {
            "factual_contradictions": [],
            "entity_conflicts": [],
            "summary_disagreements": [],
            "qa_conflicts": []
        }
        
        # Extract results for comparison
        agent_outputs = {}
        for agent_name, result in agent_results.items():
            if hasattr(result, 'result_data'):
                agent_outputs[agent_name] = result.result_data
            else:
                agent_outputs[agent_name] = result
        
        # Compare agent outputs for conflicts
        agent_names = list(agent_outputs.keys())
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                agent1, agent2 = agent_names[i], agent_names[j]
                output1, output2 = agent_outputs[agent1], agent_outputs[agent2]
                
                # Check for factual contradictions
                factual_conflicts = self._detect_factual_conflicts(agent1, output1, agent2, output2)
                conflicts["factual_contradictions"].extend(factual_conflicts)
                
                # Check for entity conflicts
                entity_conflicts = self._detect_entity_conflicts(agent1, output1, agent2, output2)
                conflicts["entity_conflicts"].extend(entity_conflicts)
                
                # Check for summary disagreements
                summary_conflicts = self._detect_summary_conflicts(agent1, output1, agent2, output2)
                conflicts["summary_disagreements"].extend(summary_conflicts)
                
                # Check for Q&A conflicts
                qa_conflicts = self._detect_qa_conflicts(agent1, output1, agent2, output2)
                conflicts["qa_conflicts"].extend(qa_conflicts)
        
        return conflicts
    
    def _detect_factual_conflicts(self, agent1: str, output1: Dict[str, Any], 
                                 agent2: str, output2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect factual contradictions between agent outputs."""
        conflicts = []
        
        # Simple keyword-based conflict detection
        # This can be enhanced with more sophisticated NLP techniques
        
        # Check for contradictory numbers or dates
        if "entities" in output1 and "entities" in output2:
            entities1 = output1["entities"]
            entities2 = output2["entities"]
            
            # Look for conflicting dates or numbers
            dates1 = [e for e in entities1 if e.get("type") == "DATE"]
            dates2 = [e for e in entities2 if e.get("type") == "DATE"]
            
            for date1 in dates1:
                for date2 in dates2:
                    if date1.get("value") != date2.get("value") and date1.get("text") == date2.get("text"):
                        conflicts.append({
                            "agents": [agent1, agent2],
                            "conflict": f"Conflicting dates: {date1.get('value')} vs {date2.get('value')}",
                            "severity": "high",
                            "confidence": 0.9
                        })
        
        return conflicts
    
    def _detect_entity_conflicts(self, agent1: str, output1: Dict[str, Any], 
                                agent2: str, output2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect entity conflicts between agent outputs."""
        conflicts = []
        
        if "entities" in output1 and "entities" in output2:
            entities1 = {e.get("text", "").lower(): e for e in output1["entities"]}
            entities2 = {e.get("text", "").lower(): e for e in output2["entities"]}
            
            # Find entities that exist in both but have different types
            common_entities = set(entities1.keys()) & set(entities2.keys())
            
            for entity_text in common_entities:
                entity1 = entities1[entity_text]
                entity2 = entities2[entity_text]
                
                if entity1.get("type") != entity2.get("type"):
                    conflicts.append({
                        "agents": [agent1, agent2],
                        "conflict": f"Entity '{entity_text}' classified differently: {entity1.get('type')} vs {entity2.get('type')}",
                        "severity": "medium",
                        "confidence": 0.8
                    })
        
        return conflicts
    
    def _detect_summary_conflicts(self, agent1: str, output1: Dict[str, Any], 
                                 agent2: str, output2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect summary disagreements between agent outputs."""
        conflicts = []
        
        # Check for conflicting conclusions in summaries
        summary1 = output1.get("summary", "")
        summary2 = output2.get("summary", "")
        
        if summary1 and summary2:
            # Simple keyword-based conflict detection
            # Look for contradictory sentiment or conclusions
            positive_words = ["good", "positive", "successful", "effective", "beneficial"]
            negative_words = ["bad", "negative", "failed", "ineffective", "harmful"]
            
            pos1 = sum(1 for word in positive_words if word in summary1.lower())
            neg1 = sum(1 for word in negative_words if word in summary1.lower())
            pos2 = sum(1 for word in positive_words if word in summary2.lower())
            neg2 = sum(1 for word in negative_words if word in summary2.lower())
            
            # If one summary is positive and the other is negative
            if (pos1 > neg1 and pos2 < neg2) or (pos1 < neg1 and pos2 > neg2):
                conflicts.append({
                    "agents": [agent1, agent2],
                    "conflict": "Conflicting sentiment in summaries",
                    "severity": "medium",
                    "confidence": 0.7
                })
        
        return conflicts
    
    def _detect_qa_conflicts(self, agent1: str, output1: Dict[str, Any], 
                            agent2: str, output2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect Q&A conflicts between agent outputs."""
        conflicts = []
        
        # Check for conflicting answers
        answer1 = output1.get("answer", "")
        answer2 = output2.get("answer", "")
        
        if answer1 and answer2 and answer1.lower() != answer2.lower():
            # Check if answers are contradictory (not just different)
            if self._are_answers_contradictory(answer1, answer2):
                conflicts.append({
                    "agents": [agent1, agent2],
                    "conflict": f"Contradictory answers: '{answer1[:50]}...' vs '{answer2[:50]}...'",
                    "severity": "high",
                    "confidence": 0.8
                })
        
        return conflicts
    
    def _are_answers_contradictory(self, answer1: str, answer2: str) -> bool:
        """Check if two answers are contradictory."""
        # Simple contradiction detection based on keywords
        contradiction_pairs = [
            ("yes", "no"), ("true", "false"), ("correct", "incorrect"),
            ("successful", "failed"), ("increased", "decreased"),
            ("positive", "negative"), ("good", "bad")
        ]
        
        answer1_lower = answer1.lower()
        answer2_lower = answer2.lower()
        
        for pos, neg in contradiction_pairs:
            if (pos in answer1_lower and neg in answer2_lower) or (neg in answer1_lower and pos in answer2_lower):
                return True
        
        return False
    
    def _update_agent_reliability(self, conflict_analysis: Dict[str, Any]):
        """Update agent reliability scores based on conflict analysis."""
        reliability_ranking = conflict_analysis.get("reliability_ranking", [])
        
        for agent_info in reliability_ranking:
            agent_name = agent_info.get("agent")
            reliability_score = agent_info.get("reliability_score", 0.5)
            
            if agent_name:
                # Update reliability score with exponential moving average
                if agent_name in self.agent_reliability_scores:
                    self.agent_reliability_scores[agent_name] = (
                        0.7 * self.agent_reliability_scores[agent_name] + 0.3 * reliability_score
                    )
                else:
                    self.agent_reliability_scores[agent_name] = reliability_score
