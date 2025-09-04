"""Metrics storage and tracking system for performance and critic feedback data."""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pathlib import Path


class MetricsStorage:
    """Storage system for performance metrics and critic feedback data."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize metrics storage with optional custom path."""
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            # Default to data/metrics directory
            self.storage_path = Path("data/metrics")
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # File paths for different metric types
        self.performance_file = self.storage_path / "performance_metrics.json"
        self.critic_file = self.storage_path / "critic_feedback.json"
        self.content_review_file = self.storage_path / "content_review_metrics.json"
        self.security_file = self.storage_path / "security_metrics.json"
        
        # Initialize files if they don't exist
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialize metric files with empty structures if they don't exist."""
        files_to_init = [
            (self.performance_file, {"metrics": [], "summary": {}}),
            (self.critic_file, {"disagreements": [], "patterns": {}}),
            (self.content_review_file, {"reviews": [], "bias_analysis": {}}),
            (self.security_file, {"security_checks": [], "compliance": {}})
        ]
        
        for file_path, default_data in files_to_init:
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    json.dump(default_data, f, indent=2)
    
    def store_performance_metrics(self, workflow_id: str, metrics: Dict[str, Any]) -> bool:
        """Store performance metrics for a workflow."""
        try:
            # Load existing data
            with open(self.performance_file, 'r') as f:
                data = json.load(f)
            
            # Add new metrics entry
            metrics_entry = {
                "workflow_id": workflow_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics
            }
            
            data["metrics"].append(metrics_entry)
            
            # Keep only last 1000 entries to prevent file from growing too large
            if len(data["metrics"]) > 1000:
                data["metrics"] = data["metrics"][-1000:]
            
            # Update summary statistics
            data["summary"] = self._calculate_performance_summary(data["metrics"])
            
            # Save updated data
            with open(self.performance_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error storing performance metrics: {e}")
            return False
    
    def store_critic_feedback(self, workflow_id: str, feedback: Dict[str, Any]) -> bool:
        """Store critic feedback and disagreement data."""
        try:
            # Load existing data
            with open(self.critic_file, 'r') as f:
                data = json.load(f)
            
            # Add new feedback entry
            feedback_entry = {
                "workflow_id": workflow_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "feedback": feedback
            }
            
            data["disagreements"].append(feedback_entry)
            
            # Keep only last 1000 entries
            if len(data["disagreements"]) > 1000:
                data["disagreements"] = data["disagreements"][-1000:]
            
            # Update pattern analysis
            data["patterns"] = self._analyze_disagreement_patterns(data["disagreements"])
            
            # Save updated data
            with open(self.critic_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error storing critic feedback: {e}")
            return False
    
    def store_content_review_metrics(self, workflow_id: str, review_data: Dict[str, Any]) -> bool:
        """Store content review metrics including bias analysis."""
        try:
            # Load existing data
            with open(self.content_review_file, 'r') as f:
                data = json.load(f)
            
            # Add new review entry
            review_entry = {
                "workflow_id": workflow_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "review_data": review_data
            }
            
            data["reviews"].append(review_entry)
            
            # Keep only last 1000 entries
            if len(data["reviews"]) > 1000:
                data["reviews"] = data["reviews"][-1000:]
            
            # Update bias analysis summary
            data["bias_analysis"] = self._analyze_bias_patterns(data["reviews"])
            
            # Save updated data
            with open(self.content_review_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error storing content review metrics: {e}")
            return False
    
    def store_security_metrics(self, workflow_id: str, security_data: Dict[str, Any]) -> bool:
        """Store security review metrics and compliance data."""
        try:
            # Load existing data
            with open(self.security_file, 'r') as f:
                data = json.load(f)
            
            # Add new security check entry
            security_entry = {
                "workflow_id": workflow_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "security_data": security_data
            }
            
            data["security_checks"].append(security_entry)
            
            # Keep only last 1000 entries
            if len(data["security_checks"]) > 1000:
                data["security_checks"] = data["security_checks"][-1000:]
            
            # Update compliance summary
            data["compliance"] = self._analyze_compliance_patterns(data["security_checks"])
            
            # Save updated data
            with open(self.security_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error storing security metrics: {e}")
            return False
    
    def get_performance_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the specified time window."""
        try:
            with open(self.performance_file, 'r') as f:
                data = json.load(f)
            
            # Filter by time window
            cutoff_time = datetime.now(timezone.utc).timestamp() - (time_window_hours * 3600)
            recent_metrics = [
                m for m in data["metrics"]
                if datetime.fromisoformat(m["timestamp"].replace('Z', '+00:00')).timestamp() >= cutoff_time
            ]
            
            if not recent_metrics:
                return {"error": "No recent performance data available"}
            
            # Calculate summary statistics
            execution_times = [m["metrics"].get("execution_time", 0) for m in recent_metrics]
            
            return {
                "time_window_hours": time_window_hours,
                "total_workflows": len(recent_metrics),
                "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
                "min_execution_time": min(execution_times) if execution_times else 0,
                "max_execution_time": max(execution_times) if execution_times else 0,
                "performance_trend": self._calculate_performance_trend(recent_metrics)
            }
            
        except Exception as e:
            return {"error": f"Error retrieving performance summary: {e}"}
    
    def get_critic_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get critic feedback summary for the specified time window."""
        try:
            with open(self.critic_file, 'r') as f:
                data = json.load(f)
            
            # Filter by time window
            cutoff_time = datetime.now(timezone.utc).timestamp() - (time_window_hours * 3600)
            recent_feedback = [
                f for f in data["disagreements"]
                if datetime.fromisoformat(f["timestamp"].replace('Z', '+00:00')).timestamp() >= cutoff_time
            ]
            
            if not recent_feedback:
                return {"error": "No recent critic feedback available"}
            
            # Calculate summary statistics
            total_conflicts = sum(f["feedback"].get("total_conflicts", 0) for f in recent_feedback)
            
            return {
                "time_window_hours": time_window_hours,
                "total_workflows": len(recent_feedback),
                "total_conflicts": total_conflicts,
                "avg_conflicts_per_workflow": total_conflicts / len(recent_feedback) if recent_feedback else 0,
                "conflict_trend": self._calculate_conflict_trend(recent_feedback),
                "frequent_conflicts": data.get("patterns", {}).get("frequent_conflicts", [])
            }
            
        except Exception as e:
            return {"error": f"Error retrieving critic summary: {e}"}
    
    def _calculate_performance_summary(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance summary from metrics data."""
        if not metrics:
            return {}
        
        execution_times = [m["metrics"].get("execution_time", 0) for m in metrics]
        
        return {
            "total_workflows": len(metrics),
            "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
            "min_execution_time": min(execution_times) if execution_times else 0,
            "max_execution_time": max(execution_times) if execution_times else 0,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    def _analyze_disagreement_patterns(self, disagreements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in disagreement data."""
        if not disagreements:
            return {}
        
        # Count frequent conflicts
        agent_conflicts = {}
        for disagreement in disagreements:
            feedback = disagreement.get("feedback", {})
            conflict_analysis = feedback.get("conflict_analysis", {})
            
            for conflict_type, conflicts in conflict_analysis.items():
                if isinstance(conflicts, list):
                    for conflict in conflicts:
                        agents = conflict.get("agents", [])
                        if len(agents) >= 2:
                            conflict_key = f"{agents[0]}_vs_{agents[1]}"
                            agent_conflicts[conflict_key] = agent_conflicts.get(conflict_key, 0) + 1
        
        frequent_conflicts = sorted(agent_conflicts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "frequent_conflicts": [{"agents": k, "count": v} for k, v in frequent_conflicts],
            "total_disagreements": len(disagreements),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    def _analyze_bias_patterns(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze bias patterns from content reviews."""
        if not reviews:
            return {}
        
        bias_counts = {}
        for review in reviews:
            review_data = review.get("review_data", {})
            bias_analysis = review_data.get("bias_analysis", {})
            
            for bias_type, bias_info in bias_analysis.items():
                if isinstance(bias_info, dict) and bias_info.get("detected"):
                    bias_counts[bias_type] = bias_counts.get(bias_type, 0) + 1
        
        return {
            "bias_detection_counts": bias_counts,
            "total_reviews": len(reviews),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    def _analyze_compliance_patterns(self, security_checks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze compliance patterns from security checks."""
        if not security_checks:
            return {}
        
        compliance_violations = {}
        for check in security_checks:
            security_data = check.get("security_data", {})
            compliance_check = security_data.get("compliance_check", {})
            
            for framework, compliance_info in compliance_check.items():
                if isinstance(compliance_info, dict) and not compliance_info.get("compliant", True):
                    compliance_violations[framework] = compliance_violations.get(framework, 0) + 1
        
        return {
            "compliance_violations": compliance_violations,
            "total_security_checks": len(security_checks),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    def _calculate_performance_trend(self, recent_metrics: List[Dict[str, Any]]) -> str:
        """Calculate performance trend from recent metrics."""
        if len(recent_metrics) < 2:
            return "insufficient_data"
        
        # Compare first half vs second half
        first_half = recent_metrics[:len(recent_metrics)//2]
        second_half = recent_metrics[len(recent_metrics)//2:]
        
        first_avg = sum(m["metrics"].get("execution_time", 0) for m in first_half) / len(first_half)
        second_avg = sum(m["metrics"].get("execution_time", 0) for m in second_half) / len(second_half)
        
        if second_avg < first_avg * 0.95:
            return "improving"
        elif second_avg > first_avg * 1.05:
            return "degrading"
        else:
            return "stable"
    
    def _calculate_conflict_trend(self, recent_feedback: List[Dict[str, Any]]) -> str:
        """Calculate conflict trend from recent feedback."""
        if len(recent_feedback) < 2:
            return "insufficient_data"
        
        # Compare first half vs second half
        first_half = recent_feedback[:len(recent_feedback)//2]
        second_half = recent_feedback[len(recent_feedback)//2:]
        
        first_avg = sum(f["feedback"].get("total_conflicts", 0) for f in first_half) / len(first_half)
        second_avg = sum(f["feedback"].get("total_conflicts", 0) for f in second_half) / len(second_half)
        
        if second_avg < first_avg * 0.9:
            return "decreasing"
        elif second_avg > first_avg * 1.1:
            return "increasing"
        else:
            return "stable"
