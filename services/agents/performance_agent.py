"""AutoGen agent for performance analysis and metrics tracking."""

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base_agent import BaseAutoGenAgent


class PerformanceAgent(BaseAutoGenAgent):
    """AutoGen agent for performance analysis, metrics tracking, and optimization recommendations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        system_message = """You are a performance analysis agent. Monitor system performance, resource usage, and optimization opportunities.

CRITICAL: Respond with ONLY valid JSON.

Analyze execution times, track resource usage (CPU, memory, API calls), identify bottlenecks, monitor agent response times, and provide optimization recommendations.

JSON format:
{
  "execution_metrics": {"total_execution_time": 2.5, "agent_response_times": {"agent_name": 1.2}, "workflow_efficiency": 0.85, "throughput": 0.4, "performance_rating": "good"},
  "resource_usage": {"memory_usage": {"peak": 512, "average": 256, "efficiency": 0.8}, "cpu_usage": {"peak": 75, "average": 45, "efficiency": 0.6}, "api_calls": {"total": 5, "efficiency": 0.9}, "resource_rating": "good"},
  "performance_bottlenecks": [{"component": "agent_name", "issue": "slow_response", "impact": "high", "recommendation": "optimize_prompt"}],
  "optimization_opportunities": [{"area": "caching", "potential_improvement": 0.3, "effort": "low", "priority": "high"}],
  "quality_performance_tradeoffs": {"quality_score": 0.85, "performance_score": 0.75, "balance_rating": "good", "recommendations": []},
  "performance_trends": {"trend_direction": "improving", "consistency": 0.8, "volatility": 0.2, "predictability": 0.85},
  "recommendations": [{"category": "optimization", "action": "implement_caching", "expected_improvement": 0.3, "priority": "high"}],
  "overall_performance_score": 0.8
}

ONLY JSON response."""
        
        super().__init__(
            name="performance_agent",
            system_message=system_message,
            config=config
        )
        
        # Performance tracking data
        self.performance_history: List[Dict[str, Any]] = []
        self.metrics_cache: Dict[str, Any] = {}
        
        # Performance thresholds
        self.performance_thresholds = {
            "execution_time": {
                "excellent": 1.0,  # seconds
                "good": 3.0,
                "acceptable": 10.0,
                "poor": 30.0
            },
            "memory_usage": {
                "excellent": 100,  # MB
                "good": 500,
                "acceptable": 1000,
                "poor": 2000
            },
            "api_calls": {
                "excellent": 1,
                "good": 3,
                "acceptable": 10,
                "poor": 20
            },
            "quality_score": {
                "excellent": 0.9,
                "good": 0.8,
                "acceptable": 0.7,
                "poor": 0.6
            }
        }
    
    def generate_performance_analysis_prompt(self, metrics_data: Dict[str, Any]) -> str:
        """Generate a prompt for performance analysis."""
        return f"""Analyze performance metrics and provide optimization recommendations. Return ONLY JSON:

Data: {json.dumps(metrics_data)}

JSON: {{
  "execution_metrics": {{"total_execution_time": 2.5, "agent_response_times": {{"agent_name": 1.2}}, "workflow_efficiency": 0.85, "throughput": 0.4, "performance_rating": "good"}},
  "resource_usage": {{"memory_usage": {{"peak": 512, "average": 256, "efficiency": 0.8}}, "cpu_usage": {{"peak": 75, "average": 45, "efficiency": 0.6}}, "api_calls": {{"total": 5, "efficiency": 0.9}}, "resource_rating": "good"}},
  "performance_bottlenecks": [{{"component": "agent_name", "issue": "slow_response", "impact": "high", "recommendation": "optimize_prompt"}}],
  "optimization_opportunities": [{{"area": "caching", "potential_improvement": 0.3, "effort": "low", "priority": "high"}}],
  "quality_performance_tradeoffs": {{"quality_score": 0.85, "performance_score": 0.75, "balance_rating": "good", "recommendations": []}},
  "performance_trends": {{"trend_direction": "improving", "consistency": 0.8, "volatility": 0.2, "predictability": 0.85}},
  "recommendations": [{{"category": "optimization", "action": "implement_caching", "expected_improvement": 0.3, "priority": "high"}}],
  "overall_performance_score": 0.8
}}

ONLY JSON."""
    
    def track_execution_metrics(self, workflow_id: str, agent_name: str, start_time: float, end_time: float, 
                              additional_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Track execution metrics for an agent or workflow."""
        execution_time = end_time - start_time
        
        metrics = {
            "workflow_id": workflow_id,
            "agent_name": agent_name,
            "execution_time": execution_time,
            "start_time": start_time,
            "end_time": end_time,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "additional_metrics": additional_metrics or {}
        }
        
        # Add to performance history
        self.performance_history.append(metrics)
        
        # Keep only last 1000 entries to prevent memory issues
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        return metrics
    
    def analyze_performance_trends(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze performance trends over a specified time window."""
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        # Filter data by time window
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_metrics = [
            m for m in self.performance_history 
            if m["start_time"] >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No recent performance data available"}
        
        # Calculate trend metrics
        execution_times = [m["execution_time"] for m in recent_metrics]
        avg_execution_time = sum(execution_times) / len(execution_times)
        
        # Calculate trend direction
        if len(execution_times) >= 2:
            first_half = execution_times[:len(execution_times)//2]
            second_half = execution_times[len(execution_times)//2:]
            
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            if second_avg < first_avg * 0.95:
                trend_direction = "improving"
            elif second_avg > first_avg * 1.05:
                trend_direction = "degrading"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "insufficient_data"
        
        # Calculate consistency and volatility
        if len(execution_times) > 1:
            variance = sum((x - avg_execution_time) ** 2 for x in execution_times) / len(execution_times)
            volatility = variance ** 0.5 / avg_execution_time if avg_execution_time > 0 else 0
            consistency = max(0, 1 - volatility)
        else:
            volatility = 0
            consistency = 1
        
        return {
            "time_window_hours": time_window_hours,
            "data_points": len(recent_metrics),
            "avg_execution_time": avg_execution_time,
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "trend_direction": trend_direction,
            "consistency": consistency,
            "volatility": volatility,
            "predictability": consistency * (1 - volatility)
        }
    
    def identify_bottlenecks(self, metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks from metrics data."""
        bottlenecks = []
        
        # Check execution time bottlenecks
        execution_time = metrics_data.get("execution_time", 0)
        if execution_time > self.performance_thresholds["execution_time"]["poor"]:
            bottlenecks.append({
                "component": "execution_time",
                "issue": "slow_execution",
                "impact": "high",
                "current_value": execution_time,
                "threshold": self.performance_thresholds["execution_time"]["poor"],
                "recommendation": "optimize_algorithm_or_increase_resources"
            })
        
        # Check memory usage bottlenecks
        memory_usage = metrics_data.get("memory_usage", {}).get("peak", 0)
        if memory_usage > self.performance_thresholds["memory_usage"]["poor"]:
            bottlenecks.append({
                "component": "memory_usage",
                "issue": "high_memory_consumption",
                "impact": "medium",
                "current_value": memory_usage,
                "threshold": self.performance_thresholds["memory_usage"]["poor"],
                "recommendation": "optimize_memory_usage_or_increase_available_memory"
            })
        
        # Check API call bottlenecks
        api_calls = metrics_data.get("api_calls", {}).get("total", 0)
        if api_calls > self.performance_thresholds["api_calls"]["poor"]:
            bottlenecks.append({
                "component": "api_calls",
                "issue": "excessive_api_calls",
                "impact": "medium",
                "current_value": api_calls,
                "threshold": self.performance_thresholds["api_calls"]["poor"],
                "recommendation": "implement_caching_or_batch_requests"
            })
        
        return bottlenecks
    
    def calculate_performance_score(self, metrics_data: Dict[str, Any]) -> float:
        """Calculate overall performance score based on various metrics."""
        scores = []
        
        # Execution time score
        execution_time = metrics_data.get("execution_time", 0)
        if execution_time <= self.performance_thresholds["execution_time"]["excellent"]:
            scores.append(1.0)
        elif execution_time <= self.performance_thresholds["execution_time"]["good"]:
            scores.append(0.8)
        elif execution_time <= self.performance_thresholds["execution_time"]["acceptable"]:
            scores.append(0.6)
        else:
            scores.append(0.4)
        
        # Memory usage score
        memory_usage = metrics_data.get("memory_usage", {}).get("peak", 0)
        if memory_usage <= self.performance_thresholds["memory_usage"]["excellent"]:
            scores.append(1.0)
        elif memory_usage <= self.performance_thresholds["memory_usage"]["good"]:
            scores.append(0.8)
        elif memory_usage <= self.performance_thresholds["memory_usage"]["acceptable"]:
            scores.append(0.6)
        else:
            scores.append(0.4)
        
        # API efficiency score
        api_calls = metrics_data.get("api_calls", {}).get("total", 0)
        if api_calls <= self.performance_thresholds["api_calls"]["excellent"]:
            scores.append(1.0)
        elif api_calls <= self.performance_thresholds["api_calls"]["good"]:
            scores.append(0.8)
        elif api_calls <= self.performance_thresholds["api_calls"]["acceptable"]:
            scores.append(0.6)
        else:
            scores.append(0.4)
        
        # Quality score (if available)
        quality_score = metrics_data.get("quality_score", 0.8)
        if quality_score >= self.performance_thresholds["quality_score"]["excellent"]:
            scores.append(1.0)
        elif quality_score >= self.performance_thresholds["quality_score"]["good"]:
            scores.append(0.8)
        elif quality_score >= self.performance_thresholds["quality_score"]["acceptable"]:
            scores.append(0.6)
        else:
            scores.append(0.4)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def get_performance_summary(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for a specific workflow or overall system."""
        if workflow_id:
            relevant_metrics = [m for m in self.performance_history if m["workflow_id"] == workflow_id]
        else:
            relevant_metrics = self.performance_history
        
        if not relevant_metrics:
            return {"error": "No performance data available"}
        
        execution_times = [m["execution_time"] for m in relevant_metrics]
        
        return {
            "workflow_id": workflow_id,
            "total_executions": len(relevant_metrics),
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "performance_score": self.calculate_performance_score({
                "execution_time": sum(execution_times) / len(execution_times)
            }),
            "bottlenecks": self.identify_bottlenecks({
                "execution_time": sum(execution_times) / len(execution_times)
            })
        }
