"""AutoGen agent for security review including sensitive data detection and compliance checks."""

import json
import re
from typing import Any, Dict, List, Optional

from .base_agent import BaseAutoGenAgent


class SecurityAgent(BaseAutoGenAgent):
    """AutoGen agent for security review, sensitive data detection, and compliance analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        system_message = """You are a security review agent. Detect sensitive data, check compliance, and identify security risks.

CRITICAL: Respond with ONLY valid JSON.

Scan for PII (SSN, credit cards, emails, phones), identify sensitive business data, check GDPR/CCPA/HIPAA compliance, assess security risks, and provide remediation recommendations.

JSON format:
{
  "pii_detection": {"ssn": {"detected": false, "count": 0, "examples": [], "risk_level": "none"}, ...},
  "sensitive_data": {"financial_data": {"detected": false, "count": 0, "examples": [], "risk_level": "medium"}, ...},
  "compliance_check": {"gdpr": {"compliant": true, "violations": [], "risk_level": "low"}, ...},
  "security_risks": {"data_exposure": {"risk_level": "low", "issues": [], "recommendations": []}, ...},
  "confidential_info": {"classified": {"detected": false, "level": "none", "examples": []}, ...},
  "risk_assessment": {"overall_risk": "low", "risk_factors": [], "mitigation_needed": false},
  "recommendations": [{"category": "pii", "issue": "description", "recommendation": "fix", "priority": "medium"}],
  "overall_security_score": 0.95
}

ONLY JSON response."""
        
        super().__init__(
            name="security_agent",
            system_message=system_message,
            config=config
        )
        
        # PII detection patterns
        self.pii_patterns = {
            "ssn": r"\b\d{3}-?\d{2}-?\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b",
            "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
            "mac_address": r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b",
            "driver_license": r"\b[A-Z]{1,2}\d{6,8}\b",
            "passport": r"\b[A-Z]{1,2}\d{6,9}\b"
        }
        
        # Sensitive business information patterns
        self.business_sensitive_patterns = {
            "financial_data": [
                r"\$[\d,]+\.?\d*",
                r"\b(?:revenue|profit|loss|income|expense|budget|cost)\b",
                r"\b(?:salary|wage|compensation|bonus|stock|equity)\b"
            ],
            "customer_data": [
                r"\b(?:customer|client|user|account)\s+(?:id|number|name|email|phone)\b",
                r"\b(?:database|record|profile|information)\b"
            ],
            "trade_secrets": [
                r"\b(?:proprietary|confidential|secret|internal|private)\b",
                r"\b(?:algorithm|formula|process|method|technique)\b"
            ],
            "legal_info": [
                r"\b(?:lawsuit|litigation|settlement|agreement|contract)\b",
                r"\b(?:patent|trademark|copyright|intellectual property)\b"
            ]
        }
        
        # Compliance frameworks
        self.compliance_frameworks = {
            "gdpr": {
                "data_types": ["personal_data", "sensitive_personal_data", "biometric_data"],
                "rights": ["access", "rectification", "erasure", "portability", "objection"],
                "principles": ["lawfulness", "fairness", "transparency", "purpose_limitation"]
            },
            "ccpa": {
                "data_types": ["personal_information", "sensitive_personal_information"],
                "rights": ["know", "delete", "opt_out", "non_discrimination"],
                "categories": ["identifiers", "commercial", "biometric", "internet_activity"]
            },
            "hipaa": {
                "data_types": ["phi", "health_information", "medical_records"],
                "safeguards": ["administrative", "physical", "technical"],
                "requirements": ["minimum_necessary", "access_controls", "audit_logs"]
            }
        }
    
    def generate_security_review_prompt(self, content: str, content_type: str = "general") -> str:
        """Generate a prompt for security review analysis."""
        return f"""Review content for security risks, sensitive data, compliance. Return ONLY JSON:

Content: {content}

JSON: {{
  "pii_detection": {{"ssn": {{"detected": false, "count": 0, "examples": [], "risk_level": "none"}}, "credit_card": {{"detected": false, "count": 0, "examples": [], "risk_level": "none"}}, "email": {{"detected": false, "count": 0, "examples": [], "risk_level": "low"}}, "phone": {{"detected": false, "count": 0, "examples": [], "risk_level": "low"}}, "ip_address": {{"detected": false, "count": 0, "examples": [], "risk_level": "medium"}}, "other_pii": {{"detected": false, "count": 0, "examples": [], "risk_level": "none"}}, "overall_pii_risk": "low"}},
  "sensitive_data": {{"financial_data": {{"detected": false, "count": 0, "examples": [], "risk_level": "medium"}}, "customer_data": {{"detected": false, "count": 0, "examples": [], "risk_level": "high"}}, "trade_secrets": {{"detected": false, "count": 0, "examples": [], "risk_level": "high"}}, "legal_info": {{"detected": false, "count": 0, "examples": [], "risk_level": "medium"}}, "overall_sensitivity": "low"}},
  "compliance_check": {{"gdpr": {{"compliant": true, "violations": [], "risk_level": "low"}}, "ccpa": {{"compliant": true, "violations": [], "risk_level": "low"}}, "hipaa": {{"compliant": true, "violations": [], "risk_level": "low"}}, "overall_compliance": "compliant"}},
  "security_risks": {{"data_exposure": {{"risk_level": "low", "issues": [], "recommendations": []}}, "access_control": {{"risk_level": "low", "issues": [], "recommendations": []}}, "encryption": {{"risk_level": "low", "issues": [], "recommendations": []}}, "overall_security_risk": "low"}},
  "confidential_info": {{"classified": {{"detected": false, "level": "none", "examples": []}}, "confidential": {{"detected": false, "level": "none", "examples": []}}, "internal": {{"detected": false, "level": "none", "examples": []}}, "public": {{"detected": true, "level": "public", "examples": []}}}},
  "risk_assessment": {{"overall_risk": "low", "risk_factors": [], "mitigation_needed": false, "immediate_actions": []}},
  "recommendations": [{{"category": "pii", "issue": "issue_description", "recommendation": "remediation_step", "priority": "medium"}}],
  "overall_security_score": 0.95
}}

ONLY JSON."""
    
    def detect_pii(self, content: str) -> Dict[str, Any]:
        """Detect PII patterns in content."""
        pii_results = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Mask sensitive data in examples for security
                masked_examples = []
                for match in matches[:3]:  # Limit to 3 examples
                    if pii_type in ["ssn", "credit_card"]:
                        masked_examples.append(self._mask_sensitive_data(match, pii_type))
                    else:
                        masked_examples.append(match)
                
                pii_results[pii_type] = {
                    "detected": True,
                    "count": len(matches),
                    "examples": masked_examples,
                    "risk_level": self._get_pii_risk_level(pii_type)
                }
            else:
                pii_results[pii_type] = {
                    "detected": False,
                    "count": 0,
                    "examples": [],
                    "risk_level": "none"
                }
        
        return pii_results
    
    def detect_sensitive_business_data(self, content: str) -> Dict[str, Any]:
        """Detect sensitive business information in content."""
        sensitive_results = {}
        
        for data_type, patterns in self.business_sensitive_patterns.items():
            detected_examples = []
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                detected_examples.extend(matches)
            
            if detected_examples:
                sensitive_results[data_type] = {
                    "detected": True,
                    "count": len(detected_examples),
                    "examples": detected_examples[:3],  # Limit examples
                    "risk_level": self._get_business_data_risk_level(data_type)
                }
            else:
                sensitive_results[data_type] = {
                    "detected": False,
                    "count": 0,
                    "examples": [],
                    "risk_level": "none"
                }
        
        return sensitive_results
    
    def assess_compliance(self, content: str, frameworks: Optional[List[str]] = None) -> Dict[str, Any]:
        """Assess compliance with various frameworks."""
        if not frameworks:
            frameworks = ["gdpr", "ccpa", "hipaa"]
        
        compliance_results = {}
        
        for framework in frameworks:
            if framework in self.compliance_frameworks:
                framework_info = self.compliance_frameworks[framework]
                violations = []
                
                # Simple keyword-based compliance check
                for principle in framework_info.get("principles", []):
                    if principle not in content.lower():
                        violations.append(f"Missing {principle} consideration")
                
                compliance_results[framework] = {
                    "compliant": len(violations) == 0,
                    "violations": violations,
                    "risk_level": "high" if len(violations) > 2 else "medium" if len(violations) > 0 else "low"
                }
        
        return compliance_results
    
    def _mask_sensitive_data(self, data: str, data_type: str) -> str:
        """Mask sensitive data for security."""
        if data_type == "ssn":
            return "XXX-XX-" + data[-4:] if len(data) >= 4 else "XXX-XX-XXXX"
        elif data_type == "credit_card":
            return "XXXX-XXXX-XXXX-" + data[-4:] if len(data) >= 4 else "XXXX-XXXX-XXXX-XXXX"
        else:
            return data
    
    def _get_pii_risk_level(self, pii_type: str) -> str:
        """Get risk level for PII type."""
        high_risk = ["ssn", "credit_card", "driver_license", "passport"]
        medium_risk = ["ip_address", "mac_address"]
        low_risk = ["email", "phone"]
        
        if pii_type in high_risk:
            return "high"
        elif pii_type in medium_risk:
            return "medium"
        elif pii_type in low_risk:
            return "low"
        else:
            return "none"
    
    def _get_business_data_risk_level(self, data_type: str) -> str:
        """Get risk level for business data type."""
        high_risk = ["customer_data", "trade_secrets"]
        medium_risk = ["financial_data", "legal_info"]
        
        if data_type in high_risk:
            return "high"
        elif data_type in medium_risk:
            return "medium"
        else:
            return "low"
