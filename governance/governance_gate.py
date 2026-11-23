"""Orchestrate safety and compliance validation with an in-memory audit trail."""
from datetime import datetime, timezone
from typing import Any, Dict

from governance.compliance_checker import ComplianceChecker
from governance.safety_validator import SafetyValidator


class GovernanceGate:
    """Coordinates input and output guardrail checks."""

    def __init__(self):
        self.safety_validator = SafetyValidator()
        self.compliance_checker = ComplianceChecker()
        self.audit_log = []

    def _validate(self, action: str, text: str) -> Dict[str, Any]:
        safety_result = self.safety_validator.validate(text)
        compliance_result = self.compliance_checker.check_compliance(
            text, compliance_standards=["GDPR"]
        )
        violations = safety_result["flags"] + compliance_result["violations"]
        result = {
            "passed": safety_result["is_safe"] and compliance_result["compliant"],
            "violations": violations,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._log_audit(action, result)
        return result

    def validate_input(self, text: str) -> Dict[str, Any]:
        return self._validate("validate_input", text)

    def validate_output(self, text: str) -> Dict[str, Any]:
        return self._validate("validate_output", text)

    def get_audit_log(self):
        return list(self.audit_log)

    def _log_audit(self, action: str, result: Dict[str, Any]):
        self.audit_log.append(
            {
                "action": action,
                "result": "PASS" if result["passed"] else "FAIL",
                "details": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
