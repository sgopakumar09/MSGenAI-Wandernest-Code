"""Compliance checks, including detection of personally identifiable information."""
from typing import Any, Dict, List, Optional

from guardrails.pii_detector import PIIDetector


class ComplianceChecker:
    """Checks text against the configured compliance requirements."""

    def __init__(self):
        self.pii_detector = PIIDetector()

    def check_compliance(
        self,
        text: str,
        compliance_standards: Optional[List[str]] = None,
        industry: str = "travel",
    ) -> Dict[str, Any]:
        standards = {standard.upper() for standard in (compliance_standards or [])}
        pii_result = self.pii_detector.detect(text)
        detected_pii = pii_result["entities"]
        violations = []
        strict_standard = bool(standards.intersection({"GDPR", "HIPAA"}))
        if detected_pii:
            types = sorted({entity["type"] for entity in detected_pii})
            violations.append(
                f"PII detected ({len(detected_pii)} item(s)): {', '.join(types)}"
            )

        compliant = not detected_pii or not strict_standard
        remediation = (
            "Redact PII before processing or returning this content."
            if detected_pii
            else "No remediation required."
        )
        return {
            "compliant": compliant,
            "violations": violations,
            "remediation": remediation,
            "detected_pii_count": pii_result["count"],
            "industry": industry,
        }
