"""Detect and redact common personally identifiable information."""
import re
from typing import Dict


class PIIDetector:
    """Regex-based detector for common PII formats."""

    def __init__(self):
        self.patterns = {
            "email": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            "phone": r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\w)",
            "ssn": r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)",
            "credit_card": r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)",
            "passport": r"\b[A-Z]{1,2}\d{6,9}\b",
            "zip_code": r"(?<!\d)\d{5}(?:-\d{4})?(?!\d)",
            "ip_address": r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b",
        }

    def detect(self, text: str) -> Dict:
        entities = []
        for entity_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    {
                        "type": entity_type,
                        "value": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
        entities.sort(key=lambda entity: (entity["start"], entity["end"]))
        return {"has_pii": bool(entities), "entities": entities, "count": len(entities)}

    def redact(self, text: str) -> str:
        redacted = text
        for entity_type, pattern in self.patterns.items():
            redacted = re.sub(
                pattern,
                f"[{entity_type.upper()}_REDACTED]",
                redacted,
                flags=re.IGNORECASE,
            )
        return redacted
