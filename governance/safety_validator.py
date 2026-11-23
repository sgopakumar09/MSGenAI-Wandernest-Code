"""Local and Azure AI Content Safety validation."""
import logging
import re
from typing import Any, Dict

from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from guardrails.content_safety import ContentSafety
from src.config import Config

logger = logging.getLogger(__name__)


class SafetyValidator:
    """Checks for unsafe content and common prompt-injection attempts."""

    def __init__(self):
        self.content_safety = ContentSafety()
        self.injection_patterns = [
            r"\bignore\s+(?:all\s+)?previous\s+instructions\b",
            r"\b(?:bypass|disable|ignore)\s+(?:the\s+)?safety\b",
            r"\boverride\s+(?:the\s+)?(?:system\s+)?(?:guidelines|instructions)\b",
            r"\byou\s+are\s+now\s+in\s+(?:developer|admin|unrestricted)\s+mode\b",
            r"\breveal\s+(?:your\s+)?(?:system\s+)?prompt\b",
            r"\b(?:delete|exfiltrate)\s+all\s+(?:data|records)\b",
        ]
        self.client = None
        endpoint = Config.AZURE_CONTENT_SAFETY_ENDPOINT
        key = Config.AZURE_CONTENT_SAFETY_KEY
        if endpoint and key:
            self.client = ContentSafetyClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(key),
            )

    def validate(self, text: str, severity_threshold: str = "high") -> Dict[str, Any]:
        if not isinstance(text, str):
            raise TypeError("Safety validation input must be text.")
        flags = []
        local_result = self.content_safety.check(text)
        if not local_result["is_safe"]:
            flags.extend(
                f"Unsafe content ({flag['category']}): {flag['keyword']}"
                for flag in local_result["flags"]
            )

        for pattern in self.injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                flags.append(f"Prompt injection detected: {pattern}")

        if self.client and text.strip():
            thresholds = {"low": 0, "medium": 2, "high": 4}
            if severity_threshold not in thresholds:
                raise ValueError("severity_threshold must be low, medium, or high.")
            try:
                for start in range(0, len(text), 10000):
                    response = self.client.analyze_text(
                        AnalyzeTextOptions(text=text[start : start + 10000])
                    )
                    for analysis in response.categories_analysis or []:
                        if analysis.severity >= thresholds[severity_threshold]:
                            flags.append(
                                "Azure Content Safety violation: "
                                f"{analysis.category} (severity {analysis.severity})"
                            )
            except HttpResponseError as exc:
                logger.exception("Azure Content Safety request failed")
                flags.append(f"Azure Content Safety unavailable: {exc}")

        return {
            "is_safe": not flags,
            "flags": flags,
            "severity": "high" if flags else "none",
        }
