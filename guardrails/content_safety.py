"""Local keyword-based safety checks used alongside Azure Content Safety."""
import re
from typing import Dict


class ContentSafety:
    """Conservative local screening for unsafe or fraudulent requests."""

    def __init__(self):
        self.unsafe_patterns = {
            "violence": ["kill someone", "murder", "shoot someone", "bomb threat", "attack someone"],
            "hate_speech": ["racial slur", "ethnic cleansing", "genocide"],
            "profanity": ["fuck", "shit", "bitch", "asshole"],
            "personal_attack": ["you are worthless", "you are an idiot", "go kill yourself"],
        }
        self.travel_red_flags = ["fake booking", "travel document fraud", "evade airport security"]

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None

    def check(self, text: str) -> Dict:
        text_lower = text.lower()
        flags = []
        high_severity = {"violence", "hate_speech"}
        for category, keywords in self.unsafe_patterns.items():
            for keyword in keywords:
                if self._contains_phrase(text_lower, keyword):
                    severity = "high" if category in high_severity else (
                        "medium" if category == "profanity" else "low"
                    )
                    flags.append(
                        {"category": category, "keyword": keyword, "severity": severity}
                    )
        for phrase in self.travel_red_flags:
            if self._contains_phrase(text_lower, phrase):
                flags.append(
                    {"category": "travel_fraud", "keyword": phrase, "severity": "high"}
                )
        severity = max(
            (flag["severity"] for flag in flags),
            key={"low": 1, "medium": 2, "high": 3}.get,
            default="none",
        )
        return {"is_safe": not flags, "flags": flags, "severity": severity}

    def get_safety_score(self, text: str) -> float:
        result = self.check(text)
        if result["is_safe"]:
            return 1.0
        return max(0.0, 1.0 - 0.2 * len(result["flags"]))
