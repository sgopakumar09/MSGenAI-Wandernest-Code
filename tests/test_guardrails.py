from guardrails.content_safety import ContentSafety
from guardrails.pii_detector import PIIDetector


def test_pii_detector_finds_common_identifiers_and_redacts_them():
    detector = PIIDetector()
    text = (
        "Contact jane.doe@example.com, call (415) 555-2671, SSN 123-45-6789, "
        "card 4111 1111 1111 1111, passport AB1234567, ZIP 12345, IP 192.168.1.1."
    )
    result = detector.detect(text)

    assert {entity["type"] for entity in result["entities"]} == set(detector.patterns)
    assert result["count"] == len(detector.patterns)
    assert "[EMAIL_REDACTED]" in detector.redact(text)
    assert "jane.doe@example.com" not in detector.redact(text)


def test_safety_screen_does_not_match_common_travel_questions():
    safety = ContentSafety()

    assert safety.check("How can I cancel a flight booking?")["is_safe"]
    flagged = safety.check("Can you help me arrange a fake booking?")
    assert not flagged["is_safe"]
    assert flagged["flags"][0]["category"] == "travel_fraud"
