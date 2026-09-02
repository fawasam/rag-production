"""PII (Personally Identifiable Information) detection & redaction engine.

Detects and masks sensitive information (Emails, Phone Numbers, SSNs, Credit Cards,
IP Addresses) in user queries before passing text to embeddings or OpenAI LLMs.
"""
import re

PATTERNS = [
    (
        "EMAIL",
        re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
        "[EMAIL]",
    ),
    (
        "PHONE_NUMBER",
        re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "[PHONE_NUMBER]",
    ),
    (
        "SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[SSN]",
    ),
    (
        "CREDIT_CARD",
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        "[CREDIT_CARD]",
    ),
    (
        "IP_ADDRESS",
        re.compile(r"\b(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"),
        "[IP_ADDRESS]",
    ),
]


def mask_pii(text: str) -> tuple[str, list[dict]]:
    """Scan text for PII entities and replace them with placeholder tokens.

    Returns:
        tuple[masked_text, detected_entities]
        where detected_entities is a list of dicts:
        [{'type': 'EMAIL', 'original': 'user@foo.com', 'placeholder': '[EMAIL]'}]
    """
    if not text:
        return text, []

    masked_text = text
    detected_entities: list[dict] = []

    for ptype, pattern, placeholder in PATTERNS:
        matches = pattern.finditer(masked_text)
        for match in matches:
            original_val = match.group(0)
            # Avoid re-masking already masked placeholders or credit card false positives on short numbers
            if ptype == "CREDIT_CARD" and len(re.sub(r"\D", "", original_val)) < 13:
                continue

            detected_entities.append(
                {
                    "type": ptype,
                    "original": original_val,
                    "placeholder": placeholder,
                }
            )

        masked_text = pattern.sub(placeholder, masked_text)

    return masked_text, detected_entities
