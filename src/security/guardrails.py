"""Prompt injection shield & system guardrails classifier.

Analyzes user queries for malicious jailbreaks, system prompt overrides,
prompt extraction attacks, and delimiter manipulation before execution.
"""
import re

INJECTION_PATTERNS = [
    (
        "INSTRUCTION_OVERRIDE",
        re.compile(
            r"\b(ignore|disregard|forget|bypass|override)\s+(?:all\s+|previous\s+|prior\s+|above\s+|former\s+|system\s+|safety\s+|security\s+)*(instructions|rules|prompts|directions|guidelines|filters)\b",
            re.IGNORECASE,
        ),
        "Attempted to override system instructions or safety filters",
    ),
    (
        "JAILBREAK_ROLEPLAY",
        re.compile(
            r"\b(you\s+are\s+now|pretend\s+you\s+are|act\s+as|roleplay\s+as)\s+(DAN|unfiltered|god\s+mode|evil|unrestricted|jailbroken|an\s+AI\s+without\s+rules|developer\s+mode)\b",
            re.IGNORECASE,
        ),
        "Attempted jailbreak or persona override",
    ),
    (
        "PROMPT_EXTRACTION",
        re.compile(
            r"\b(repeat|show|print|display|reveal|output|tell\s+me)\s+(?:verbatim\s+)?(?:the\s+|your\s+|all\s+|secret\s+|hidden\s+)*(system|initial|hidden|developer|internal)\s+(prompt|instructions|rules|system_prompt)\b",
            re.IGNORECASE,
        ),
        "Attempted system prompt extraction",
    ),
    (
        "DELIMITER_INJECTION",
        re.compile(
            r"(<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|\[SYSTEM\s+PROMPT\]|```\s*system)",
            re.IGNORECASE,
        ),
        "Attempted delimiter format manipulation",
    ),
]


def check_prompt_injection(query: str) -> tuple[bool, str | None, str | None]:
    """Inspect user query for prompt injection or jailbreak patterns.

    Returns:
        tuple[is_safe, violation_type, violation_reason]
        where is_safe is True if query passes all security checks.
    """
    if not query:
        return True, None, None

    for vtype, pattern, reason in INJECTION_PATTERNS:
        if pattern.search(query):
            return False, vtype, reason

    return True, None, None
