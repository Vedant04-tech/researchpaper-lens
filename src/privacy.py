"""ResearchPaper Lens — Privacy & Redaction Module.

Implements Privacy-by-Default policies:
- Suppresses / redacts email addresses, credentials/API keys, phone numbers, and IP/location identifiers.
- Deterministic, zero-network, regex-based privacy filtering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Regular expression patterns for common sensitive personal & credential entities
EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
PHONE_REGEX = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
API_KEY_REGEX = re.compile(
    r"\b(?:sk-[a-zA-Z0-9]{20,}|hf_[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|AIza[0-9A-Za-z-_]{35})\b"
)
IP_ADDR_REGEX = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)


@dataclass(frozen=True)
class PrivacyAudit:
    """Summary of privacy redactions performed."""

    sanitized_text: str
    total_redactions: int
    emails_redacted: int
    keys_redacted: int
    phones_redacted: int
    ips_redacted: int


def redact_sensitive_text(text: str, mask_ips: bool = False) -> PrivacyAudit:
    """Applies privacy-by-default redaction to arbitrary text content.

    Args:
        text: Raw document or summary string.
        mask_ips: Whether to redact IPv4 addresses (default False to avoid masking section numbers like 1.2.3.4).

    Returns:
        PrivacyAudit containing redacted text and counts of masked entities.
    """
    if not text:
        return PrivacyAudit(
            sanitized_text="",
            total_redactions=0,
            emails_redacted=0,
            keys_redacted=0,
            phones_redacted=0,
            ips_redacted=0,
        )

    emails_found = len(EMAIL_REGEX.findall(text))
    text_masked = EMAIL_REGEX.sub("[EMAIL REDACTED]", text)

    keys_found = len(API_KEY_REGEX.findall(text_masked))
    text_masked = API_KEY_REGEX.sub("[KEY REDACTED]", text_masked)

    phones_found = len(PHONE_REGEX.findall(text_masked))
    text_masked = PHONE_REGEX.sub("[PHONE REDACTED]", text_masked)

    ips_found = 0
    if mask_ips:
        ips_found = len(IP_ADDR_REGEX.findall(text_masked))
        text_masked = IP_ADDR_REGEX.sub("[IP REDACTED]", text_masked)

    total = emails_found + keys_found + phones_found + ips_found

    return PrivacyAudit(
        sanitized_text=text_masked,
        total_redactions=total,
        emails_redacted=emails_found,
        keys_redacted=keys_found,
        phones_redacted=phones_found,
        ips_redacted=ips_found,
    )
