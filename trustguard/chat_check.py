"""TrustGuard Chat Check Module.

Combines zero-shot classification (facebook/bart-large-mnli) with a deterministic
rule layer (English & Hinglish regexes) to detect social engineering, high-pressure
coercion, and financial scam patterns in text messages.
"""

import functools
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Suppress symlink warnings on Windows for Hugging Face Hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Default thresholds
DEFAULT_SUSPICIOUS_MIN = 0.55
DEFAULT_HIGH_PRESSURE_MIN = 0.80

# Candidate labels for zero-shot classification
SCAM_LABELS = [
    "urgent money request",
    "asks for OTP, password or bank details",
    "tells the person not to verify or to keep it secret",
    "threat or pressure",
]

BENIGN_LABELS = [
    "normal conversation",
    "casual everyday update",
    "polite work or study message",
]

CANDIDATE_LABELS = SCAM_LABELS + BENIGN_LABELS


@dataclass
class Signal:
    """Represents the outcome of a chat safety analysis."""

    score: Optional[float]
    status: str  # "HIGH_PRESSURE", "SUSPICIOUS", "NORMAL", "MISSING", "ERROR"
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


@functools.lru_cache(maxsize=1)
def get_classifier():
    """Lazy-loads and caches the zero-shot classification pipeline."""
    from transformers import pipeline

    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        multi_label=False,
    )


# ---------------------------------------------------------------------------
# Rule Definitions (English + common Hinglish)
# ---------------------------------------------------------------------------

# Strong rules (weight = 0.8)
STRONG_RULES = [
    (
        "otp_credentials_request",
        re.compile(
            r"\b(?:otp|one[- ]?time[- ]?(?:password|passcode)|pin(?:\s*code)?|cvv|cvc|net[- ]?banking password|banking pin|atm pin|login credentials?|secret code)\b|"
            r"(?<!wifi\s)(?<!wi-fi\s)(?<!hotspot\s)\b(?:passwords?|passcodes?)\s+(?:bhejo|batao|share karo|de do|bhej do|send karo|share kardo)\b|"
            r"\b(?:send|share|give|tell)\s+(?:me\s+)?(?:your\s+)?(?:otp|pin|cvv|cvc|banking password|account password)\b",
            flags=re.IGNORECASE,
        ),
        0.8,
    ),
    (
        "isolation_and_secrecy",
        re.compile(
            r"\b(?:don'?t|do not)\s+(?:call|tell|inform|share|disclose|contact|verify|ask)\b|"
            r"\bkeep\s+(?:this|it)?\s*(?:a\s*)?secret\b|"
            r"\b(?:can'?t|cannot|unable to)\s+(?:take|receive|answer|speak on)\s+(?:calls?|phone)\b|"
            r"\b(?:kisi ko|kisi se)\s+(?:mat|nahi)\s+(?:batana|batao|bolna|poochna|share karna)\b|"
            r"\b(?:call|phone)\s+mat\s+(?:karna|karo)\b|"
            r"\braaz\s+rakhna\b",
            flags=re.IGNORECASE,
        ),
        0.8,
    ),
    (
        "threat_arrest_blackmail",
        re.compile(
            r"\b(?:arrest|police|cbi|ed|narcotics|customs|court|fir|legal action|warrant|jail|lawsuit|blackmail|penalt(?:y|ies))\b|"
            r"\b(?:police\s+aayegi|jail\s+jaoge|giraftaar|hawaalat|case\s+darj|complaint\s+darj)\b",
            flags=re.IGNORECASE,
        ),
        0.8,
    ),
]

# Medium rules (weight = 0.4)
MEDIUM_RULES = [
    (
        "urgency_cues",
        re.compile(
            r"\b(?:urgent(?:ly)?|immediate(?:ly)?|right now|asap|emergency|without delay|hurry)\b|"
            r"\b(?:jaldi|turant|fatafat|abhi ke abhi|ekdum jaldi)\b",
            flags=re.IGNORECASE,
        ),
        0.4,
    ),
    (
        "payment_cues",
        re.compile(
            r"\b(?:upi|bank transfer|wire transfer|gift cards?|crypto(?:currency)?|bitcoin|gpay|google pay|phonepe|paytm|qr code)\b|"
            r"\b(?:send|transfer|pay|deposit|settle|clear)\s+(?:the\s+)?(?:fee|bill|dues|fine|penalty|(?:rs\.?|inr|usd|\$|₹)?\s*[\d,]+)\b|"
            r"\b(?:paise|rupaye)\s+(?:bhejo|daalo|transfer karo|de do)\b",
            flags=re.IGNORECASE,
        ),
        0.4,
    ),
    (
        "shortened_url",
        re.compile(
            r"(?:https?://)?(?:www\.)?(?:bit\.ly|tinyurl\.com|t\.co|is\.gd|cutt\.ly|shorturl\.at|rb\.gy|wa\.me)/[a-zA-Z0-9_-]+",
            flags=re.IGNORECASE,
        ),
        0.4,
    ),
]


def evaluate_rules(text: str) -> tuple[float, List[Dict[str, Any]]]:
    """Evaluates text against strong and medium heuristic rules.

    Returns:
        tuple[float, List[Dict[str, Any]]]: rule_score (capped at 1.0) and details of fired rules.
    """
    fired_rules: List[Dict[str, Any]] = []
    total_weight = 0.0

    all_rules = STRONG_RULES + MEDIUM_RULES

    for rule_name, pattern, weight in all_rules:
        match = pattern.search(text)
        if match:
            snippet = match.group(0)
            fired_rules.append({
                "rule": rule_name,
                "weight": weight,
                "matched": snippet,
            })
            total_weight += weight

    rule_score = min(1.0, total_weight)
    return rule_score, fired_rules


def analyze_chat(
    text: str,
    suspicious_min: float = DEFAULT_SUSPICIOUS_MIN,
    high_pressure_min: float = DEFAULT_HIGH_PRESSURE_MIN,
) -> Signal:
    """Analyzes a chat message for social engineering, urgency, and coercion signals.

    Args:
        text (str): The raw chat message to analyze.
        suspicious_min (float): Score threshold above which text is flagged SUSPICIOUS.
        high_pressure_min (float): Score threshold above which text is flagged HIGH_PRESSURE.

    Returns:
        Signal: Struct containing score, status, reason, and breakdown details.
    """
    # 1. Input Validation
    if not text or not text.strip():
        return Signal(
            score=None,
            status="MISSING",
            reason="Empty or missing chat text.",
            details={},
        )

    clean_text = text.strip()

    try:
        # 2. Rule Layer
        rule_score, fired_rules = evaluate_rules(clean_text)

        # 3. Model Layer (Zero-shot BART-large-MNLI)
        classifier = get_classifier()
        model_output = classifier(
            clean_text,
            candidate_labels=CANDIDATE_LABELS,
            multi_label=False,
        )

        label_to_score = dict(zip(model_output["labels"], model_output["scores"]))

        # Model score = HIGHEST score among the SCAM labels
        scam_scores = {lbl: float(label_to_score.get(lbl, 0.0)) for lbl in SCAM_LABELS}
        top_scam_label = max(scam_scores, key=scam_scores.get)
        model_score = scam_scores[top_scam_label]

        # 4. Combine Signals
        final_score = max(model_score, rule_score)

        # 5. Determine Status
        if final_score >= high_pressure_min:
            status = "HIGH_PRESSURE"
        elif final_score >= suspicious_min:
            status = "SUSPICIOUS"
        else:
            status = "NORMAL"

        # 6. Build Human-Readable Reason
        reasons_list = []
        if status in ("HIGH_PRESSURE", "SUSPICIOUS"):
            if model_score >= suspicious_min:
                reasons_list.append(f"Model detected '{top_scam_label}' (conf={model_score:.2f})")
            if fired_rules:
                rule_names = [r["rule"] for r in fired_rules]
                reasons_list.append(f"Fired heuristics: {', '.join(rule_names)} (rule_score={rule_score:.2f})")
        else:
            reasons_list.append(f"Normal message characteristics (top overall: '{model_output['labels'][0]}', scam_score={model_score:.2f})")

        reason_str = " | ".join(reasons_list)

        details = {
            "final_score": round(final_score, 4),
            "model_score": round(model_score, 4),
            "rule_score": round(rule_score, 4),
            "top_scam_label": top_scam_label,
            "scam_label_scores": {k: round(v, 4) for k, v in scam_scores.items()},
            "top_overall_label": model_output["labels"][0],
            "top_overall_score": round(float(model_output["scores"][0]), 4),
            "fired_rules": fired_rules,
            "thresholds": {
                "suspicious_min": suspicious_min,
                "high_pressure_min": high_pressure_min,
            },
        }

        return Signal(
            score=round(final_score, 4),
            status=status,
            reason=reason_str,
            details=details,
        )

    except Exception as exc:
        return Signal(
            score=None,
            status="ERROR",
            reason=f"Chat analysis exception: {str(exc)}",
            details={"error": str(exc)},
        )
