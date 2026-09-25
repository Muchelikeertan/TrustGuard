"""Multimodal Consistency Engine for TrustGuard.

Fuses evidence signals across face consistency (DeepFace), speaker consistency
(Resemblyzer), speech-to-text analysis (Whisper), and chat analysis (BART zero-shot)
into a unified contradiction map, risk level (LOW / UNCERTAIN / HIGH), and human-readable
explanation with actionable guidance.
"""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from trustguard.chat_check import Signal, analyze_chat

# Path to default configuration
DEFAULT_THRESHOLDS_PATH = Path(__file__).resolve().parent.parent / "config" / "thresholds.yaml"

# Fallback thresholds if YAML is unreadable
FALLBACK_THRESHOLDS = {
    "image": {
        "pass_max_distance": 0.55,
        "fail_min_distance": 0.68,
    },
    "voice": {
        "pass_min_similarity": 0.78,
        "warning_min_similarity": 0.70,
        "fail_max_similarity": 0.70,
    },
    "chat": {
        "suspicious_min": 0.65,
        "high_pressure_min": 0.80,
    },
    "transcript": {
        "suspicious_min": 0.65,
        "high_pressure_min": 0.80,
    },
}


def load_thresholds(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads verification thresholds from YAML or returns fallback defaults."""
    path = Path(config_path) if config_path else DEFAULT_THRESHOLDS_PATH
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    return loaded
        except Exception:
            pass
    return FALLBACK_THRESHOLDS


@dataclass
class ContradictionMap:
    """Detailed structural representation of cross-modal alignment."""

    agreements: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    uncertain: List[str] = field(default_factory=list)
    modality_statuses: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultimodalConsistencyResult:
    """Result object produced by the Consistency Engine."""

    claimed_identity: str
    risk_level: str  # "LOW", "UNCERTAIN", "HIGH"
    modality_statuses: Dict[str, str]  # image, voice, chat, transcript
    contradiction_map: Dict[str, Any]
    explanation: str
    recommended_action: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-Modality Status Resolvers
# ---------------------------------------------------------------------------

def resolve_image_status(image_result: Any, thresholds: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Maps image verification result to PASS / WARNING / FAIL / CAN'T TELL."""
    meta: Dict[str, Any] = {}
    if image_result is None:
        return "CAN'T TELL", {"reason": "No image evidence provided."}

    # Normalize dict or object
    if isinstance(image_result, dict):
        status_raw = image_result.get("status")
        label = image_result.get("label")
        verified = image_result.get("verified")
        distance = image_result.get("distance")
        score = image_result.get("score")
        reason = image_result.get("reason", "")
    else:
        status_raw = getattr(image_result, "status", None)
        label = getattr(image_result, "label", None)
        verified = getattr(image_result, "verified", None)
        distance = getattr(image_result, "distance", None)
        score = getattr(image_result, "score", None)
        reason = getattr(image_result, "reason", "")

    pass_max = thresholds.get("image", {}).get("pass_max_distance", 0.55)
    fail_min = thresholds.get("image", {}).get("fail_min_distance", 0.68)

    meta = {
        "label": label,
        "distance": distance,
        "pass_max_distance": pass_max,
        "fail_min_distance": fail_min,
        "score": score,
        "source_reason": reason,
    }

    # 1. Missing / Indeterminate
    if label in ("WAITING FOR DATA", "MISSING", "CAN'T TELL") or status_raw == "CAN'T TELL":
        return "CAN'T TELL", meta

    # 2. Numerical distance check (Primary standard)
    if distance is not None and isinstance(distance, (int, float)):
        if distance <= pass_max:
            return "PASS", meta
        elif distance >= fail_min:
            return "FAIL", meta
        else:
            return "WARNING", meta

    # 3. Explicit direct status
    if status_raw in ("PASS", "WARNING", "FAIL"):
        return status_raw, meta

    # 4. Consistency boolean flag or label fallback
    if verified is not None:
        return ("PASS" if verified else "FAIL"), meta

    if label == "CONSISTENT":
        return "PASS", meta
    elif label == "INCONSISTENT":
        return "FAIL", meta

    return "CAN'T TELL", meta


def resolve_voice_status(audio_result: Any, thresholds: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Maps Resemblyzer speaker verification result to PASS / WARNING / FAIL / CAN'T TELL."""
    meta: Dict[str, Any] = {}
    if audio_result is None:
        return "CAN'T TELL", {"reason": "No audio evidence provided."}

    # Handle tuple format: (score, transcript)
    if isinstance(audio_result, (list, tuple)) and len(audio_result) >= 1:
        similarity = audio_result[0]
        status_raw = None
        label = None
        reason = ""
    elif isinstance(audio_result, dict):
        status_raw = audio_result.get("status")
        label = audio_result.get("label")
        similarity = audio_result.get("similarity", audio_result.get("score"))
        reason = audio_result.get("reason", "")
    else:
        status_raw = getattr(audio_result, "status", None)
        label = getattr(audio_result, "label", None)
        similarity = getattr(audio_result, "similarity", getattr(audio_result, "score", None))
        reason = getattr(audio_result, "reason", "")

    v_thresh = thresholds.get("voice", {})
    pass_sim = v_thresh.get("pass_min_similarity", 0.78)
    fail_sim = v_thresh.get("fail_max_similarity", 0.70)

    meta = {
        "label": label,
        "similarity": similarity,
        "pass_min_similarity": pass_sim,
        "fail_max_similarity": fail_sim,
        "source_reason": reason,
    }

    # 1. Missing / Indeterminate
    if label in ("WAITING FOR DATA", "MISSING", "CAN'T TELL") or status_raw == "CAN'T TELL":
        return "CAN'T TELL", meta

    # 2. Numerical similarity score
    if similarity is not None and isinstance(similarity, (int, float)):
        if similarity >= pass_sim:
            return "PASS", meta
        elif similarity < fail_sim:
            return "FAIL", meta
        else:
            return "WARNING", meta

    # 3. Explicit direct status
    if status_raw in ("PASS", "WARNING", "FAIL"):
        return status_raw, meta

    # 4. Label fallback
    if label in ("VERIFIED_SPEAKER", "CONSISTENT_SPEAKER", "CONSISTENT", "MATCH"):
        return "PASS", meta
    elif label in ("SUSPICIOUS_SPEAKER", "INCONSISTENT_SPEAKER", "INCONSISTENT", "MISMATCH"):
        return "FAIL", meta

    return "CAN'T TELL", meta


def extract_transcript(audio_result: Any) -> Optional[str]:
    """Extracts transcript text from audio result if present."""
    if audio_result is None:
        return None
    if isinstance(audio_result, (list, tuple)) and len(audio_result) >= 2:
        return str(audio_result[1]) if audio_result[1] else None
    if isinstance(audio_result, dict):
        return audio_result.get("transcript") or audio_result.get("text") or audio_result.get("transcription")
    return getattr(audio_result, "transcript", getattr(audio_result, "text", None))


def map_chat_signal_status(signal: Signal) -> str:
    """Maps chat_check Signal status (NORMAL/SUSPICIOUS/HIGH_PRESSURE) to PASS/WARNING/FAIL/CAN'T TELL."""
    if signal.status == "HIGH_PRESSURE":
        return "FAIL"
    elif signal.status == "SUSPICIOUS":
        return "WARNING"
    elif signal.status == "NORMAL":
        return "PASS"
    return "CAN'T TELL"


# ---------------------------------------------------------------------------
# Multimodal Consistency Engine
# ---------------------------------------------------------------------------

def evaluate_consistency(
    claimed_identity: str,
    image_result: Optional[Any] = None,
    audio_result: Optional[Any] = None,
    chat_text: Optional[str] = None,
    thresholds_path: Optional[Union[str, Path]] = None,
) -> MultimodalConsistencyResult:
    """Evaluates cross-modal consistency across face, voice, chat, and transcript.

    Args:
        claimed_identity (str): The name/identity claimed in the communication.
        image_result (Optional[Any]): DeepFace verification output or status dict.
        audio_result (Optional[Any]): Resemblyzer score and/or Whisper transcript.
        chat_text (Optional[str]): The incoming text/chat message.
        thresholds_path (Optional[Union[str, Path]]): Custom path to thresholds.yaml.

    Returns:
        MultimodalConsistencyResult: Structured result with risk level and explanation.
    """
    thresholds = load_thresholds(thresholds_path)
    claimed_name = claimed_identity.strip() if claimed_identity else "Unknown Contact"

    # 1. Resolve Image Status
    img_status, img_meta = resolve_image_status(image_result, thresholds)

    # 2. Resolve Voice Status
    voice_status, voice_meta = resolve_voice_status(audio_result, thresholds)

    # 3. Analyze Chat Text
    chat_thresh = thresholds.get("chat", {})
    chat_suspicious = chat_thresh.get("suspicious_min", 0.65)
    chat_high = chat_thresh.get("high_pressure_min", 0.80)

    chat_signal = (
        analyze_chat(chat_text, suspicious_min=chat_suspicious, high_pressure_min=chat_high)
        if chat_text
        else Signal(score=None, status="MISSING", reason="No chat text provided.")
    )
    chat_status = map_chat_signal_status(chat_signal)

    # 4. Analyze Whisper Transcript (Reusing the same chat_check model and thresholds)
    trans_thresh = thresholds.get("transcript", {})
    trans_suspicious = trans_thresh.get("suspicious_min", 0.65)
    trans_high = trans_thresh.get("high_pressure_min", 0.80)

    transcript_text = extract_transcript(audio_result)
    transcript_signal = (
        analyze_chat(transcript_text, suspicious_min=trans_suspicious, high_pressure_min=trans_high)
        if transcript_text
        else Signal(score=None, status="MISSING", reason="No audio transcript available.")
    )
    transcript_status = map_chat_signal_status(transcript_signal)

    modality_statuses = {
        "image": img_status,
        "voice": voice_status,
        "chat": chat_status,
        "transcript": transcript_status,
    }

    # 5. Build Contradiction Map
    agreements: List[str] = []
    conflicts: List[str] = []
    uncertain: List[str] = []

    # Record uncertain modalities
    for mod_name, stat in modality_statuses.items():
        if stat == "CAN'T TELL":
            uncertain.append(f"{mod_name.capitalize()} evidence is missing or inconclusive.")

    # Cross-modal identity conflict: Image vs Voice
    if img_status == "PASS" and voice_status == "FAIL":
        conflicts.append(
            f"Visual facial analysis is consistent with the reference for '{claimed_name}' (PASS), "
            f"but voice biometric similarity failed (FAIL)."
        )
    elif img_status == "FAIL" and voice_status == "PASS":
        conflicts.append(
            f"Voice biometric is consistent with the reference for '{claimed_name}' (PASS), "
            f"but facial image analysis failed (FAIL)."
        )
    elif img_status == "PASS" and voice_status == "PASS":
        agreements.append(
            f"Both facial and voice biometric signals are consistent with the reference for '{claimed_name}'."
        )
    elif img_status == "FAIL" and voice_status == "FAIL":
        agreements.append(
            f"Both facial and voice biometric signals diverge from the reference for '{claimed_name}'."
        )

    # Text Coercion vs Identity Consistency Conflict
    if img_status == "PASS" and chat_status == "FAIL":
        conflicts.append(
            f"Photo is consistent with '{claimed_name}', but chat text contains high-pressure coercion or payment demands (FAIL)."
        )
    if voice_status == "PASS" and chat_status == "FAIL":
        conflicts.append(
            f"Voice is consistent with '{claimed_name}', but text demands immediate action or secrecy under high pressure (FAIL)."
        )

    # Audio biometrics vs Transcript pressure
    if voice_status == "FAIL" and chat_status == "FAIL":
        agreements.append(
            "Voice acoustic mismatch and high-pressure chat communication both indicate potential impersonation."
        )
    if voice_status == "FAIL" and transcript_status == "FAIL":
        agreements.append(
            "Voice acoustic mismatch and spoken transcript cues both indicate potential impersonation."
        )

    # Agreement on normal communication
    if chat_status == "PASS" and transcript_status == "PASS":
        agreements.append("Both written chat and audio transcript reflect normal, non-coercive communication.")

    contradiction_map = ContradictionMap(
        agreements=agreements,
        conflicts=conflicts,
        uncertain=uncertain,
        modality_statuses=modality_statuses,
    )

    # 6. Overall Risk Level Computation
    # Rules:
    # - Chat and Whisper transcript count as ONE unified "text" signal when counting failures.
    # - No single failing signal can produce HIGH.
    # - Two or more failing signals, or a biometric FAIL (image or voice) together with a high-pressure text FAIL, produce HIGH.
    # - Missing evidence with no FAIL produces UNCERTAIN, never LOW.
    img_is_fail = (img_status == "FAIL")
    voice_is_fail = (voice_status == "FAIL")
    text_is_fail = (chat_status == "FAIL") or (transcript_status == "FAIL")

    failing_signals = (1 if img_is_fail else 0) + (1 if voice_is_fail else 0) + (1 if text_is_fail else 0)

    warn_count = sum(1 for s in modality_statuses.values() if s == "WARNING")
    has_missing_evidence = (
        img_status == "CAN'T TELL"
        or voice_status == "CAN'T TELL"
        or (chat_status == "CAN'T TELL" and transcript_status == "CAN'T TELL")
    )

    if failing_signals >= 2:
        risk_level = "HIGH"
    elif failing_signals == 1:
        risk_level = "UNCERTAIN"
    elif has_missing_evidence:
        risk_level = "UNCERTAIN"
    elif warn_count > 0:
        risk_level = "UNCERTAIN"
    elif img_status == "PASS" and voice_status == "PASS" and (chat_status == "PASS" or transcript_status == "PASS"):
        risk_level = "LOW"
    else:
        risk_level = "UNCERTAIN"

    # 7. Generate Explanation (Evidence -> What conflicts -> Why it matters)
    explanation_parts = []

    # Part A: Evidence observed
    observed_ev = []
    for mod, stat in modality_statuses.items():
        observed_ev.append(f"{mod.capitalize()}: {stat}")
    explanation_parts.append(f"Observed Evidence: {', '.join(observed_ev)}.")

    # Part B: What conflicts
    if conflicts:
        explanation_parts.append(f"Cross-Modal Conflict: {' '.join(conflicts)}")
    elif agreements and risk_level == "LOW":
        explanation_parts.append(
            "Cross-Modal Alignment: All evaluated modalities are consistent with the reference for the claimed identity."
        )
    elif uncertain:
        explanation_parts.append(f"Evidence Gaps: {' '.join(uncertain)}")

    # Part C: Why it matters (Strictly adheres to decision support, no 'hallmark', 'verified', 'confirmed')
    if risk_level == "HIGH":
        explanation_parts.append(
            f"Why it matters: Combining reference imagery with divergent voice biometrics and high-pressure coercion "
            f"is a pattern consistent with impersonation or pressure tactics targeting '{claimed_name}'. "
            f"This assessment provides decision support, not proof. A human makes the final call."
        )
    elif risk_level == "UNCERTAIN":
        explanation_parts.append(
            f"Why it matters: Evidence signals for '{claimed_name}' are incomplete or partially conflicting. "
            f"Executing financial transfers or sharing sensitive details without corroborating evidence carries risk. "
            f"This assessment provides decision support, not proof. A human makes the final call."
        )
    else:
        explanation_parts.append(
            f"Why it matters: No contradictory or coercive signals detected for '{claimed_name}'. "
            f"Evaluated evidence is consistent with the reference. "
            f"This assessment provides decision support, not proof. A human makes the final call."
        )

    full_explanation = " ".join(explanation_parts)

    # 8. Generate Recommended Next Action
    if risk_level == "HIGH":
        recommended_action = (
            "Pause the interaction immediately. Do not send money, gift cards, or credentials. "
            f"Do not respond using the current communication channel. Independently call {claimed_name} "
            "using a known, trusted phone number or check with mutual contacts in person."
        )
    elif risk_level == "UNCERTAIN":
        recommended_action = (
            "Do not execute any financial or sensitive requests yet. Request additional corroborating evidence: "
            f"ask {claimed_name} for a short live video call or check their situation through a trusted secondary contact."
        )
    else:
        recommended_action = (
            "Standard security vigilance applies. The communication appears consistent with the reference identity, "
            "but always follow standard out-of-band checks before initiating funds transfers."
        )

    # 9. Return Unified Result Object
    return MultimodalConsistencyResult(
        claimed_identity=claimed_name,
        risk_level=risk_level,
        modality_statuses=modality_statuses,
        contradiction_map=contradiction_map.to_dict(),
        explanation=full_explanation,
        recommended_action=recommended_action,
        details={
            "image": img_meta,
            "voice": voice_meta,
            "chat_signal": chat_signal.details if hasattr(chat_signal, "details") else {},
            "transcript_signal": transcript_signal.details if hasattr(transcript_signal, "details") else {},
            "failing_signals": failing_signals,
            "warn_count": warn_count,
            "has_missing_evidence": has_missing_evidence,
        },
    )
