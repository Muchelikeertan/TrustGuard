"""Unit tests for TrustGuard Consistency Engine.

Covers requirements a through i:
a. Rahul scenario: image PASS, voice FAIL, high-pressure chat gives HIGH
b. Incomplete evidence: image PASS, no voice, normal chat gives UNCERTAIN
c. Single failure: image FAIL, everything else PASS gives UNCERTAIN
d. All pass gives LOW
e. Image FAIL + voice FAIL + normal chat gives HIGH
f. Chat FAIL + transcript FAIL only (image and voice PASS) counts as one signal, so result is UNCERTAIN, not HIGH
g. Everything empty gives UNCERTAIN with all four statuses CAN'T TELL
h. Threshold boundary values for image and voice map to the right status
i. No user-facing string contains "verified" or "confirmed"
"""

import re
import pytest
from pathlib import Path

from trustguard.consistency_engine import (
    evaluate_consistency,
    load_thresholds,
    resolve_image_status,
    resolve_voice_status,
)

THRESHOLDS_PATH = Path(__file__).resolve().parent.parent / "config" / "thresholds.yaml"


def test_a_rahul_scenario():
    """a. Rahul scenario: image PASS, voice FAIL, high-pressure chat gives HIGH."""
    image_result = {"status": "PASS", "distance": 0.25}
    audio_result = {
        "status": "FAIL",
        "similarity": 0.40,
        "transcript": "Urgent! Send Rs 50,000 right now to this UPI, don't call me or tell anyone, it is an emergency!",
    }
    chat_text = "Urgent! Send Rs 50,000 right now to this UPI, don't call me or tell anyone, it is an emergency!"

    result = evaluate_consistency(
        claimed_identity="Rahul",
        image_result=image_result,
        audio_result=audio_result,
        chat_text=chat_text,
        thresholds_path=THRESHOLDS_PATH,
    )

    assert result.risk_level == "HIGH"
    assert result.modality_statuses["image"] == "PASS"
    assert result.modality_statuses["voice"] == "FAIL"
    assert result.modality_statuses["chat"] == "FAIL"
    assert len(result.contradiction_map["conflicts"]) > 0


def test_b_incomplete_evidence_uncertain():
    """b. Incomplete evidence: image PASS, no voice, normal chat gives UNCERTAIN."""
    image_result = {"status": "PASS", "distance": 0.30}
    audio_result = None  # No voice
    chat_text = "Hi, see you tomorrow for the presentation."

    result = evaluate_consistency(
        claimed_identity="Rahul",
        image_result=image_result,
        audio_result=audio_result,
        chat_text=chat_text,
        thresholds_path=THRESHOLDS_PATH,
    )

    assert result.risk_level == "UNCERTAIN"
    assert result.modality_statuses["voice"] == "CAN'T TELL"
    assert result.modality_statuses["transcript"] == "CAN'T TELL"
    assert result.modality_statuses["image"] == "PASS"
    assert result.modality_statuses["chat"] == "PASS"


def test_c_single_failure_uncertain():
    """c. Single failure: image FAIL, everything else PASS gives UNCERTAIN."""
    image_result = {"status": "FAIL", "distance": 0.75}
    audio_result = {
        "status": "PASS",
        "similarity": 0.85,
        "transcript": "Hey, checking in on the project files.",
    }
    chat_text = "Hey, checking in on the project files."

    result = evaluate_consistency(
        claimed_identity="Rahul",
        image_result=image_result,
        audio_result=audio_result,
        chat_text=chat_text,
        thresholds_path=THRESHOLDS_PATH,
    )

    assert result.risk_level == "UNCERTAIN"
    assert result.modality_statuses["image"] == "FAIL"
    assert result.modality_statuses["voice"] == "PASS"
    assert result.modality_statuses["chat"] == "PASS"


def test_d_all_pass_low():
    """d. All pass gives LOW."""
    image_result = {"status": "PASS", "distance": 0.20}
    audio_result = {
        "status": "PASS",
        "similarity": 0.88,
        "transcript": "Good morning team, let us start our regular sync.",
    }
    chat_text = "Good morning team, let us start our regular sync."

    result = evaluate_consistency(
        claimed_identity="Rahul",
        image_result=image_result,
        audio_result=audio_result,
        chat_text=chat_text,
        thresholds_path=THRESHOLDS_PATH,
    )

    assert result.risk_level == "LOW"
    assert result.modality_statuses["image"] == "PASS"
    assert result.modality_statuses["voice"] == "PASS"
    assert result.modality_statuses["chat"] == "PASS"
    assert result.modality_statuses["transcript"] == "PASS"


def test_e_image_fail_voice_fail_high():
    """e. Image FAIL + voice FAIL + normal chat gives HIGH."""
    image_result = {"status": "FAIL", "distance": 0.72}
    audio_result = {
        "status": "FAIL",
        "similarity": 0.52,
        "transcript": "Hello, hope you are having a nice day.",
    }
    chat_text = "Hello, hope you are having a nice day."

    result = evaluate_consistency(
        claimed_identity="Rahul",
        image_result=image_result,
        audio_result=audio_result,
        chat_text=chat_text,
        thresholds_path=THRESHOLDS_PATH,
    )

    assert result.risk_level == "HIGH"
    assert result.modality_statuses["image"] == "FAIL"
    assert result.modality_statuses["voice"] == "FAIL"
    assert result.modality_statuses["chat"] == "PASS"


def test_f_chat_and_transcript_fail_only_is_uncertain():
    """f. Chat FAIL + transcript FAIL only (image and voice PASS) counts as one signal, so result is UNCERTAIN, not HIGH."""
    image_result = {"status": "PASS", "distance": 0.22}
    audio_result = {
        "status": "PASS",
        "similarity": 0.85,
        "transcript": "Urgent! Send Rs 25,000 right now to this UPI, don't call me or ask questions!",
    }
    chat_text = "Urgent! Send Rs 25,000 right now to this UPI, don't call me or ask questions!"

    result = evaluate_consistency(
        claimed_identity="Rahul",
        image_result=image_result,
        audio_result=audio_result,
        chat_text=chat_text,
        thresholds_path=THRESHOLDS_PATH,
    )

    # Even though both chat and transcript triggered FAIL, they come from the same text content
    # and count as ONE text signal, so with image PASS and voice PASS, single failure gives UNCERTAIN.
    assert result.risk_level == "UNCERTAIN"
    assert result.modality_statuses["image"] == "PASS"
    assert result.modality_statuses["voice"] == "PASS"
    assert result.modality_statuses["chat"] == "FAIL"
    assert result.modality_statuses["transcript"] == "FAIL"


def test_g_everything_empty_uncertain_cant_tell():
    """g. Everything empty gives UNCERTAIN with all four statuses CAN'T TELL."""
    result = evaluate_consistency(
        claimed_identity="",
        image_result=None,
        audio_result=None,
        chat_text="",
        thresholds_path=THRESHOLDS_PATH,
    )

    assert result.risk_level == "UNCERTAIN"
    assert result.modality_statuses["image"] == "CAN'T TELL"
    assert result.modality_statuses["voice"] == "CAN'T TELL"
    assert result.modality_statuses["chat"] == "CAN'T TELL"
    assert result.modality_statuses["transcript"] == "CAN'T TELL"


def test_h_threshold_boundaries():
    """h. Threshold boundary values for image and voice map to the right status."""
    thresholds = load_thresholds(THRESHOLDS_PATH)

    # Image: PASS at or below 0.55, FAIL at or above 0.68, WARNING in between
    # Boundary: exactly 0.55 -> PASS
    stat, _ = resolve_image_status({"distance": 0.55}, thresholds)
    assert stat == "PASS", f"Expected PASS at 0.55, got {stat}"

    # Below 0.55 -> PASS
    stat, _ = resolve_image_status({"distance": 0.40}, thresholds)
    assert stat == "PASS", f"Expected PASS at 0.40, got {stat}"

    # Boundary: exactly 0.68 -> FAIL
    stat, _ = resolve_image_status({"distance": 0.68}, thresholds)
    assert stat == "FAIL", f"Expected FAIL at 0.68, got {stat}"

    # Above 0.68 -> FAIL
    stat, _ = resolve_image_status({"distance": 0.75}, thresholds)
    assert stat == "FAIL", f"Expected FAIL at 0.75, got {stat}"

    # In between 0.55 and 0.68 -> WARNING
    stat, _ = resolve_image_status({"distance": 0.60}, thresholds)
    assert stat == "WARNING", f"Expected WARNING at 0.60, got {stat}"

    stat, _ = resolve_image_status({"distance": 0.56}, thresholds)
    assert stat == "WARNING", f"Expected WARNING at 0.56, got {stat}"

    stat, _ = resolve_image_status({"distance": 0.67}, thresholds)
    assert stat == "WARNING", f"Expected WARNING at 0.67, got {stat}"

    # Voice: PASS at or above 0.78, FAIL below 0.70, WARNING in between
    # Boundary: exactly 0.78 -> PASS
    stat, _ = resolve_voice_status({"similarity": 0.78}, thresholds)
    assert stat == "PASS", f"Expected PASS at 0.78, got {stat}"

    # Above 0.78 -> PASS
    stat, _ = resolve_voice_status({"similarity": 0.85}, thresholds)
    assert stat == "PASS", f"Expected PASS at 0.85, got {stat}"

    # Below 0.70 -> FAIL
    stat, _ = resolve_voice_status({"similarity": 0.69}, thresholds)
    assert stat == "FAIL", f"Expected FAIL at 0.69, got {stat}"

    stat, _ = resolve_voice_status({"similarity": 0.50}, thresholds)
    assert stat == "FAIL", f"Expected FAIL at 0.50, got {stat}"

    # In between [0.70, 0.78) -> WARNING
    stat, _ = resolve_voice_status({"similarity": 0.70}, thresholds)
    assert stat == "WARNING", f"Expected WARNING at 0.70, got {stat}"

    stat, _ = resolve_voice_status({"similarity": 0.75}, thresholds)
    assert stat == "WARNING", f"Expected WARNING at 0.75, got {stat}"

    stat, _ = resolve_voice_status({"similarity": 0.779}, thresholds)
    assert stat == "WARNING", f"Expected WARNING at 0.779, got {stat}"


def test_i_no_forbidden_words_in_user_facing_strings():
    """i. No user-facing string contains 'verified' or 'confirmed'."""
    forbidden = [r"\bverified\b", r"\bconfirmed\b"]

    scenarios = [
        # Rahul scenario
        {
            "claimed_identity": "Rahul",
            "image_result": {"status": "PASS", "distance": 0.25},
            "audio_result": {
                "status": "FAIL",
                "similarity": 0.40,
                "transcript": "Urgent! Send Rs 50,000 right now, don't call me.",
            },
            "chat_text": "Urgent! Send Rs 50,000 right now, don't call me.",
        },
        # All pass scenario
        {
            "claimed_identity": "Sarah",
            "image_result": {"status": "PASS", "distance": 0.20},
            "audio_result": {
                "status": "PASS",
                "similarity": 0.85,
                "transcript": "Normal friendly check in.",
            },
            "chat_text": "Normal friendly check in.",
        },
        # Both fail scenario
        {
            "claimed_identity": "Alex",
            "image_result": {"status": "FAIL", "distance": 0.75},
            "audio_result": {
                "status": "FAIL",
                "similarity": 0.45,
                "transcript": "Hey there.",
            },
            "chat_text": "Hey there.",
        },
        # Incomplete scenario
        {
            "claimed_identity": "Priya",
            "image_result": None,
            "audio_result": None,
            "chat_text": None,
        },
    ]

    for sc in scenarios:
        res = evaluate_consistency(
            claimed_identity=sc["claimed_identity"],
            image_result=sc["image_result"],
            audio_result=sc["audio_result"],
            chat_text=sc["chat_text"],
            thresholds_path=THRESHOLDS_PATH,
        )

        all_user_facing = [
            res.risk_level,
            res.explanation,
            res.recommended_action,
        ]
        all_user_facing.extend(res.contradiction_map.get("agreements", []))
        all_user_facing.extend(res.contradiction_map.get("conflicts", []))
        all_user_facing.extend(res.contradiction_map.get("uncertain", []))
        all_user_facing.extend(res.modality_statuses.values())

        for text in all_user_facing:
            for pattern in forbidden:
                match = re.search(pattern, text, re.IGNORECASE)
                assert match is None, f"Found forbidden word '{match.group(0)}' in user-facing text: '{text}'"


def test_j_no_face_image_result_resolution():
    """j. Image result with no face detected resolves to CAN'T TELL with no distance."""
    thresholds = load_thresholds(THRESHOLDS_PATH)

    image_result = {
        "status": "CAN'T TELL",
        "distance": None,
        "reason": "No clear face was found in the reference photo. Please upload a clear, front-facing photo.",
    }

    stat, meta = resolve_image_status(image_result, thresholds)
    assert stat == "CAN'T TELL"
    assert meta["distance"] is None
    assert "No clear face was found" in meta["source_reason"]

