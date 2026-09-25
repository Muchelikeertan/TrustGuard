"""Smoke test for TrustGuard Multimodal Consistency Engine.

Verifies:
1. The exact "Rahul" deepfake scenario (image PASS, audio FAIL, chat HIGH_PRESSURE) -> HIGH RISK.
2. The incomplete evidence scenario (missing audio) -> UNCERTAIN (not false LOW or HIGH).
3. The isolated single-failure scenario -> UNCERTAIN (never single model auto-declaring HIGH).
4. The all-pass verified scenario -> LOW.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from trustguard.consistency_engine import evaluate_consistency


def test_rahul_scenario():
    """Exact 'Rahul' scenario from design doc:

    - Claimed identity: Rahul
    - Image: PASS (DeepFace verified face match)
    - Audio: FAIL (Resemblyzer speaker similarity mismatch)
    - Chat Text: HIGH_PRESSURE ("I'm Rahul, don't call me, send Rs 20,000 now")
    - Output must be HIGH RISK with non-empty explanation and next action.
    """
    print("\n" + "=" * 80)
    print("TEST 1: 'Rahul' Multimodal Deepfake Scenario")
    print("=" * 80)

    image_result = {
        "status": "PASS",
        "verified": True,
        "distance": 0.22,
        "threshold": 0.60,
        "label": "CONSISTENT",
    }
    audio_result = {
        "status": "FAIL",
        "similarity": 0.38,
        "label": "SUSPICIOUS_SPEAKER",
        "transcript": "I'm Rahul, don't call me, send Rs 20,000 now",
    }
    chat_text = "I'm Rahul, don't call me, send Rs 20,000 now"

    result = evaluate_consistency(
        claimed_identity="Rahul",
        image_result=image_result,
        audio_result=audio_result,
        chat_text=chat_text,
    )

    print(f"Claimed Identity:    {result.claimed_identity}")
    print(f"Risk Level:          {result.risk_level}")
    print(f"Modality Statuses:   {result.modality_statuses}")
    print(f"Agreements:          {result.contradiction_map['agreements']}")
    print(f"Conflicts:           {result.contradiction_map['conflicts']}")
    print(f"Explanation:         {result.explanation}")
    print(f"Recommended Action:  {result.recommended_action}")

    # Assertions
    assert result.risk_level == "HIGH", f"Expected HIGH risk, got {result.risk_level}"
    assert result.modality_statuses["image"] == "PASS", f"Expected image PASS, got {result.modality_statuses['image']}"
    assert result.modality_statuses["voice"] == "FAIL", f"Expected voice FAIL, got {result.modality_statuses['voice']}"
    assert result.modality_statuses["chat"] == "FAIL", f"Expected chat FAIL (HIGH_PRESSURE), got {result.modality_statuses['chat']}"
    assert len(result.explanation.strip()) > 0, "Explanation must not be empty"
    assert len(result.recommended_action.strip()) > 0, "Recommended action must not be empty"
    assert "Pause the interaction" in result.recommended_action or "Do not send money" in result.recommended_action, (
        f"Action text missing key warning tone: {result.recommended_action}"
    )
    assert len(result.contradiction_map["conflicts"]) > 0, "Contradiction map must record cross-modal conflicts"

    print(">>> PASS: 'Rahul' scenario correctly assessed as HIGH RISK with complete contradiction report.")
    return result


def test_uncertain_scenario():
    """Incomplete evidence scenario:

    - Image: PASS
    - Audio: None (Missing audio evidence -> CAN'T TELL)
    - Chat Text: Benign everyday conversation -> PASS
    - Engine must lean UNCERTAIN, avoiding false LOW or false HIGH.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Incomplete Evidence Scenario (Missing Audio)")
    print("=" * 80)

    image_result = {
        "status": "PASS",
        "verified": True,
        "distance": 0.28,
    }
    audio_result = None  # Missing audio evidence
    chat_text = "Hey, are you free for lunch tomorrow at the cafeteria around 1 PM?"

    result = evaluate_consistency(
        claimed_identity="Rahul",
        image_result=image_result,
        audio_result=audio_result,
        chat_text=chat_text,
    )

    print(f"Claimed Identity:    {result.claimed_identity}")
    print(f"Risk Level:          {result.risk_level}")
    print(f"Modality Statuses:   {result.modality_statuses}")
    print(f"Uncertain Elements:  {result.contradiction_map['uncertain']}")
    print(f"Explanation:         {result.explanation}")
    print(f"Recommended Action:  {result.recommended_action}")

    # Assertions
    assert result.risk_level == "UNCERTAIN", (
        f"Expected UNCERTAIN risk for incomplete evidence, got {result.risk_level}"
    )
    assert result.modality_statuses["voice"] == "CAN'T TELL", (
        f"Expected voice CAN'T TELL, got {result.modality_statuses['voice']}"
    )
    assert result.modality_statuses["transcript"] == "CAN'T TELL", (
        f"Expected transcript CAN'T TELL, got {result.modality_statuses['transcript']}"
    )
    assert result.modality_statuses["image"] == "PASS"
    assert result.modality_statuses["chat"] == "PASS"
    assert len(result.explanation.strip()) > 0
    assert len(result.recommended_action.strip()) > 0

    print(">>> PASS: Incomplete evidence correctly leans UNCERTAIN (preventing false LOW/HIGH).")
    return result


def test_single_failure_scenario():
    """Isolated single failure scenario:

    - Rule: 'never let any single model auto-declare HIGH'
    - Image: FAIL
    - Voice: PASS
    - Chat: PASS (no high pressure)
    - Engine must output UNCERTAIN rather than auto-declaring HIGH.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Single Isolated Model Failure (Image FAIL, Voice PASS, Chat PASS)")
    print("=" * 80)

    image_result = {"status": "FAIL", "verified": False, "distance": 0.75}
    audio_result = {"status": "PASS", "similarity": 0.85, "transcript": "Hey, just confirming our meeting time."}
    chat_text = "Hey, just confirming our meeting time for tomorrow."

    result = evaluate_consistency(
        claimed_identity="Sarah",
        image_result=image_result,
        audio_result=audio_result,
        chat_text=chat_text,
    )

    print(f"Claimed Identity:    {result.claimed_identity}")
    print(f"Risk Level:          {result.risk_level}")
    print(f"Modality Statuses:   {result.modality_statuses}")
    print(f"Conflicts:           {result.contradiction_map['conflicts']}")

    assert result.risk_level == "UNCERTAIN", (
        f"Single model failure must not auto-declare HIGH; expected UNCERTAIN, got {result.risk_level}"
    )
    print(">>> PASS: Single model failure did not auto-declare HIGH; returned UNCERTAIN.")
    return result


def test_all_pass_scenario():
    """All-pass verified scenario:

    - Image: PASS
    - Voice: PASS
    - Chat: PASS
    - Engine must output LOW.
    """
    print("\n" + "=" * 80)
    print("TEST 4: All-Pass Verified Scenario")
    print("=" * 80)

    image_result = {"status": "PASS", "verified": True, "distance": 0.18}
    audio_result = {"status": "PASS", "similarity": 0.89, "transcript": "Hi Mom, just landed safely at the airport."}
    chat_text = "Hi Mom, just landed safely at the airport. Taking an Uber home now."

    result = evaluate_consistency(
        claimed_identity="Alex",
        image_result=image_result,
        audio_result=audio_result,
        chat_text=chat_text,
    )

    print(f"Claimed Identity:    {result.claimed_identity}")
    print(f"Risk Level:          {result.risk_level}")
    print(f"Modality Statuses:   {result.modality_statuses}")
    print(f"Agreements:          {result.contradiction_map['agreements']}")

    assert result.risk_level == "LOW", f"Expected LOW risk, got {result.risk_level}"
    print(">>> PASS: All-pass verified scenario correctly returns LOW risk.")
    return result


if __name__ == "__main__":
    test_rahul_scenario()
    test_uncertain_scenario()
    test_single_failure_scenario()
    test_all_pass_scenario()
    print("\n" + "=" * 80)
    print("ALL MULTIMODAL CONSISTENCY TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)
