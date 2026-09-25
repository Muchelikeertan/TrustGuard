"""End-to-End Test for TrustGuard using REAL sample_data files.

Executes the exact same pipeline function called by the UI Analyze button for:
1. Same person + Same voice (Legitimate contact scenario -> Expected LOW Risk)
2. Different person + Different voice (Impersonation / scam scenario -> Expected HIGH Risk)

Prints statuses, scores, and final risk levels.
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from trustguard.pipeline import run_multimodal_analysis


def test_real_same_person_same_voice():
    """Runs real files for same person and same voice."""
    print("\n" + "=" * 80)
    print("REAL PIPELINE TEST 1: Same Person + Same Voice (Legitimate)")
    print("=" * 80)

    def progress(step: str, pct: float):
        print(f"  [{pct*100:3.0f}%] {step}")

    result, warnings = run_multimodal_analysis(
        claimed_identity="Alex",
        ref_photo_path="sample_data/images/same_person_a.jpg",
        test_photo_path="sample_data/images/same_person_b.jpg",
        ref_audio_path="sample_data/audio/reference.wav",
        test_audio_path="sample_data/audio/test.wav",
        chat_text="Hi Mom, just landed safely at the airport. Taking an Uber home now.",
        progress_callback=progress,
    )

    print("\n--- RESULTS ---")
    print(f"Claimed Identity:    {result.claimed_identity}")
    print(f"Risk Level:          {result.risk_level}")
    print(f"Modality Statuses:   {result.modality_statuses}")
    print(f"Image Distance:      {result.details.get('image', {}).get('distance')}")
    print(f"Voice Similarity:    {result.details.get('voice', {}).get('similarity')}")
    print(f"Execution Time:      {result.details.get('elapsed_time_seconds')}s")
    print(f"Warnings:            {warnings}")
    print(f"Explanation:         {result.explanation}")
    print(f"Recommended Action:  {result.recommended_action}")

    # Assertions
    assert result.modality_statuses["image"] == "PASS", f"Expected image PASS, got {result.modality_statuses['image']}"
    assert result.modality_statuses["voice"] == "PASS", f"Expected voice PASS, got {result.modality_statuses['voice']}"
    assert result.modality_statuses["chat"] == "PASS", f"Expected chat PASS, got {result.modality_statuses['chat']}"
    assert result.risk_level == "LOW", f"Expected LOW risk, got {result.risk_level}"
    print("\n>>> TEST 1 PASSED: Real legitimate samples evaluated as LOW risk.")
    return result


def test_real_different_person_different_voice():
    """Runs real files for different person, different voice, and high-pressure text."""
    print("\n" + "=" * 80)
    print("REAL PIPELINE TEST 2: Different Person + Different Voice (Impersonation)")
    print("=" * 80)

    def progress(step: str, pct: float):
        print(f"  [{pct*100:3.0f}%] {step}")

    result, warnings = run_multimodal_analysis(
        claimed_identity="Rahul",
        ref_photo_path="sample_data/images/same_person_a.jpg",
        test_photo_path="sample_data/images/different_person.jpg",
        ref_audio_path="sample_data/audio/reference.wav",
        test_audio_path="sample_data/audio/different_speaker.wav",
        chat_text="Urgent! Send Rs 25,000 right now to this UPI, don't call me or ask questions!",
        progress_callback=progress,
    )

    print("\n--- RESULTS ---")
    print(f"Claimed Identity:    {result.claimed_identity}")
    print(f"Risk Level:          {result.risk_level}")
    print(f"Modality Statuses:   {result.modality_statuses}")
    print(f"Image Distance:      {result.details.get('image', {}).get('distance')}")
    print(f"Voice Similarity:    {result.details.get('voice', {}).get('similarity')}")
    print(f"Execution Time:      {result.details.get('elapsed_time_seconds')}s")
    print(f"Warnings:            {warnings}")
    print(f"Agreements:          {result.contradiction_map['agreements']}")
    print(f"Conflicts:           {result.contradiction_map['conflicts']}")
    print(f"Explanation:         {result.explanation}")
    print(f"Recommended Action:  {result.recommended_action}")

    # Assertions
    assert result.modality_statuses["image"] == "FAIL", f"Expected image FAIL, got {result.modality_statuses['image']}"
    assert result.modality_statuses["voice"] == "FAIL", f"Expected voice FAIL, got {result.modality_statuses['voice']}"
    assert result.modality_statuses["chat"] == "FAIL", f"Expected chat FAIL, got {result.modality_statuses['chat']}"
    assert result.risk_level == "HIGH", f"Expected HIGH risk, got {result.risk_level}"
    print("\n>>> TEST 2 PASSED: Real divergent samples evaluated as HIGH risk.")
    return result


if __name__ == "__main__":
    test_real_same_person_same_voice()
    test_real_different_person_different_voice()
    print("\n" + "=" * 80)
    print("ALL REAL PIPELINE END-TO-END TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
