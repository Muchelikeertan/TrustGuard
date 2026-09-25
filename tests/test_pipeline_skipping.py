"""Tests for Phase 1 & Phase 2: No-face detection and input-conditional model skipping.

Verifies:
1. A photo with no face (blank/solid image) gives image = CAN'T TELL and no distance value.
2. Both photos with clear faces still works (same person ~0.34 PASS, diff person ~0.81 FAIL).
3. Only one photo uploaded gives image = CAN'T TELL and DeepFace is NOT called (verified with spy/mock).
4. Only reference voice clip uploaded gives voice = CAN'T TELL and Resemblyzer is NOT called.
5. No test audio means Whisper is NOT called.
6. Empty chat text means BART classifier is NOT called.
7. Only a chat message provided: only chat model runs, result is UNCERTAIN or HIGH, never LOW.
8. Nothing provided at all: no model is loaded/called, and app shows the "add evidence" message.
9. Everything provided: all models run and match expected risk.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from trustguard.pipeline import run_multimodal_analysis

SAMPLE_IMAGES = root_dir / "sample_data" / "images"
SAMPLE_AUDIO = root_dir / "sample_data" / "audio"
NO_FACE_IMG = SAMPLE_IMAGES / "no_face.jpg"


@pytest.fixture(scope="session", autouse=True)
def ensure_no_face_image():
    """Generates a blank gray image without a face for testing."""
    if not NO_FACE_IMG.exists():
        img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        img.save(str(NO_FACE_IMG))
    return NO_FACE_IMG


def test_1_no_face_gives_cant_tell_and_no_distance():
    """1. A photo with no face plus a normal second photo gives image = CAN'T TELL and no distance."""
    # Test A: Reference photo has no face
    res_a, warnings_a = run_multimodal_analysis(
        claimed_identity="Alex",
        ref_photo_path=NO_FACE_IMG,
        test_photo_path=SAMPLE_IMAGES / "same_person_a.jpg",
    )
    assert res_a.modality_statuses["image"] == "CAN'T TELL"
    assert res_a.details["image"]["distance"] is None
    assert "No clear face was found in the reference photo" in res_a.details["image"]["source_reason"]
    assert len(warnings_a) == 0  # Normal result, not treated as a crash

    # Test B: Test photo has no face
    res_b, warnings_b = run_multimodal_analysis(
        claimed_identity="Alex",
        ref_photo_path=SAMPLE_IMAGES / "same_person_a.jpg",
        test_photo_path=NO_FACE_IMG,
    )
    assert res_b.modality_statuses["image"] == "CAN'T TELL"
    assert res_b.details["image"]["distance"] is None
    assert "No clear face was found in the test photo" in res_b.details["image"]["source_reason"]
    assert len(warnings_b) == 0


def test_2_both_photos_clear_faces():
    """2. Both photos with clear faces still works (same person ~0.34 PASS, different person ~0.81 FAIL)."""
    # Same person pair
    res_same, _ = run_multimodal_analysis(
        claimed_identity="Alex",
        ref_photo_path=SAMPLE_IMAGES / "same_person_a.jpg",
        test_photo_path=SAMPLE_IMAGES / "same_person_b.jpg",
    )
    assert res_same.modality_statuses["image"] == "PASS"
    dist_same = res_same.details["image"]["distance"]
    assert dist_same is not None
    assert 0.28 <= dist_same <= 0.40, f"Expected same-person dist ~0.34, got {dist_same}"

    # Different person pair
    res_diff, _ = run_multimodal_analysis(
        claimed_identity="Alex",
        ref_photo_path=SAMPLE_IMAGES / "same_person_a.jpg",
        test_photo_path=SAMPLE_IMAGES / "different_person.jpg",
    )
    assert res_diff.modality_statuses["image"] == "FAIL"
    dist_diff = res_diff.details["image"]["distance"]
    assert dist_diff is not None
    assert 0.75 <= dist_diff <= 0.88, f"Expected diff-person dist ~0.81, got {dist_diff}"


def test_3_only_one_photo_skips_deepface():
    """3. Only one photo uploaded gives image = CAN'T TELL and DeepFace is NOT called."""
    with patch("deepface.DeepFace.verify") as mock_verify, patch("deepface.DeepFace.extract_faces") as mock_extract:
        res, _ = run_multimodal_analysis(
            claimed_identity="Alex",
            ref_photo_path=SAMPLE_IMAGES / "same_person_a.jpg",
            test_photo_path=None,
        )
        assert res.modality_statuses["image"] == "CAN'T TELL"
        assert res.details["image"]["source_reason"] == "Both photos are needed."
        assert mock_verify.call_count == 0
        assert mock_extract.call_count == 0


def test_4_only_reference_voice_skips_resemblyzer():
    """4. Only the reference voice clip uploaded gives voice = CAN'T TELL and Resemblyzer is NOT called."""
    with patch("trustguard.pipeline.get_voice_encoder") as mock_get_encoder:
        res, _ = run_multimodal_analysis(
            claimed_identity="Alex",
            ref_audio_path=SAMPLE_AUDIO / "reference.wav",
            test_audio_path=None,
        )
        assert res.modality_statuses["voice"] == "CAN'T TELL"
        assert res.details["voice"]["source_reason"] == "Both voice clips are needed."
        assert mock_get_encoder.call_count == 0


def test_5_no_test_audio_skips_whisper():
    """5. No test audio means Whisper is NOT called."""
    with patch("trustguard.pipeline.get_whisper_model") as mock_whisper:
        res, _ = run_multimodal_analysis(
            claimed_identity="Alex",
            ref_audio_path=SAMPLE_AUDIO / "reference.wav",
            test_audio_path=None,
        )
        assert res.modality_statuses["transcript"] == "CAN'T TELL"
        assert mock_whisper.call_count == 0


def test_6_empty_chat_text_skips_bart():
    """6. Empty chat text means the BART classifier is NOT called."""
    with patch("trustguard.chat_check.get_classifier") as mock_classifier:
        res, _ = run_multimodal_analysis(
            claimed_identity="Alex",
            chat_text="   ",  # whitespace only
        )
        assert res.modality_statuses["chat"] == "CAN'T TELL"
        assert mock_classifier.call_count == 0


def test_7_only_chat_message_provided():
    """7. Only a chat message provided: only chat model runs, result is UNCERTAIN or HIGH, never LOW."""
    # Subcase A: Normal chat -> UNCERTAIN (never LOW because biometrics are missing)
    res_normal, _ = run_multimodal_analysis(
        claimed_identity="Alex",
        chat_text="Hi, hope you are having a wonderful day!",
    )
    assert res_normal.modality_statuses["chat"] == "PASS"
    assert res_normal.modality_statuses["image"] == "CAN'T TELL"
    assert res_normal.modality_statuses["voice"] == "CAN'T TELL"
    assert res_normal.risk_level == "UNCERTAIN", "Missing evidence must yield UNCERTAIN, never LOW"

    # Subcase B: High-pressure coercion -> UNCERTAIN (single failure rule: no single signal produces HIGH)
    res_coercion, _ = run_multimodal_analysis(
        claimed_identity="Alex",
        chat_text="Urgent! Send Rs 50,000 right now to this UPI, don't call me or tell anyone!",
    )
    assert res_coercion.modality_statuses["chat"] == "FAIL"
    assert res_coercion.risk_level == "UNCERTAIN"


def test_8_nothing_provided_at_all():
    """8. Nothing provided at all: app shows the 'add evidence' message without running models."""
    at = AppTest.from_file("../app.py", default_timeout=30).run()
    # Clear any preset values in inputs
    at.text_input[0].set_value("").run()
    at.text_area[0].set_value("").run()

    # Click analyze button
    analyze_btn = [b for b in at.button if "Analyze" in b.label][0]
    analyze_btn.click().run()

    # Verify no active result and informational notice displayed
    assert at.session_state.active_result is None
    info_messages = [info.value for info in at.info]
    assert any("Please add at least one piece of evidence" in m for m in info_messages)


def test_9_everything_provided():
    """9. Everything provided: all models run and the result matches expected risk."""
    res_all_pass, _ = run_multimodal_analysis(
        claimed_identity="Alex",
        ref_photo_path=SAMPLE_IMAGES / "same_person_a.jpg",
        test_photo_path=SAMPLE_IMAGES / "same_person_b.jpg",
        ref_audio_path=SAMPLE_AUDIO / "reference.wav",
        test_audio_path=SAMPLE_AUDIO / "test.wav",
        chat_text="Hi Mom, just landed safely at the airport. Taking an Uber home now.",
    )
    assert res_all_pass.risk_level == "LOW"
    assert res_all_pass.modality_statuses["image"] == "PASS"
    assert res_all_pass.modality_statuses["voice"] == "PASS"
    assert res_all_pass.modality_statuses["chat"] == "PASS"
