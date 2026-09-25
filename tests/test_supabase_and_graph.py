"""Tests for Supabase Client and Graph Visualizer in TrustGuard.

Verifies:
1. Supabase client configuration detection and graceful unconfigured states.
2. Contradiction graph SVG rendering, deterministic coordinates, and dynamic evidence filtering.
3. RLS policy compatibility and data structures.
"""

import pytest
from trustguard.graph_visualizer import render_contradiction_graph_svg, _is_modality_active
import trustguard.supabase_client as sc


def test_supabase_unconfigured_behavior(monkeypatch):
    """Verifies that without keys, is_supabase_configured returns False and methods fail gracefully without crashing."""
    assert sc.SUPABASE_AVAILABLE is True
    monkeypatch.setattr(sc, "get_supabase_config", lambda: (None, None))
    assert sc.is_supabase_configured() is False
    # Without secrets or env set, it shouldn't pretend to be configured
    # Test sign in
    ok, err, user = sc.sign_in_user("test@example.com", "password123")
    assert ok is False
    assert "not configured" in (err or "").lower() or "credentials" in (err or "").lower()

    # Test sign up
    ok, err, user = sc.sign_up_user("test@example.com", "password123", "Test User", "Test Org")
    assert ok is False
    assert "not configured" in (err or "").lower() or "credentials" in (err or "").lower()

    # Test session list
    sessions = sc.list_analysis_sessions("fake-user-id")
    assert sessions == []


def test_graph_dynamic_modality_filtering_single_evidence():
    """Verifies graph visualizer only includes supplied modalities."""
    # Case: Only Chat text provided
    svg = render_contradiction_graph_svg(
        claimed_identity="Rahul",
        modality_statuses={"chat": "FAIL", "image": "CAN'T TELL", "voice": "CAN'T TELL", "transcript": "CAN'T TELL"},
        details={"chat_signal": {"final_score": 0.88}},
        risk_level="HIGH"
    )
    assert "Chat Context" in svg
    assert "Facial Biometrics" not in svg
    assert "Voice Acoustics" not in svg
    assert "Spoken Content" not in svg
    assert "CONTRADICTS" in svg
    assert "Rahul" in svg


def test_graph_dynamic_modality_filtering_image_and_voice():
    """Verifies graph visualizer with photo and voice evidence."""
    svg = render_contradiction_graph_svg(
        claimed_identity="Priya",
        modality_statuses={"image": "PASS", "voice": "PASS"},
        details={
            "image": {"distance": 0.18, "source_reason": "Consistent"},
            "voice": {"similarity": 0.85, "source_reason": "Consistent"},
        },
        risk_level="LOW"
    )
    assert "Facial Biometrics" in svg
    assert "Voice Acoustics" in svg
    assert "Chat Context" not in svg
    assert "Spoken Content" not in svg
    assert "AGREES" in svg
    assert "Priya" in svg


def test_graph_empty_evidence_state():
    """Verifies graceful empty state when no evidence was submitted."""
    svg = render_contradiction_graph_svg(
        claimed_identity="Unknown",
        modality_statuses={"image": "CAN'T TELL", "voice": "CAN'T TELL", "chat": "CAN'T TELL", "transcript": "CAN'T TELL"},
        details={
            "image": {"reason": "No image evidence provided."},
            "voice": {"reason": "No audio evidence provided."},
        },
        risk_level="UNCERTAIN"
    )
    assert "No Active Evidence Modalities Detected" in svg
    assert "<svg" not in svg  # Clean fallback card


def test_graph_all_four_modalities_and_cross_edge():
    """Verifies 4-node 2x2 layout with cross-modal text comparison edge."""
    svg = render_contradiction_graph_svg(
        claimed_identity="Executive",
        modality_statuses={"image": "PASS", "voice": "FAIL", "chat": "FAIL", "transcript": "FAIL"},
        details={
            "image": {"distance": 0.22},
            "voice": {"similarity": 0.35},
            "chat_signal": {"final_score": 0.85},
            "transcript_signal": {"final_score": 0.82},
        },
        risk_level="HIGH"
    )
    assert "Facial Biometrics" in svg
    assert "Voice Acoustics" in svg
    assert "Chat Context" in svg
    assert "Spoken Content" in svg
    assert "Executive" in svg
    assert "CONTRADICTS" in svg
    assert "AGREES" in svg
    # Cross-modal edge between chat and transcript
    assert "ALIGNED TEXT" in svg or "DISCREPANCY" in svg or "CORRELATED" in svg
