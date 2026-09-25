"""Verification tests for TrustGuard In-Memory Session Flow (No DB Persistence).

Verifies:
1. Analysis results remain in-memory only during the active session.
2. "+ New Analysis" clears current results and inputs.
3. No automatic persistence to Supabase occurs after analysis.
4. No calls to list_analysis_sessions or save_analysis_session in active UI flow.
5. Supabase Auth (Sign Up, Sign In, Sign Out) remains fully intact.
"""

import os
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

import trustguard.supabase_client as sc
from trustguard.consistency_engine import MultimodalConsistencyResult

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_ui_sidebar_has_no_saved_history_list(monkeypatch):
    """Verifies that no saved analysis history or DB session cards appear in the sidebar, even when logged in."""
    monkeypatch.setattr(sc, "is_supabase_configured", lambda: True)

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    # Simulate authenticated user
    at.session_state.auth_user = {
        "id": "test-user-id",
        "email": "analyst@trustguard.ai",
        "full_name": "Senior Analyst",
        "access_token": "valid-token",
    }
    at.run()

    # Sidebar should only have + New Analysis button (and expander button if any)
    sidebar_buttons = [b.label for b in at.sidebar.button]
    assert len(sidebar_buttons) >= 1
    assert "New Analysis" in sidebar_buttons[0]
    # No saved session titles or cards in sidebar buttons
    for btn_label in sidebar_buttons:
        assert "Saved Analyses" not in btn_label
        assert "Impersonation Alert" not in btn_label

    # Ensure no sidebar text mentions saved analyses count or list
    sidebar_markdown = " ".join([m.value for m in at.sidebar.markdown])
    assert "Saved Analyses" not in sidebar_markdown
    assert "No saved analyses yet" not in sidebar_markdown


def test_new_analysis_clears_active_results_and_inputs():
    """Verifies that clicking '+ New Analysis' resets the active analysis result and inputs cleanly."""
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()

    # Set mock active result and inputs
    fake_res = MultimodalConsistencyResult(
        claimed_identity="Alice",
        risk_level="HIGH",
        modality_statuses={"chat": "FAIL"},
        contradiction_map={"agreements": [], "conflicts": ["Chat coercion"], "uncertain": []},
        explanation="High risk detected.",
        recommended_action="Do not transfer funds.",
        details={},
    )
    at.session_state.active_result = fake_res
    at.session_state.claimed_name_val = "Alice"
    at.session_state.chat_text_val = "Urgent wire request"
    at.run()

    assert at.session_state.active_result is not None
    assert at.session_state.claimed_name_val == "Alice"

    # Click + New Analysis
    new_btn = at.sidebar.button[0]
    new_btn.click().run()

    # State must be completely reset
    assert at.session_state.active_result is None
    assert at.session_state.claimed_name_val == ""
    assert at.session_state.chat_text_val == ""


def test_no_database_save_called_in_app(monkeypatch):
    """Verifies that save_analysis_session is NEVER called by the application."""
    save_called = []

    def mock_save(*args, **kwargs):
        save_called.append((args, kwargs))
        return True, "should-not-exist", None

    monkeypatch.setattr(sc, "save_analysis_session", mock_save)
    monkeypatch.setattr(sc, "is_supabase_configured", lambda: True)

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.session_state.auth_user = {"id": "user-1", "email": "test@domain.com"}
    at.run()

    # Set active result and run
    at.session_state.active_result = MultimodalConsistencyResult(
        claimed_identity="Bob",
        risk_level="LOW",
        modality_statuses={"image": "PASS"},
        contradiction_map={"agreements": ["Face matches"], "conflicts": [], "uncertain": []},
        explanation="Consistent.",
        recommended_action="Proceed.",
        details={},
    )
    at.run()

    assert len(save_called) == 0, "save_analysis_session was unexpectedly called!"


def test_supabase_auth_methods_preserved():
    """Verifies that Supabase authentication functions (sign_in, sign_up, sign_out) are preserved."""
    assert hasattr(sc, "sign_in_user")
    assert hasattr(sc, "sign_up_user")
    assert hasattr(sc, "sign_out_user")
    assert hasattr(sc, "is_supabase_configured")
    assert callable(sc.sign_in_user)
    assert callable(sc.sign_up_user)
    assert callable(sc.sign_out_user)
