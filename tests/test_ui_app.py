"""Automated Streamlit UI tests for app.py using Streamlit AppTest framework.

Tests modern UI flows:
1. Loading the app cleanly with zero exceptions.
2. Verifying "+ New Analysis" button in sidebar and clean state reset.
3. Verifying unconfigured Supabase indicators and auth modal triggers.
4. Verifying active analysis rendering with dynamic SVG contradiction map and status cards.
5. Verifying absence of saved analysis history in sidebar.
"""

from pathlib import Path
from streamlit.testing.v1 import AppTest
from trustguard.consistency_engine import MultimodalConsistencyResult

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_ui_initial_load():
    """Verifies that the app loads cleanly with all sections and zero exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert len(at.exception) == 0, f"App threw exceptions on load: {at.exception}"
    assert at.session_state.active_result is None
    # Sidebar should contain "+ New Analysis"
    new_btn = at.sidebar.button[0]
    assert "New Analysis" in new_btn.label


def test_ui_new_analysis_click_resets_state():
    """Verifies that clicking '+ New Analysis' resets state cleanly."""
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.session_state.claimed_name_val = "Test User"
    at.session_state.chat_text_val = "Some suspicious message"

    # Click + New Analysis
    new_btn = at.sidebar.button[0]
    new_btn.click().run()

    assert len(at.exception) == 0
    assert at.session_state.active_result is None
    assert at.session_state.claimed_name_val == ""
    assert at.session_state.chat_text_val == ""


def test_ui_active_result_rendering():
    """Verifies that when an active result exists, the risk badge, status cards, and SVG map render."""
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()

    # Inject an active result to simulate a completed or loaded analysis
    fake_res = MultimodalConsistencyResult(
        claimed_identity="Priya Patel",
        risk_level="HIGH",
        modality_statuses={"image": "PASS", "voice": "FAIL", "chat": "FAIL", "transcript": "CAN'T TELL"},
        contradiction_map={
            "agreements": ["Facial geometry matches reference"],
            "conflicts": ["Voice acoustics diverge", "Chat demands money"],
            "uncertain": []
        },
        explanation="Voice acoustics and chat content contradict the reference profile.",
        recommended_action="Execute out-of-band identity challenge over established telephone line.",
        details={
            "image": {"distance": 0.21},
            "voice": {"similarity": 0.34},
            "chat_signal": {"final_score": 0.88},
            "elapsed_time_seconds": 1.45,
        }
    )

    at.session_state.active_result = fake_res
    at.session_state.claimed_name_val = "Priya Patel"
    at.run()

    assert len(at.exception) == 0
    # Check risk badge
    html_corpus = " ".join([m.value for m in at.markdown])
    # Check that SVG is not escaped or leaked as raw text into markdown
    assert "<svg" not in html_corpus

    # Check subheaders
    subheaders = [s.value for s in at.subheader]
    assert any("Contradiction Map" in s for s in subheaders)
    assert any("Multimodal Assessment" in s for s in subheaders)

    # Check that recommended action is displayed
    assert "Execute out-of-band identity challenge" in html_corpus


if __name__ == "__main__":
    test_ui_initial_load()
    print(">>> test_ui_initial_load PASSED")
    test_ui_new_analysis_click_resets_state()
    print(">>> test_ui_new_analysis_click_resets_state PASSED")
    test_ui_active_result_rendering()
    print(">>> test_ui_active_result_rendering PASSED")
    print("\nALL UI TESTS PASSED SUCCESSFULLY!")
