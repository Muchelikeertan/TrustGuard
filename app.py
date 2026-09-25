"""TrustGuard — Multimodal Identity Consistency Platform.

Project PS-02 | Innovators Conclave 2026
Tagline: "Do the image, voice and conversation tell the same story?"

Fuses facial biometrics (DeepFace), speaker acoustics (Resemblyzer), speech-to-text
transcription (Whisper), and NLP threat classification (BART) to detect impersonation
and coercion.

Features real Supabase Authentication and a deterministic dynamic contradiction map.
"""

import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 1. Windows UTF-8 console output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 2. Legacy Keras configuration for TensorFlow 2.16+ / DeepFace compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 3. Ensure ffmpeg is discovered on Windows
if not shutil.which("ffmpeg"):
    winget_ffmpeg_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    for p in winget_ffmpeg_dir.glob("Gyan.FFmpeg*/**/bin"):
        if (p / "ffmpeg.exe").exists():
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
            break

import streamlit as st
import streamlit.components.v1 as components

import trustguard.supabase_client as sc
from trustguard.chat_check import get_classifier
from trustguard.consistency_engine import (
    MultimodalConsistencyResult,
    evaluate_consistency,
    load_thresholds,
)
from trustguard.graph_visualizer import render_contradiction_graph_svg
from trustguard.pipeline import get_voice_encoder, get_whisper_model, run_multimodal_analysis


# ---------------------------------------------------------------------------
# Strict Text Sanitizer (Never use 'verified', 'confirmed', 'hallmark')
# ---------------------------------------------------------------------------

def sanitize_ui_text(text: Optional[str]) -> str:
    """Ensures forbidden words are replaced with approved decision-support phrasing."""
    if not text:
        return ""
    clean = re.sub(r"\bverified\b", "consistent with the reference", text, flags=re.IGNORECASE)
    clean = re.sub(r"\bconfirmed\b", "consistent with the reference", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bverifications?\b", "consistency checks", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bhallmark\b", "characteristic pattern", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bdeepfake detected\b", "pattern consistent with impersonation", clean, flags=re.IGNORECASE)
    return clean


# ---------------------------------------------------------------------------
# Thresholds Loader (Dynamic from config/thresholds.yaml)
# ---------------------------------------------------------------------------

THRESHOLDS_PATH = Path(__file__).resolve().parent / "config" / "thresholds.yaml"
thresholds = load_thresholds(THRESHOLDS_PATH)


# ---------------------------------------------------------------------------
# Streamlit Page Configuration & Professional Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TrustGuard | Multimodal Identity Consistency",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Header Layout */
    .top-header-container {
        padding: 0.6rem 0 1.2rem 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.2rem;
    }
    .brand-title {
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .brand-tagline {
        font-size: 0.98rem !important;
        font-weight: 500 !important;
        color: #475569;
        margin-top: 0.25rem;
    }

    /* Auth Badge Area in Top Right */
    .user-profile-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.65rem;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 30px;
        padding: 0.35rem 0.85rem;
        font-size: 0.88rem;
        color: #0f172a;
        font-weight: 600;
    }
    .user-avatar-circle {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        background: #0284c7;
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 800;
    }

    /* How it works Strip */
    .how-it-works-strip {
        display: flex;
        justify-content: space-between;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.65rem 1.2rem;
        margin-bottom: 1.2rem;
        gap: 0.8rem;
    }
    .step-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.88rem;
        color: #334155;
        font-weight: 600;
    }
    .step-number {
        background: #0284c7;
        color: #ffffff;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.76rem;
        font-weight: 700;
    }

    /* Decision Support Notice */
    .decision-banner {
        background-color: #f1f5f9;
        border-left: 4px solid #475569;
        padding: 0.6rem 0.9rem;
        font-size: 0.86rem;
        color: #1e293b;
        border-radius: 4px;
        margin-bottom: 1.2rem;
        font-weight: 500;
    }

    /* Risk Badges */
    .risk-badge-box {
        padding: 1.1rem 1.5rem;
        border-radius: 8px;
        text-align: center;
        margin: 1.2rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .risk-badge-title {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: 0.03em;
    }
    .risk-badge-subtitle {
        font-size: 0.92rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    .risk-high {
        background-color: #fee2e2;
        color: #991b1b;
        border: 2px solid #ef4444;
    }
    .risk-uncertain {
        background-color: #fef3c7;
        color: #92400e;
        border: 2px solid #f59e0b;
    }
    .risk-low {
        background-color: #d1fae5;
        color: #065f46;
        border: 2px solid #10b981;
    }

    /* Modality Status Cards */
    .status-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        min-height: 145px;
    }
    .status-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .status-title {
        font-weight: 700;
        font-size: 0.96rem;
        color: #0f172a;
    }
    .status-score {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 600;
        background: #f1f5f9;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
    }
    .pill {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 0.45rem;
    }
    .pill-pass { background-color: #dcfce7; color: #15803d; border: 1px solid #86efac; }
    .pill-warning { background-color: #fef9c3; color: #a16207; border: 1px solid #fde047; }
    .pill-fail { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
    .pill-cant-tell { background-color: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
    .status-reason {
        font-size: 0.84rem;
        color: #334155;
        line-height: 1.4;
    }

    /* Action Callout Boxes */
    .action-box {
        border-radius: 8px;
        padding: 1rem 1.3rem;
        margin-top: 1.2rem;
    }
    .action-box-high {
        background-color: #fff1f2;
        border: 2px solid #e11d48;
        color: #881337;
    }
    .action-box-uncertain {
        background-color: #fffbeb;
        border: 2px solid #d97706;
        color: #78350f;
    }
    .action-box-low {
        background-color: #f0fdf4;
        border: 2px solid #16a34a;
        color: #14532d;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

if "active_result" not in st.session_state:
    st.session_state.active_result = None

if "claimed_name_val" not in st.session_state:
    st.session_state.claimed_name_val = ""

if "chat_text_val" not in st.session_state:
    st.session_state.chat_text_val = ""

if "analysis_time" not in st.session_state:
    st.session_state.analysis_time = None

if "active_warnings" not in st.session_state:
    st.session_state.active_warnings = []


# ---------------------------------------------------------------------------
# Authentication Dialogs (Supabase Auth Modal Controls)
# ---------------------------------------------------------------------------

@st.dialog("Sign In to TrustGuard")
def show_signin_dialog():
    st.markdown("Enter your registered credentials to access TrustGuard.")
    email = st.text_input("Work Email Address", key="signin_email", placeholder="analyst@domain.com")
    password = st.text_input("Password", type="password", key="signin_password")

    err_container = st.empty()

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Sign In", type="primary", use_container_width=True):
            if not email or not password:
                err_container.error("Please enter both email and password.")
            else:
                with st.spinner("Authenticating with Supabase..."):
                    success, err, user_info = sc.sign_in_user(email, password)
                if success:
                    st.session_state.auth_user = user_info
                    st.rerun()
                else:
                    err_container.error(err or "Sign in failed.")
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


@st.dialog("Create a TrustGuard Account")
def show_signup_dialog():
    st.markdown("Register an authorized analyst account to securely persist multimodal analyses.")
    full_name = st.text_input("Full Name", key="signup_name", placeholder="Jane Doe")
    org = st.text_input("Organization (Optional)", key="signup_org", placeholder="Cyber Defense Unit / Fraud Risk")
    email = st.text_input("Work Email Address", key="signup_email", placeholder="analyst@domain.com")
    password = st.text_input("Password (min 6 characters)", type="password", key="signup_password")
    confirm_pwd = st.text_input("Confirm Password", type="password", key="signup_confirm_password")

    err_container = st.empty()

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Create Account", type="primary", use_container_width=True):
            if not email or not password:
                err_container.error("Email and password are required.")
            elif password != confirm_pwd:
                err_container.error("Passwords do not match.")
            elif len(password) < 6:
                err_container.error("Password must be at least 6 characters.")
            else:
                with st.spinner("Registering with Supabase Auth..."):
                    success, err, user_info = sc.sign_up_user(
                        email=email,
                        password=password,
                        full_name=full_name,
                        organization=org,
                    )
                if success:
                    st.session_state.auth_user = user_info
                    st.success("Account created successfully!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    err_container.error(err or "Registration failed.")
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


# ---------------------------------------------------------------------------
# Header Section with Top-Right Authentication Controls
# ---------------------------------------------------------------------------

col_brand, col_auth = st.columns([7, 3])

with col_brand:
    st.markdown(
        """
        <div class="brand-title">🛡️ TrustGuard</div>
        <div class="brand-tagline">AI for Digital Trust — Multimodal Identity Consistency Platform</div>
        """,
        unsafe_allow_html=True,
    )

with col_auth:
    supabase_configured = sc.is_supabase_configured()

    if not supabase_configured:
        # Prompt for configuration without pretending fake auth works
        st.markdown(
            """
            <div style="text-align: right; padding-top: 0.2rem;">
                <span style="background: #fef3c7; color: #92400e; border: 1px solid #fde047; padding: 0.3rem 0.65rem; border-radius: 20px; font-size: 0.78rem; font-weight: 700;">
                    ⚠️ Supabase Unconfigured
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.popover("⚙️ Supabase Setup Instructions"):
            st.markdown(
                """
                **Authentication Setup:**
                
                TrustGuard requires Supabase for secure user authentication.
                
                Add your Supabase credentials to `.streamlit/secrets.toml` or environment variables:
                ```toml
                SUPABASE_URL = "https://your-project-id.supabase.co"
                SUPABASE_KEY = "your-anon-publishable-key"
                ```
                """
            )
    else:
        # Supabase is configured: show Auth controls
        user = st.session_state.auth_user
        if user:
            # Authenticated User Profile Area
            u_email = user.get("email", "")
            u_name = user.get("full_name") or u_email.split("@")[0]
            initial = u_name[0].upper() if u_name else "U"

            c_info, c_btn = st.columns([2, 1])
            with c_info:
                st.markdown(
                    f"""
                    <div style="text-align: right; padding-top: 0.2rem;">
                        <span class="user-profile-badge">
                            <span class="user-avatar-circle">{initial}</span>
                            <span>{u_name}</span>
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_btn:
                if st.button("Log Out", key="logout_btn", use_container_width=True):
                    sc.sign_out_user()
                    st.session_state.auth_user = None
                    st.session_state.active_result = None
                    st.session_state.claimed_name_val = ""
                    st.session_state.chat_text_val = ""
                    st.session_state.active_warnings = []
                    st.session_state.analysis_time = None
                    if "claimed_name_input" in st.session_state:
                        st.session_state.claimed_name_input = ""
                    if "chat_text_input" in st.session_state:
                        st.session_state.chat_text_input = ""
                    st.rerun()
        else:
            # Logged Out State: Show Login & Sign Up buttons on Top-Right
            c_in, c_up = st.columns(2)
            with c_in:
                if st.button("Sign In", key="btn_open_signin", use_container_width=True):
                    show_signin_dialog()
            with c_up:
                if st.button("Sign Up", key="btn_open_signup", type="primary", use_container_width=True):
                    show_signup_dialog()

st.markdown("<hr style='margin: 0.8rem 0 1.2rem 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Left Sidebar: Navigation & Controls
# ---------------------------------------------------------------------------

st.sidebar.markdown("### 🛡️ TrustGuard")
st.sidebar.caption("AI for Digital Trust | PS-02 Conclave 2026")

# Prominent "+ New Analysis" button at the top
if st.sidebar.button("➕ New Analysis", type="primary", use_container_width=True):
    st.session_state.active_result = None
    st.session_state.claimed_name_val = ""
    st.session_state.chat_text_val = ""
    st.session_state.active_warnings = []
    st.session_state.analysis_time = None
    if "claimed_name_input" in st.session_state:
        st.session_state.claimed_name_input = ""
    if "chat_text_input" in st.session_state:
        st.session_state.chat_text_input = ""
    st.rerun()

st.sidebar.markdown("---")

# Model pre-warming tool in sidebar
with st.sidebar.expander("⚡ Model Diagnostics & Pre-Warming"):
    st.caption("Active YAML Verification Thresholds:")
    st.markdown(
        f"""
        - **Face Pass Dist:** `≤ {thresholds['image']['pass_max_distance']}`
        - **Face Fail Dist:** `≥ {thresholds['image']['fail_min_distance']}`
        - **Voice Pass Sim:** `≥ {thresholds['voice']['pass_min_similarity']}`
        - **Voice Fail Sim:** `< {thresholds['voice']['fail_max_similarity']}`
        - **Chat Suspicious:** `≥ {thresholds['chat']['suspicious_min']}`
        - **Chat High Pressure:** `≥ {thresholds['chat']['high_pressure_min']}`
        """
    )
    if st.button("Pre-Warm AI Models into Memory", use_container_width=True):
        with st.status("Pre-warming models..."):
            get_voice_encoder()
            get_whisper_model("base")
            get_classifier()
        st.success("All AI models pre-warmed!")


# ---------------------------------------------------------------------------
# Main Panel: Information Strips & Notice
# ---------------------------------------------------------------------------

# How it Works Strip
st.markdown(
    """
    <div class="how-it-works-strip">
        <div class="step-item"><div class="step-number">1</div> <b>Add Evidence</b> (Photos, Voice, Chat)</div>
        <div class="step-item"><div class="step-number">2</div> <b>Analyze</b> (Real AI Models)</div>
        <div class="step-item"><div class="step-number">3</div> <b>Consistency Graph</b> (Dynamic Map)</div>
        <div class="step-item"><div class="step-number">4</div> <b>Risk & Action</b> (Actionable guidance)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Permanent Decision Support Notice
st.markdown(
    """
    <div class="decision-banner">
        ⚖️ <b>Decision Support Protocol:</b> TrustGuard computes multimodal consistency signals to assist fraud analysts and users. Human review remains mandatory for consequential security decisions.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Evidence Inputs (Sections with helpful captions)
# ---------------------------------------------------------------------------

with st.container():
    st.markdown("#### 📥 Submit Evidence for Analysis")
    st.caption("Provide whatever evidence is available. Any input left empty will be evaluated as 'CAN'T TELL'.")

    col_id, col_chat = st.columns([1, 2])
    with col_id:
        claimed_name = st.text_input(
            "🪪 Claimed Identity Name",
            value=st.session_state.claimed_name_val,
            placeholder="e.g. Rahul, Boss, Family Member",
            help="The identity being claimed in the incoming communication.",
            key="claimed_name_input",
        )
    with col_chat:
        chat_text = st.text_area(
            "💬 Incoming Chat Message / Communication",
            value=st.session_state.chat_text_val,
            placeholder="Paste text message, WhatsApp, email, or coercion attempt here...",
            height=68,
            help="The text message received from the contact.",
            key="chat_text_input",
        )

    col_img, col_aud = st.columns(2)
    with col_img:
        st.markdown("##### 🖼️ Facial Imagery Evidence")
        st.caption("Upload clear frontal face images (jpg/png) to run DeepFace cosine distance analysis.")
        ref_photo = st.file_uploader("Reference Photo (Known trusted photo)", type=["jpg", "jpeg", "png"], key="ref_photo_input")
        test_photo = st.file_uploader("Test Photo (Incoming / Unverified photo)", type=["jpg", "jpeg", "png"], key="test_photo_input")

    with col_aud:
        st.markdown("##### 🎙️ Voice Audio Evidence")
        st.caption("Upload speech recordings (wav/mp3/ogg) for Resemblyzer speaker acoustics and Whisper STT.")
        ref_audio = st.file_uploader("Reference Voice Clip (Known trusted audio)", type=["wav", "mp3", "ogg"], key="ref_audio_input")
        test_audio = st.file_uploader("Test Voice Clip (Incoming / Voicemail audio)", type=["wav", "mp3", "ogg"], key="test_audio_input")

    analyze_clicked = st.button("🔍 Analyze Multimodal Consistency", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Live Pipeline Execution & Automatic Supabase Persistence
# ---------------------------------------------------------------------------

if analyze_clicked:
    st.session_state.claimed_name_val = claimed_name
    st.session_state.chat_text_val = chat_text
    st.session_state.active_warnings = []

    has_any_evidence = bool(ref_photo or test_photo or ref_audio or test_audio or (chat_text and chat_text.strip()))
    if not has_any_evidence:
        st.info("ℹ️ Please add at least one piece of evidence (photo, voice clip, or chat message) before analyzing.")
    else:
        temp_dir = tempfile.mkdtemp(prefix="trustguard_run_")

        progress_bar = st.progress(0)
        status_label = st.empty()

        def update_progress(message: str, pct: float):
            progress_bar.progress(int(pct * 100))
            status_label.markdown(f"**{message}**")

        try:
            # Save uploads into temp dir if provided
            p_ref_img = None
            p_test_img = None
            if ref_photo:
                p_ref_img = os.path.join(temp_dir, f"ref_{ref_photo.name}")
                with open(p_ref_img, "wb") as f:
                    f.write(ref_photo.getbuffer())
            if test_photo:
                p_test_img = os.path.join(temp_dir, f"test_{test_photo.name}")
                with open(p_test_img, "wb") as f:
                    f.write(test_photo.getbuffer())

            p_ref_aud = None
            p_test_aud = None
            if ref_audio:
                p_ref_aud = os.path.join(temp_dir, f"ref_{ref_audio.name}")
                with open(p_ref_aud, "wb") as f:
                    f.write(ref_audio.getbuffer())
            if test_audio:
                p_test_aud = os.path.join(temp_dir, f"test_{test_audio.name}")
                with open(p_test_aud, "wb") as f:
                    f.write(test_audio.getbuffer())

            analysis_result, warnings = run_multimodal_analysis(
                claimed_identity=claimed_name or "Unspecified Identity",
                ref_photo_path=p_ref_img,
                test_photo_path=p_test_img,
                ref_audio_path=p_ref_aud,
                test_audio_path=p_test_aud,
                chat_text=chat_text,
                progress_callback=update_progress,
                thresholds_path=THRESHOLDS_PATH,
            )

            st.session_state.active_result = analysis_result
            st.session_state.active_warnings = warnings
            st.session_state.analysis_time = analysis_result.details.get("elapsed_time_seconds", 0.0)

            progress_bar.progress(100)
            status_label.success(
                f"✅ Multimodal Analysis complete in {st.session_state.analysis_time:.2f}s! Assessment details rendered below."
            )

        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)}" if str(e) else type(e).__name__
            print(f"[Fatal Live Pipeline Exception] {err_msg}")
            traceback.print_exc()
            progress_bar.empty()
            status_label.error(f"❌ Analysis failed: {err_msg}")
            st.error(f"⚠️ Analysis Pipeline Error: {err_msg}")
            with st.expander("🔍 View Error Details & Full Traceback", expanded=True):
                st.exception(e)
                st.code(traceback.format_exc(), language="python")
        finally:
            # Strict privacy cleanup in finally block
            shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Results Display Section
# ---------------------------------------------------------------------------

if st.session_state.active_result is not None:
    res = st.session_state.active_result
    # Support both dataclass and dictionary
    claimed_id = res.claimed_identity if hasattr(res, "claimed_identity") else res.get("claimed_identity", "Unknown")
    risk = (res.risk_level if hasattr(res, "risk_level") else res.get("risk_level", "LOW")).upper()
    mod_statuses = res.modality_statuses if hasattr(res, "modality_statuses") else res.get("modality_statuses", {})
    details = res.details if hasattr(res, "details") else res.get("details", {})
    explanation_raw = res.explanation if hasattr(res, "explanation") else res.get("explanation", "")
    action_raw = res.recommended_action if hasattr(res, "recommended_action") else res.get("recommended_action", "")
    cmap = res.contradiction_map if hasattr(res, "contradiction_map") else res.get("contradiction_map", {})

    st.markdown("---")
    st.subheader(f"📊 Multimodal Assessment: {claimed_id}")

    if st.session_state.analysis_time is not None:
        st.caption(f"⏱️ Full multimodal analysis executed in {st.session_state.analysis_time:.2f} seconds.")

    # Any model warnings displayed non-fatally
    if st.session_state.active_warnings:
        for w in st.session_state.active_warnings:
            st.warning(f"⚠️ Modality Notice: {w}")

    # 1. Color-Coded Risk Badge
    if risk == "HIGH":
        badge_html = """
        <div class="risk-badge-box risk-high">
            <div class="risk-badge-title">🚨 HIGH RISK — MULTIMODAL CONTRADICTION DETECTED</div>
            <div class="risk-badge-subtitle">Cross-modal signals diverge or indicate coercive social engineering tactics.</div>
        </div>
        """
        action_class = "action-box-high"
    elif risk == "UNCERTAIN":
        badge_html = """
        <div class="risk-badge-box risk-uncertain">
            <div class="risk-badge-title">⚠️ UNCERTAIN RISK — INSUFFICIENT OR CONFLICTING SIGNALS</div>
            <div class="risk-badge-subtitle">Evidence is incomplete or ambiguous. Independent out-of-band checks required.</div>
        </div>
        """
        action_class = "action-box-uncertain"
    else:
        badge_html = """
        <div class="risk-badge-box risk-low">
            <div class="risk-badge-title">✅ LOW RISK — EVIDENCE IS CONSISTENT WITH REFERENCE</div>
            <div class="risk-badge-subtitle">Evaluated modalities consistently align with the reference identity profile.</div>
        </div>
        """
        action_class = "action-box-low"

    st.markdown(badge_html, unsafe_allow_html=True)

    # 2. Four Modality Status Cards (Image, Voice, Chat, Transcript)
    col1, col2, col3, col4 = st.columns(4)

    card_definitions = [
        ("Image (DeepFace)", mod_statuses.get("image", "CAN'T TELL"), col1, "image"),
        ("Voice (Resemblyzer)", mod_statuses.get("voice", "CAN'T TELL"), col2, "voice"),
        ("Chat (BART Zero-Shot)", mod_statuses.get("chat", "CAN'T TELL"), col3, "chat"),
        ("Transcript (Whisper)", mod_statuses.get("transcript", "CAN'T TELL"), col4, "transcript"),
    ]

    for title, status, col, key in card_definitions:
        pill_class = {
            "PASS": "pill-pass",
            "WARNING": "pill-warning",
            "FAIL": "pill-fail",
            "CAN'T TELL": "pill-cant-tell",
        }.get(status, "pill-cant-tell")

        if key == "image":
            img_det = details.get("image", {})
            dist = img_det.get("distance")
            score_str = f"Dist: {dist:.3f}" if dist is not None else "No Data"
            reason_str = img_det.get("source_reason") or (
                "Facial geometry is consistent with reference." if status == "PASS" else
                "Facial geometry diverges from reference." if status == "FAIL" else
                "Borderline facial consistency." if status == "WARNING" else
                "No photo comparison available."
            )
        elif key == "voice":
            voc_det = details.get("voice", {})
            sim = voc_det.get("similarity")
            score_str = f"Sim: {sim:.3f}" if sim is not None else "No Data"
            reason_str = voc_det.get("source_reason") or (
                "Voice acoustics are consistent with reference." if status == "PASS" else
                "Voice acoustics diverge from reference speaker." if status == "FAIL" else
                "Borderline voice acoustic similarity." if status == "WARNING" else
                "No voice comparison available."
            )
        elif key == "chat":
            chat_det = details.get("chat_signal", {})
            f_score = chat_det.get("final_score")
            score_str = f"Score: {f_score:.2f}" if f_score is not None else "No Data"
            reason_str = (
                "Message reflects normal everyday conversation." if status == "PASS" else
                "High-pressure financial or isolation demand detected." if status == "FAIL" else
                "Moderate urgency or suspicious phrasing cues." if status == "WARNING" else
                "No text message provided."
            )
        else:  # transcript
            tr_det = details.get("transcript_signal", {})
            f_score = tr_det.get("final_score")
            score_str = f"Score: {f_score:.2f}" if f_score is not None else "No Data"
            reason_str = (
                "Spoken words reflect normal conversation." if status == "PASS" else
                "Spoken words exhibit high-pressure coercion cues." if status == "FAIL" else
                "Moderate spoken urgency cues." if status == "WARNING" else
                "No audio transcript available."
            )

        clean_reason = sanitize_ui_text(reason_str)

        with col:
            st.markdown(
                f"""
                <div class="status-card">
                    <div class="status-header">
                        <span class="status-title">{title}</span>
                        <span class="status-score">{score_str}</span>
                    </div>
                    <div>
                        <span class="pill {pill_class}">{status}</span>
                    </div>
                    <div class="status-reason">{clean_reason}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 3. Dynamic Deterministic Contradiction Map
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🗺️ Multimodal Contradiction Map")
    st.caption("Deterministic graph visualization of supplied evidence relationships and cross-modal consistency.")

    # Render dynamic SVG graph (filters only actually supplied modalities)
    svg_content = render_contradiction_graph_svg(
        claimed_identity=claimed_id,
        modality_statuses=mod_statuses,
        details=details,
        risk_level=risk,
    )
    # Render dynamic SVG graph via Streamlit components to prevent markdown escaping
    if "<svg" in svg_content:
        # Determine iframe height dynamically from SVG viewBox to prevent any vertical scrollbars or clipping
        vb_match = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg_content)
        svg_h = int(vb_match.group(2)) if vb_match else 420
        iframe_h = svg_h + 24

        html_wrapper = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}
  body {{
    background: transparent;
    overflow: hidden;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding-top: 4px;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
  }}
  svg {{
    width: 100%;
    max-width: 820px;
    height: auto;
    display: block;
  }}
</style>
</head>
<body>
{svg_content}
</body>
</html>"""
        components.html(html_wrapper, height=iframe_h, scrolling=False)
    else:
        # Empty state or non-SVG fallback banner
        components.html(f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0;
    padding: 4px;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: transparent;
  }}
</style>
</head>
<body>
{svg_content}
</body>
</html>""", height=160, scrolling=False)

    # Structured Contradiction Details
    col_ag, col_cf, col_un = st.columns(3)

    with col_ag:
        st.markdown(
            """
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; min-height: 120px;">
                <b>🤝 Agreements</b><br>
            """,
            unsafe_allow_html=True,
        )
        agreements = cmap.get("agreements", [])
        if agreements:
            for item in agreements:
                st.markdown(f"- ✅ {sanitize_ui_text(item)}")
        else:
            st.markdown("<span style='color: #64748b; font-size: 0.88rem;'>No cross-modal agreements noted.</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_cf:
        st.markdown(
            """
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; min-height: 120px;">
                <b>⚡ Conflicts & Contradictions</b><br>
            """,
            unsafe_allow_html=True,
        )
        conflicts = cmap.get("conflicts", [])
        if conflicts:
            for item in conflicts:
                st.markdown(f"- ⚠️ **{sanitize_ui_text(item)}**")
        else:
            st.markdown("<span style='color: #64748b; font-size: 0.88rem;'>No cross-modal conflicts detected.</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_un:
        st.markdown(
            """
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; min-height: 120px;">
                <b>❓ Uncertain Elements</b><br>
            """,
            unsafe_allow_html=True,
        )
        uncertain = cmap.get("uncertain", [])
        if uncertain:
            for item in uncertain:
                st.markdown(f"- ⚪ {sanitize_ui_text(item)}")
        else:
            st.markdown("<span style='color: #64748b; font-size: 0.88rem;'>All submitted modalities evaluated.</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 4. Explanation & Highlighted Recommended Next Action
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("💡 Analysis Explanation")
    clean_explanation = sanitize_ui_text(explanation_raw)
    st.write(clean_explanation)

    clean_action = sanitize_ui_text(action_raw)
    st.markdown(
        f"""
        <div class="action-box {action_class}">
            <div style="font-weight: 800; font-size: 1.05rem; margin-bottom: 0.3rem;">📌 Recommended Next Action:</div>
            <div style="font-size: 0.95rem; line-height: 1.45;">{clean_action}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5. Permanent Privacy Notices
    st.markdown(
        """
        <div style="margin-top: 1.2rem; font-size: 0.86rem; color: #15803d; font-weight: 600;">
            🔒 Uploaded files were processed in ephemeral storage and deleted immediately.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.info("👋 Enter a claimed identity name, submit available evidence above, and click **'Analyze Multimodal Consistency'** to start.")
