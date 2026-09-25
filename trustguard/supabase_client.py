"""Supabase Client for TrustGuard Authentication, User Profiles, and Analysis History.

Implements real Supabase Auth and PostgreSQL storage with Row-Level Security (RLS).
Reads credentials from Streamlit secrets (st.secrets) or environment variables.
Does NOT use in-memory fallback. If credentials are missing, clearly reports configuration state.
"""

import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    from supabase import Client, create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = Any  # type: ignore


def get_supabase_config() -> Tuple[Optional[str], Optional[str]]:
    """Retrieves Supabase URL and anon/publishable key from st.secrets or os.environ."""
    url = None
    key = None

    # 1. Try Streamlit secrets
    try:
        if hasattr(st, "secrets"):
            # Check nested table [supabase]
            if "supabase" in st.secrets:
                sub = st.secrets["supabase"]
                url = sub.get("url") or sub.get("SUPABASE_URL")
                key = sub.get("key") or sub.get("anon_key") or sub.get("SUPABASE_KEY") or sub.get("SUPABASE_ANON_KEY")
            # Check top-level secrets
            if not url:
                url = st.secrets.get("SUPABASE_URL")
            if not key:
                key = (
                    st.secrets.get("SUPABASE_KEY")
                    or st.secrets.get("SUPABASE_ANON_KEY")
                    or st.secrets.get("supabase_key")
                )
    except Exception:
        pass

    # 2. Try OS environment variables
    if not url:
        url = os.environ.get("SUPABASE_URL")
    if not key:
        key = (
            os.environ.get("SUPABASE_KEY")
            or os.environ.get("SUPABASE_ANON_KEY")
            or os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        )

    # Clean whitespace
    url = url.strip() if url else None
    key = key.strip() if key else None

    return url, key


def is_supabase_configured() -> bool:
    """Returns True only if Supabase package is installed and valid config exists."""
    if not SUPABASE_AVAILABLE:
        return False
    url, key = get_supabase_config()
    return bool(url and key and url.startswith("http"))


@st.cache_resource(show_spinner=False)
def _init_supabase_client(url: str, key: str) -> Optional[Any]:
    """Internal cached helper to instantiate Supabase client."""
    try:
        return create_client(url, key)
    except Exception as e:
        print(f"[Supabase Init Error] {e}")
        traceback.print_exc()
        return None


def get_client() -> Optional[Any]:
    """Returns the active Supabase client or None if not configured."""
    url, key = get_supabase_config()
    if not url or not key:
        return None
    return _init_supabase_client(url, key)


# ---------------------------------------------------------------------------
# Authentication Operations (Real Supabase Auth)
# ---------------------------------------------------------------------------

def sign_up_user(
    email: str,
    password: str,
    full_name: str = "",
    organization: str = "",
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Registers a new user via Supabase Auth and creates their profile record."""
    client = get_client()
    if not client:
        return False, "Supabase credentials are not configured.", None

    email = email.strip()
    if not email or not password:
        return False, "Email and password are required.", None
    if len(password) < 6:
        return False, "Password must be at least 6 characters.", None

    try:
        options = {
            "data": {
                "full_name": full_name.strip(),
                "organization": organization.strip(),
            }
        }
        res = client.auth.sign_up({"email": email, "password": password, "options": options})

        user = res.user
        if not user:
            return False, "Sign up completed but no user returned.", None

        user_info = {
            "id": user.id,
            "email": user.email,
            "full_name": full_name.strip(),
            "organization": organization.strip(),
        }

        # Attempt to insert into profiles table
        try:
            client.table("profiles").upsert({
                "id": user.id,
                "email": user.email,
                "full_name": full_name.strip(),
                "organization": organization.strip(),
            }).execute()
        except Exception as pe:
            # Non-fatal if table trigger or RLS handles it
            print(f"[Profile upsert notice]: {pe}")

        return True, None, user_info

    except Exception as e:
        err_msg = str(e)
        if "User already registered" in err_msg:
            return False, "An account with this email already exists. Please log in.", None
        return False, f"Sign up error: {err_msg}", None


def sign_in_user(
    email: str,
    password: str,
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Authenticates an existing user via Supabase Auth."""
    client = get_client()
    if not client:
        return False, "Supabase credentials are not configured.", None

    email = email.strip()
    if not email or not password:
        return False, "Please enter both email and password.", None

    try:
        res = client.auth.sign_in_with_password({"email": email, "password": password})
        if not res or not res.user:
            return False, "Invalid email or password.", None

        user = res.user
        session = res.session

        # Fetch full profile info if table exists
        full_name = ""
        organization = ""
        try:
            prof_res = client.table("profiles").select("*").eq("id", user.id).execute()
            if prof_res.data and len(prof_res.data) > 0:
                p = prof_res.data[0]
                full_name = p.get("full_name", "") or ""
                organization = p.get("organization", "") or ""
        except Exception:
            # Fall back to user_metadata
            meta = getattr(user, "user_metadata", {}) or {}
            full_name = meta.get("full_name", "")
            organization = meta.get("organization", "")

        user_info = {
            "id": user.id,
            "email": user.email,
            "full_name": full_name,
            "organization": organization,
            "access_token": session.access_token if session else None,
            "refresh_token": session.refresh_token if session else None,
        }

        return True, None, user_info

    except Exception as e:
        err_msg = str(e)
        if "Invalid login credentials" in err_msg:
            return False, "Incorrect email or password.", None
        return False, f"Authentication failed: {err_msg}", None


def sign_out_user() -> bool:
    """Signs out the active user from Supabase."""
    client = get_client()
    if client:
        try:
            client.auth.sign_out()
        except Exception:
            pass
    return True


# ---------------------------------------------------------------------------
# Database Operations: Analysis Sessions (Isolated by user_id via RLS)
# ---------------------------------------------------------------------------

def save_analysis_session(
    user_id: str,
    title: str,
    claimed_identity: str,
    chat_text: str,
    risk_level: str,
    modality_statuses: Dict[str, str],
    contradiction_map: Dict[str, Any],
    explanation: str,
    recommended_action: str,
    details: Dict[str, Any],
    access_token: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Saves a completed analysis session to the Supabase database.
    
    Returns (success, session_id, error_message).
    """
    client = get_client()
    if not client:
        return False, None, "Supabase credentials are not configured in .streamlit/secrets.toml."

    if not user_id:
        return False, None, "User is not authenticated. Please sign in to save analysis history."

    if access_token:
        try:
            client.postgrest.auth(access_token)
        except Exception:
            pass

    try:
        record = {
            "user_id": user_id,
            "title": (title or "").strip() or f"Analysis: {claimed_identity}",
            "claimed_identity": (claimed_identity or "").strip() or "Unspecified Identity",
            "chat_text": chat_text or "",
            "risk_level": (risk_level or "LOW").upper(),
            "modality_statuses": modality_statuses or {},
            "contradiction_map": contradiction_map or {"agreements": [], "conflicts": [], "uncertain": []},
            "explanation": explanation or "",
            "recommended_action": recommended_action or "",
            "details": details or {},
        }

        res = client.table("analysis_sessions").insert(record).execute()
        if res.data and len(res.data) > 0:
            session_id = res.data[0].get("id")
            return True, session_id, None
        return True, None, None

    except Exception as e:
        err_msg = str(e)
        print(f"[Save Session Exception] {err_msg}")
        err_lower = err_msg.lower()
        if "policy" in err_lower or "rls" in err_lower:
            user_friendly = "Row-level security policy rejected save. Ensure RLS policies in schema.sql are enabled."
        elif "pgrst205" in err_lower or "schema cache" in err_lower or ("could not find" in err_lower and "table" in err_lower):
            user_friendly = "Database table 'analysis_sessions' not found in Supabase. Please run schema.sql in your Supabase SQL Editor."
        elif "jwt" in err_lower or "401" in err_lower:
            user_friendly = "Authentication session expired. Please sign out and sign in again."
        else:
            user_friendly = f"Database save error: {err_msg}"
        return False, None, user_friendly


def list_analysis_sessions(
    user_id: str,
    access_token: Optional[str] = None,
    *args,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Loads all saved analysis sessions for the authenticated user, newest first."""
    client = get_client()
    if not client or not user_id:
        return []

    token = access_token or kwargs.get("token")
    if token:
        try:
            client.postgrest.auth(token)
        except Exception:
            pass

    try:
        res = (
            client.table("analysis_sessions")
            .select("id, title, claimed_identity, chat_text, risk_level, modality_statuses, contradiction_map, explanation, recommended_action, details, created_at, updated_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return res.data or []
    except Exception as e:
        print(f"[List Sessions Error] {e}")
        return []


def get_analysis_session(session_id: str, user_id: str, access_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetches a single analysis session owned by user_id."""
    client = get_client()
    if not client or not user_id or not session_id:
        return None

    if access_token:
        try:
            client.postgrest.auth(access_token)
        except Exception:
            pass

    try:
        res = (
            client.table("analysis_sessions")
            .select("*")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .execute()
        )
        if res.data and len(res.data) > 0:
            return res.data[0]
        return None
    except Exception as e:
        print(f"[Get Session Error] {e}")
        return None


def rename_analysis_session(
    session_id: str,
    user_id: str,
    new_title: str,
    access_token: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Renames an analysis session owned by the authenticated user."""
    client = get_client()
    if not client or not user_id or not session_id:
        return False, "Missing identifiers."

    clean_title = new_title.strip()
    if not clean_title:
        return False, "Title cannot be empty."

    if access_token:
        try:
            client.postgrest.auth(access_token)
        except Exception:
            pass

    try:
        res = (
            client.table("analysis_sessions")
            .update({"title": clean_title})
            .eq("id", session_id)
            .eq("user_id", user_id)
            .execute()
        )
        if res.data:
            return True, None
        return False, "Session not found or permission denied."
    except Exception as e:
        return False, f"Rename failed: {str(e)}"


def delete_analysis_session(
    session_id: str,
    user_id: str,
    access_token: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Permanently deletes an analysis session owned by the authenticated user."""
    client = get_client()
    if not client or not user_id or not session_id:
        return False, "Missing identifiers."

    if access_token:
        try:
            client.postgrest.auth(access_token)
        except Exception:
            pass

    try:
        client.table("analysis_sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()
        return True, None
    except Exception as e:
        return False, f"Delete failed: {str(e)}"

