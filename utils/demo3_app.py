import streamlit as st
import os
import sys
import importlib.util
from pathlib import Path
from datetime import datetime
import textwrap
import pandas as pd
import traceback
import pytz
import json
import tempfile

# Define India Standard Time (IST)
IST = pytz.timezone('Asia/Kolkata')

# --- FIX 1: ENABLE AUTO REFRESH FOR REAL-TIME CHAT ---
try:
    from streamlit_autorefresh import st_autorefresh
    # Refresh every 3 seconds to check for new messages
    st_autorefresh(interval=3000, key="live_refresh")
except ImportError:
    st.error("Please install streamlit-autorefresh: pip install streamlit-autorefresh")
# -----------------------------------------------------

# Persistence: JSON state helper
THIS_DIR = Path(__file__).resolve().parent
DATA_FILE = THIS_DIR / "app_state.json"

def load_app_state():
    default = {
        "tickets": [],
        "chat": {},
        "rag_chat": {},
        "agent_feedback": [],
        "kb_uploads": []
    }
    try:
        if DATA_FILE.exists():
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # ensure keys exist
            for k, v in default.items():
                if k not in data:
                    data[k] = v
            return data
    except Exception as e:
        # if corrupted or unreadable, ignore and return defaults
        print(f"Warning: failed to load {DATA_FILE}: {e}")
    return default

def save_app_state(state: dict):
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="app_state_", suffix=".json", dir=str(THIS_DIR))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, str(DATA_FILE))
    except Exception as e:
        print(f"Error saving app state to {DATA_FILE}: {e}")

def persist_state():
    state = {
        "tickets": st.session_state.get("tickets", []),
        "chat": st.session_state.get("chat", {}),
        "rag_chat": st.session_state.get("rag_chat", {}),
        "agent_feedback": st.session_state.get("agent_feedback", []),
        "kb_uploads": st.session_state.get("kb_uploads", [])
    }
    save_app_state(state)

def sync_from_disk():
    data = load_app_state()
    # Only overwrite the shared keys — keep per-tab keys intact
    st.session_state["tickets"] = data.get("tickets", [])
    st.session_state["chat"] = data.get("chat", {})
    st.session_state["rag_chat"] = data.get("rag_chat", {})
    st.session_state["agent_feedback"] = data.get("agent_feedback", [])
    st.session_state["kb_uploads"] = data.get("kb_uploads", [])

# ---------------------------
# Secrets (prefer st.secrets if available)
# ---------------------------
GROQ_API_KEY = None
GEMINI_API_KEY = None
try:
    GROQ_API_KEY = st.secrets.get("GROQ", {}).get("API_KEY")
    GEMINI_API_KEY = st.secrets.get("GEMINI", {}).get("API_KEY")
except Exception:
    # running outside streamlit or secrets not configured
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# Backend loader

# --- FIX 2: ROBUST IMPORT LOGIC ---
# Ensure we look in the current directory for the backend file
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

BACKEND_DOCS_DIR = str(Path(__file__).resolve().parent.parent / "docs" / "dell-data")
os.makedirs(BACKEND_DOCS_DIR, exist_ok=True)

# Dynamic backend loader
def load_backend_module():
    # Attempt 1: Direct import (works if file is in same folder)
    try:
        import dell_knowledge_query_groq as mod
        return mod, None
    except ImportError:
        pass

    # Attempt 2: Utils package import (works if running from root)
    try:
        import utils.dell_knowledge_query_groq as mod
        return mod, None
    except ImportError as e:
        return None, e

_backend_module, _backend_error = load_backend_module()

# Show error on UI if backend fails to load
if _backend_error:
    st.error(f"CRITICAL BACKEND ERROR: {_backend_error}")
    st.code(traceback.format_exc()) # Print full error trace for debugging

# extract functions from backend module if available
if _backend_module:
    run_rag = getattr(_backend_module, "run_rag", None)
    create_vectorstore = getattr(_backend_module, "create_vectorstore", None)
    ensure_vectordb = getattr(_backend_module, "ensure_vectordb", None)
else:
    run_rag = None
    create_vectorstore = None
    ensure_vectordb = None
# --- FIX END ---


# Helper functions (timestamps & saving files)

def get_current_ist_time():
    """Returns the current datetime string formatted for IST (used for saving to state)."""
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

def get_current_ist_display_time(dt_str_or_obj=None):
    """Return display-friendly IST time. If dt_str_or_obj provided (string in saved format), format that."""
    if dt_str_or_obj is None:
        return datetime.now(IST).strftime("%d-%b %I:%M %p")
    try:
        if isinstance(dt_str_or_obj, datetime):
            dt = dt_str_or_obj.astimezone(IST)
        else:
            dt_naive = datetime.strptime(dt_str_or_obj, "%Y-%m-%d %H:%M:%S")
            dt = IST.localize(dt_naive)
        return dt.strftime("%d-%b %I:%M %p")
    except Exception:
        return "Unknown Time"

def fmt_dt(dt_str_or_obj):
    """Formats datetime string for display (used in message boxes)."""
    return get_current_ist_display_time(dt_str_or_obj)

def query_rag_api(query: str, n_results: int = 4):
    
    if run_rag is None:
        # Show clearer error message
        return {"rag_response": f"RAG backend not configured. Error: {_backend_error}", "doc_context": []}

    if not query or not isinstance(query, str):
        return {"rag_response": "Empty query provided to RAG.", "doc_context": []}

    try:
        with st.spinner("Querying RAG backend..."):
            try:
                result = run_rag(query, n_results=n_results)
            except TypeError:
                result = run_rag(query)
        
        if isinstance(result, dict):
            return {
                "rag_response": result.get("rag_response") or result.get("rag_response_text") or result.get("rag_response", "") or str(result),
                "doc_context": result.get("doc_context", result.get("doc_context", [])) if isinstance(result, dict) else []
            }
        else:
            
            return {"rag_response": str(result), "doc_context": []}
    except Exception as e:
        return {"rag_response": f"RAG runtime error: {e}", "doc_context": []}

def save_uploaded_file(uploaded_file, target_dir=BACKEND_DOCS_DIR):
    """Save an uploaded file to the backend docs folder and return saved path + timestamp."""
    try:
        os.makedirs(target_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Could not create target directory {target_dir}: {e}")

    filename = uploaded_file.name
    filename = "".join(c if c.isalnum() or c in " ._-()" else "_" for c in filename)
    save_path = os.path.join(target_dir, filename)
    try:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except Exception as e:
        raise RuntimeError(f"Failed to write uploaded file to {save_path}: {e}")

    # return saved path and uploaded timestamp (IST)
    uploaded_at = get_current_ist_time()
    return save_path, uploaded_at

def create_ticket_id():
    # ticket id generation based on persisted tickets length (safe across tabs)
    tickets = st.session_state.get("tickets", [])
    return f"TCK-{len(tickets)+1:03}"

def logout():
    st.session_state.logged_in_role = None
    st.session_state.logged_in_email = None
    st.session_state.pending_role = None
    st.session_state.selected_ticket = None
    st.session_state.agent_name = "Default Agent"

# App state initialization 
def init_state():
    # Load persisted state on first initialization, then ensure session keys exist
    if "initialized" not in st.session_state:
        disk = load_app_state()
        st.session_state.tickets = disk.get("tickets", [])
        st.session_state.chat = disk.get("chat", {})
        st.session_state.rag_chat = disk.get("rag_chat", {})
        st.session_state.agent_feedback = disk.get("agent_feedback", [])
        st.session_state.kb_uploads = disk.get("kb_uploads", [])
        # per-tab keys (may be None until login)
        st.session_state.logged_in_role = st.session_state.get("logged_in_role", None)
        st.session_state.logged_in_email = st.session_state.get("logged_in_email", None)
        st.session_state.pending_role = st.session_state.get("pending_role", None)
        st.session_state.agent_name = st.session_state.get("agent_name", "Default Agent")
        st.session_state.selected_ticket = st.session_state.get("selected_ticket", None)
        st.session_state.feedback_submitted = st.session_state.get("feedback_submitted", False)
        st.session_state.initialized = True

init_state()

# UI Styles

st.set_page_config(page_title="Dell Support System", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    :root{
        --accent: #0b6cf2;
        --muted: #6b7280;
        --card: #ffffff;
        --bg: #f6f8fb;
        --panel-radius: 12px;
    }
    body { background: var(--bg); }
    .stApp { color: #0f1724; }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #ffffff, #f8fbff);
        border-right: 1px solid rgba(15,23,36,0.04);
    }
    .big-title { font-size: 34px; font-weight: 700; margin-bottom: 6px; }
    .muted { color: var(--muted); font-size: 14px; }
    .card { background: var(--card); padding: 18px; border-radius: var(--panel-radius); box-shadow: 0 6px 18px rgba(12, 24, 48, 0.06); }
    .ticket-box-assigned { border-left: 4px solid #10b981; padding: 14px; border-radius: 8px; background: #f0fff4; } 
    .ticket-box { border-left: 4px solid var(--accent); padding: 14px; border-radius: 8px; background: #fff; }
    .kbd { background:#f3f6fb; padding:6px 10px; border-radius:6px; font-size:13px; color:#0b6cf2; }
    /* Chat styles */
    .chat-user { padding:8px; border-radius:8px; background:#eef6ff; margin-bottom:6px; border-left: 3px solid #0b6cf2; }
    .chat-agent { padding:8px; border-radius:8px; background:#f5faff; margin-bottom:6px; border-left: 3px solid #10b981; }
    .meta { color: #6b7280; font-size: 12px; }
    .rag-ai-response { padding:8px; border-radius:8px; background:#f7fbff; margin-bottom:6px; border-left: 3px solid #f97316; }
    .rag-context-list { margin-top: 10px; padding: 10px; background: #fef7f2; border-radius: 6px; }
    .rag-context-list ul { margin: 0; padding-left: 20px; font-size: 13px; color: #555; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Chat Renderer (message dicts include sender, name (optional), message, timestamp)

def render_chat_message(m, viewer_role, ticket_id=None):
    """Render one chat message with timestamp and role-aware labels."""
    ts = m.get("timestamp", None)
    ts_html = f"<span class='meta' style='float:right'>{fmt_dt(ts)}</span>"
    if m['sender'] == 'User':
        label = "You" if viewer_role == "Customer" else f"Customer ({ticket_id})"
        html = f"<div class='chat-user'><strong>{label}:</strong> {ts_html}<div style='margin-top:6px'>{m['message']}</div></div>"
    elif m['sender'] == 'Agent':
        label = f"{m.get('name','Agent')}"
        html = f"<div class='chat-agent'><strong>{label} (Agent):</strong> {ts_html}<div style='margin-top:6px'>{m['message']}</div></div>"
    elif m['sender'] == 'RAG':
        # RAG message (AI) - visible to agents only
        ai_html = f"<div class='rag-ai-response'><strong>RAG Response:</strong> {ts_html}<div style='margin-top:6px'>{m['message']}</div></div>"
        docs = m.get("doc_context", [])
        if docs:
            ctx = "".join([f"<li>{textwrap.shorten(d, width=140, placeholder='...')}</li>" for d in docs])
        else:
            ctx = "<li>No relevant doc context found.</li>"
        ctx_html = f"<div class='rag-context-list'><strong>KB Context:</strong><ul>{ctx}</ul></div>"
        html = ai_html + ctx_html
    else:
        html = ""
    st.markdown(html, unsafe_allow_html=True)


# Pages

def user_dashboard():
    # sync shared state from disk at start so other-tab changes appear
    sync_from_disk()

    logged_in_email = st.session_state.logged_in_email or ""
    st.markdown(f"<div class='big-title'>Customer Dashboard</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='muted' style='margin-bottom:15px;'>Logged in as: <strong>{logged_in_email}</strong></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 3])

    # Raise a ticket
    with col1:
        st.markdown("<div class='card'><strong>Raise a New Query</strong></div>", unsafe_allow_html=True)
        with st.form("raise_ticket_form", clear_on_submit=True):
            
            st.text_input("Your email", value=logged_in_email, disabled=True)
            category = st.selectbox("Category", ["Login Issue", "Hardware Issue", "Software Bug", "Warranty", "Other"])
            message = st.text_area("Describe the issue", height=120)
            submitted = st.form_submit_button("Raise Query!")
            if submitted:
                if not message.strip():
                    st.warning("Please provide a short description.")
                else:
                    # ensure freshest state before mutating
                    sync_from_disk()
                    tid = create_ticket_id()
                    now = get_current_ist_time()
                    ticket = {
                        "id": tid,
                        "email": logged_in_email,
                        "message": message.strip(),
                        "category": category,
                        "status": "Open",
                        "agent": "Not Assigned",
                        "created_at": now
                    }
                    st.session_state.tickets.append(ticket)

                    # create chat and add initial user message (with timestamp)
                    if tid not in st.session_state.chat:
                        st.session_state.chat[tid] = []
                    st.session_state.chat[tid].append({
                        "sender": "User",
                        "name": logged_in_email,
                        "message": message.strip(),
                        "timestamp": now
                    })

                    # persist after changes
                    persist_state()

                    # auto-open chat immediately (auto-show + Open Chat button effect)
                    st.session_state.selected_ticket = tid
                    st.success(f"Query **{tid}** Raised. Chat opened below.")
                    st.rerun()

    # Your tickets & chat
    with col2:
        st.markdown("<div class='card'><strong>Your Active Queries</strong></div>", unsafe_allow_html=True)

        # only show non-resolved tickets for this logged-in customer
        tickets = st.session_state.tickets
        filtered = [t for t in tickets if (t["email"].strip().lower() == logged_in_email.strip().lower() and t["status"] != "Resolved")]

        if not filtered:
            st.info("No active queries found.")
        else:
            for t in filtered:
                is_sel = st.session_state.selected_ticket == t['id']
                box_style = "border-left: 4px solid #f97316 !important; background: #fffbeb;" if is_sel else ""
                st.markdown(f"<div class='ticket-box' style='margin-top:10px; {box_style}'>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top:8px'><strong>{t['id']}</strong>: {t['message']}</div>", unsafe_allow_html=True)
                created_display = fmt_dt(t['created_at'])
                st.markdown(f"<div style='margin-top:8px' class='muted'>Status: {t['status']}  ·  Agent: {t['agent']}  ·  Created: {created_display}</div>", unsafe_allow_html=True)

                # Open Chat button (chat auto-opens after creation)
                c1, c2 = st.columns([1,3])
                with c1:
                    if st.button("Open Chat", key=f"user_open_{t['id']}"):
                        st.session_state.selected_ticket = t['id']
                        st.rerun()
                with c2:
                    # Optionally show small "View Details" or timestamp
                    st.write("")
                st.markdown("</div>", unsafe_allow_html=True)

        # Chat view for current selected ticket
        if st.session_state.selected_ticket:
            sel = st.session_state.selected_ticket
            current_ticket = next((x for x in filtered if x['id'] == sel), None)
            # if the selected ticket doesn't belong to user or is resolved, clear selection
            if not current_ticket:
                st.session_state.selected_ticket = None
            else:
                st.markdown(f"<div class='card' style='margin-top:20px;'><strong>Chat — {sel}</strong><div class='muted'>Two-way messages</div></div>", unsafe_allow_html=True)
                if sel not in st.session_state.chat:
                    st.session_state.chat[sel] = []

                chat_container = st.container()
                with chat_container:
                    for m in st.session_state.chat[sel]:
                        render_chat_message(m, "Customer", sel)

                # Compose & send message
                ticket_status = current_ticket['status']
                if ticket_status != "Resolved":
                    msg_key = f"user_msg_in_{sel}"
                    user_msg = st.text_input("Type a message to the agent", key=msg_key)
                    if st.button("Send", key=f"send_user_{sel}"):
                        if user_msg.strip():
                            # ensure freshest before mutating
                            sync_from_disk()
                            now_ts = get_current_ist_time()
                            # re-create chat list in case replaced
                            if sel not in st.session_state.chat:
                                st.session_state.chat[sel] = []
                            st.session_state.chat[sel].append({
                                "sender": "User",
                                "name": logged_in_email,
                                "message": user_msg.strip(),
                                "timestamp": now_ts
                            })
                            persist_state()
                            st.rerun()
                        else:
                            st.warning("Message is empty.")
                else:
                    st.info("This ticket is **Resolved**. Chat is closed.")

# -----------------------
# Agent Console
# -----------------------
def agent_console():
    # Sync latest shared state from disk
    sync_from_disk()

    st.markdown(f"<div class='big-title'>Agent Console</div>", unsafe_allow_html=True)
    agent_name = st.session_state.agent_name

    left_col, right_col = st.columns([2, 3])

    # show success message once after actions
    if st.session_state.feedback_submitted:
        st.success("Action completed.")
        st.session_state.feedback_submitted = False

    # Left: ticket list (hide email)
    with left_col:
        st.markdown(f"**Agent:** `{agent_name}`", unsafe_allow_html=True)
        st.markdown("<div class='card' style='margin-top:8px;'><strong>Ticket Queue</strong><div class='muted'>Open and In Progress queries</div></div>", unsafe_allow_html=True)
        queue = [t for t in st.session_state.tickets if t["status"] != "Resolved"]

        if not queue:
            st.info("No active tickets in queue.")
        else:
            for t in queue:
                is_assigned = t["agent"] == agent_name
                is_sel = st.session_state.selected_ticket == t['id']
                box_class = "ticket-box-assigned" if is_assigned else "ticket-box"
                box_style = f"margin-top:10px; {'border: 2px solid #f97316 !important;' if is_sel else ''}"
                st.markdown(f"<div class='{box_class}' style='{box_style}'>", unsafe_allow_html=True)

                # IMPORTANT: show only Ticket ID, Issue (shortened), Status
                st.markdown(f"<div><strong>{t['id']}</strong>: {textwrap.shorten(t['message'], width=60)}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top:6px' class='muted'>Status: {t['status']}</div>", unsafe_allow_html=True)

                col_a, col_b, col_c = st.columns([1.2, 1, 1])
                with col_a:
                    if t["agent"] == "Not Assigned":
                        if st.button("Assign to me", key=f"assign_{t['id']}"):
                            # ensure freshest state
                            sync_from_disk()
                            # find ticket again and modify
                            for tt in st.session_state.tickets:
                                if tt["id"] == t["id"]:
                                    tt["agent"] = agent_name
                                    tt["status"] = "In Progress"
                                    break
                            st.session_state.selected_ticket = t['id']
                            st.session_state.feedback_submitted = True
                            persist_state()
                            st.rerun()
                    elif is_assigned:
                        st.markdown(f"<span class='kbd' style='color:#10b981; background:#e0ffe5; font-size:11px'>Assigned to You</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span class='kbd' style='color:#6b7280; background:#f3f6fb; font-size:11px'>Assigned</span>", unsafe_allow_html=True)

                with col_b:
                    if st.button("Open", key=f"open_{t['id']}"):
                        st.session_state.selected_ticket = t['id']
                        st.rerun()

                with col_c:
                    if st.button("Mark Resolved", key=f"resolve_{t['id']}"):
                        # ensure freshest
                        sync_from_disk()
                        for tt in st.session_state.tickets:
                            if tt["id"] == t["id"]:
                                tt["status"] = "Resolved"
                                break
                        # add minimal feedback record
                        st.session_state.agent_feedback.append({
                            "ticket_id": t['id'],
                            "agent": agent_name,
                            "usefulness": "N/A (Resolved Quickly)",
                            "missing_kb": "",
                            "status": "Resolved",
                            "timestamp": get_current_ist_time()
                        })
                        # persist and update UI
                        if st.session_state.selected_ticket == t['id']:
                            st.session_state.selected_ticket = None
                        st.session_state.feedback_submitted = True
                        persist_state()
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    # Right: ticket handling - do NOT display user's email here
    with right_col:
        st.markdown("<div class='card'><strong>Handle Ticket</strong><div class='muted'>Chat, RAG, and feedback</div></div>", unsafe_allow_html=True)
        sel = st.session_state.selected_ticket
        ticket = next((x for x in st.session_state.tickets if x["id"] == sel), None)

        # Only proceed if ticket exists and not resolved
        if not sel or not ticket or ticket["status"] == "Resolved":
            st.info("Select an active ticket from the left queue (Assign/Open) to view details.")
            return

        # Don't show the email to the agent; show id, status and message only
        st.markdown(f"<div class='card' style='margin-bottom:10px'><strong>Customer Query: {ticket['id']}</strong><div style='margin-top:8px'>Status: **{ticket['status']}**</div><hr style='margin-top:10px; margin-bottom:10px;'>{ticket['message']}</div>", unsafe_allow_html=True)

        c1, c2 = st.columns([1.6, 1])

        with c1:
            st.markdown("<div class='card'><strong>Chat with Customer</strong><div class='muted'>Two-way messages</div></div>", unsafe_allow_html=True)
            if sel not in st.session_state.chat:
                st.session_state.chat[sel] = []

            chat_container = st.container()
            with chat_container:
                for m in st.session_state.chat[sel]:
                    render_chat_message(m, "Agent", sel)

            agent_msg = st.text_input("Type message to user", key=f"agent_msg_{sel}")
            if st.button("Send to user", key=f"send_agent_{sel}"):
                if agent_msg.strip():
                    # ensure freshest before mutating
                    sync_from_disk()
                    if sel not in st.session_state.chat:
                        st.session_state.chat[sel] = []
                    now_ts = get_current_ist_time()
                    st.session_state.chat[sel].append({
                        "sender": "Agent",
                        "name": agent_name,
                        "message": agent_msg.strip(),
                        "timestamp": now_ts
                    })
                    persist_state()
                    st.rerun()
                else:
                    st.warning("Message empty!")

        # RAG assistant
        with c2:
            st.markdown("<div class='card'><strong>RAG Assistant</strong><div class='muted'>Ask the retrieval assistant for suggestions</div></div>", unsafe_allow_html=True)
            if sel not in st.session_state.rag_chat:
                st.session_state.rag_chat[sel] = []

            with st.container():
                for m in st.session_state.rag_chat[sel]:
                    render_chat_message(m, "Agent", sel)

            rag_q = st.text_input("Ask RAG (e.g., recommended fix?)", key=f"rag_q_{sel}")
            if st.button("Query RAG", key=f"query_rag_{sel}"):
                q = rag_q.strip()
                if q:
                    # ensure freshest before mutating
                    sync_from_disk()
                    # add agent query as a record
                    if sel not in st.session_state.rag_chat:
                        st.session_state.rag_chat[sel] = []
                    st.session_state.rag_chat[sel].append({
                        "sender": "User",
                        "name": agent_name,
                        "message": f"Agent Query: {q}",
                        "timestamp": get_current_ist_time()
                    })
                    persist_state()

                    # call backend
                    rag_result = query_rag_api(q)
                    if rag_result:
                        ai_response = rag_result.get("rag_response", "")
                        doc_context = rag_result.get("doc_context", [])
                        st.session_state.rag_chat[sel].append({
                            "sender": "RAG",
                            "name": "RAG Assistant",
                            "message": ai_response,
                            "doc_context": doc_context,
                            "timestamp": get_current_ist_time()
                        })
                        persist_state()
                        st.rerun()
                else:
                    st.warning("Empty query!")

        st.markdown("---")
        # Feedback & status update
        st.markdown("<div class='card'><strong>Provide Feedback & Update Ticket</strong></div>", unsafe_allow_html=True)
        status_options = ["In Progress", "Waiting for User"]
        try:
            idx = status_options.index(ticket["status"])
        except ValueError:
            idx = 0
        colf1, colf2 = st.columns([2, 1])
        with colf1:
            usefulness = st.selectbox("RAG Usefulness", ["Very Useful", "Somewhat Useful", "Not Useful"], key=f"useful_{sel}")
            missing_suggest = st.text_area("Suggest missing KB article (optional)", key=f"missing_{sel}", height=80)
        with colf2:
            new_status = st.selectbox("Update Ticket Status", status_options + ["Resolved"], index=idx, key=f"status_{sel}")
            if st.button("Submit Feedback & Update", key=f"submit_feedback_{sel}"):
                # ensure freshest state
                sync_from_disk()
                # update tickets list
                t_index = next((i for i, x in enumerate(st.session_state.tickets) if x["id"] == sel), -1)
                if t_index != -1:
                    st.session_state.tickets[t_index]["status"] = new_status

                st.session_state.agent_feedback.append({
                    "ticket_id": sel,
                    "agent": agent_name,
                    "usefulness": usefulness,
                    "missing_kb": missing_suggest,
                    "status": new_status,
                    "timestamp": get_current_ist_time()
                })

                # persist changes
                persist_state()

                # if resolved, clear selection so dashboards stop showing it
                if new_status == "Resolved":
                    st.session_state.selected_ticket = None
                st.session_state.feedback_submitted = True
                st.rerun()

# -----------------------
# Content Manager Hub
# -----------------------
def content_manager_hub():
    # Sync latest shared state from disk
    sync_from_disk()

    st.markdown(f"<div class='big-title'>Content Manager Hub</div>", unsafe_allow_html=True)

    # feedback_submitted messages
    if st.session_state.feedback_submitted:
        st.success("Action completed.")
        st.session_state.feedback_submitted = False

    st.markdown("---")
    tabs = st.tabs(["Agent Feedback Overview", "Upload Documents & Rebuild KB", "Knowledge Base Viewer"])

    with tabs[0]:
        st.subheader("Agent Feedback Overview (All Time)")
        feedback = st.session_state.agent_feedback
        if not feedback:
            st.info("No feedback submitted yet.")
        else:
            try:
                df = pd.DataFrame(feedback)
                total_feedback = len(df)
                # counts by usefulness
                dist = df['usefulness'].fillna("Unknown").value_counts().rename_axis('usefulness').reset_index(name='count')
                # resolved count
                resolved_count = df[df['status'] == 'Resolved'].shape[0]

                # Metrics row
                col1, col2, col3 = st.columns(3)
                with col1:
                    very_useful = int(dist[dist['usefulness'] == 'Very Useful']['count'].sum() if 'Very Useful' in dist['usefulness'].values else 0)
                    st.metric("RAG Very Useful", value=very_useful, delta=f"{(very_useful/total_feedback*100):.1f}% of total" if total_feedback else "0%")
                with col2:
                    not_useful = int(dist[dist['usefulness'] == 'Not Useful']['count'].sum() if 'Not Useful' in dist['usefulness'].values else 0)
                    st.metric("RAG Not Useful", value=not_useful, delta=f"{(not_useful/total_feedback*100):.1f}% of total" if total_feedback else "0%")
                with col3:
                    st.metric("Total Feedback Records", value=total_feedback, delta=f"Resolved: {resolved_count}")

                st.markdown("#### Feedback distribution (by usefulness)")
                # Fix: use a DataFrame for the bar chart
                if not dist.empty:
                    dist = dist.set_index('usefulness')
                    st.bar_chart(dist['count'])
                else:
                    st.info("No distribution data to plot.")

                st.markdown("---")
                st.markdown("#### Raw Feedback Data (latest first)")
                df_sorted = df.sort_values(by="timestamp", ascending=False)
                st.dataframe(df_sorted)
                st.markdown("---")
                st.markdown("#### Suggestions for Missing KB Articles")
                if 'missing_kb' in df.columns:
                    suggestions = df[df['missing_kb'].astype(str).str.strip() != ''][['ticket_id','missing_kb','agent','timestamp']].rename(columns={'missing_kb':'Suggestion'})
                    if not suggestions.empty:
                        st.dataframe(suggestions)
                    else:
                        st.info("No suggestions submitted yet.")
                else:
                    st.info("No suggestions submitted yet.")
            except Exception as e:
                st.error(f"Error processing feedback data: {e}")
                st.exception(traceback.format_exc())

    with tabs[1]:
        st.subheader("Upload New KB Document")
        upload = st.file_uploader("Upload .docx (multiple allowed)", type=["docx"], accept_multiple_files=True)
        if upload:
            saved_files = []
            for f in upload:
                try:
                    # ensure freshest before mutating
                    sync_from_disk()
                    save_path, uploaded_at = save_uploaded_file(f, target_dir=BACKEND_DOCS_DIR)
                    saved_files.append(os.path.basename(save_path))
                    # store metadata for display
                    st.session_state.kb_uploads.append({"file": os.path.basename(save_path), "uploaded_at": uploaded_at})
                    # persist
                    persist_state()
                    st.success(f"Saved: {os.path.basename(save_path)}")
                except Exception as e:
                    st.error(f"Failed to save {f.name}: {e}")

            if create_vectorstore is not None:
                if st.button("Rebuild KB now"):
                    try:
                        with st.spinner("Rebuilding vectorstore..."):
                            create_vectorstore()
                        st.session_state.feedback_submitted = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Rebuild failed: {e}")
            else:
                st.warning("Backend vectorstore creation function not available.")

    with tabs[2]:
        st.subheader("Knowledge Base Snapshot")
        kb_dir = BACKEND_DOCS_DIR
        if os.path.exists(kb_dir):
            files = sorted(os.listdir(kb_dir))
            if not files:
                st.info("No documents found in the knowledge base directory.")
            else:
                for file in files:
                    file_path = os.path.join(kb_dir, file)
                    if os.path.isfile(file_path):
                        # prefer showing stored upload times if available
                        entry = next((x for x in st.session_state.kb_uploads if x['file'] == file), None)
                        if entry:
                            uploaded_at_display = fmt_dt(entry['uploaded_at'])
                        else:
                            last_updated_ts = datetime.fromtimestamp(os.path.getmtime(file_path)).astimezone(IST)
                            uploaded_at_display = last_updated_ts.strftime("%d-%b %I:%M %p")
                        st.markdown(f"**{file}** <span class='muted' style='font-size:12px;'> (Uploaded at: {uploaded_at_display})</span>", unsafe_allow_html=True)


# Credentials & Sidebar router

CREDENTIALS = {
    "agent1": {"password": "pass1", "role": "Agent"},
    "agent2": {"password": "pass2", "role": "Agent"},
    "cm": {"password": "cm_pass", "role": "Content Manager"},
}

# Add ram@gmail.com as requested
USER_EMAILS = ["aditya@gmail.com", "ram@gmail.com"]

with st.sidebar:
    st.markdown("<div style='padding:14px 6px'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:800; font-size:18px; margin-bottom:8px'>Dell Support System</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.logged_in_role:
        st.markdown(f"**Logged in as:** `{st.session_state.logged_in_role}`")
        if st.session_state.logged_in_role == "Agent":
            st.markdown(f"**Agent Name:** `{st.session_state.agent_name}`")
        elif st.session_state.logged_in_role == "Customer":
            st.markdown(f"**Email:** `{st.session_state.logged_in_email}`")

        st.markdown("---")
        st.button("Logout", on_click=logout)
    else:
        # Customer Login with Dropdown (ram@gmail.com added)
        st.markdown("##### Customer Login")
        user_selected_email = st.selectbox("Select Your Email", USER_EMAILS, key="user_email_selector")
        if st.button("Login as Customer"):
            st.session_state.logged_in_role = "Customer"
            st.session_state.logged_in_email = user_selected_email
            st.session_state.pending_role = None
            st.rerun()

        st.markdown("---")
        # Staff Login
        st.markdown("##### Staff Login")
        internal_role = st.radio("Select Internal Role to Login", ["Agent", "Content Manager"], index=0, key="internal_role_selector_sidebar")
        if st.button(f"Proceed to {internal_role} Login"):
            st.session_state.pending_role = internal_role
            st.rerun()

# Internal login page
def internal_login_page():
    role = st.session_state.pending_role
    if not role:
        st.error("Error: No internal role selected. Please select a role in the sidebar.")
        return
    st.markdown(f"<div class='big-title' style='margin-bottom: 20px;'>{role} Login</div>", unsafe_allow_html=True)
    st.markdown("<div class='card' style='max-width: 400px; margin: 0 auto; padding: 30px;'>", unsafe_allow_html=True)

    with st.form("internal_login_form_page", clear_on_submit=False):
        username = st.text_input("Username", value="", key="login_username")
        password = st.text_input("Password", type="password", value="", key="login_password")
        login_submitted = st.form_submit_button(f"Login as {role}")
        if login_submitted:
            if (username in CREDENTIALS and CREDENTIALS[username]["password"] == password and CREDENTIALS[username]["role"] == role):
                st.session_state.logged_in_role = role
                st.session_state.pending_role = None
                if role == "Agent":
                    st.session_state.agent_name = username
                st.success(f"Successfully logged in as {role}!")
                st.rerun()
            else:
                st.error("Invalid credentials or role mismatch.")
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("Cancel & Go Back"):
        st.session_state.pending_role = None
        st.rerun()

# Router
current_role = st.session_state.logged_in_role
pending_role = st.session_state.pending_role

if current_role == "Customer":
    user_dashboard()
elif current_role == "Agent":
    agent_console()
elif current_role == "Content Manager":
    content_manager_hub()
elif pending_role in ["Agent", "Content Manager"]:
    internal_login_page()
else:
    st.markdown("<div style='text-align:center; padding: 100px;'>", unsafe_allow_html=True)
    st.markdown(f"<div class='big-title' style='color: var(--accent);'>Welcome to the Dell Support Portal</div>", unsafe_allow_html=True)
    st.markdown("<h3 class='muted'>Please select your access point from the sidebar to begin.</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
