import streamlit as st
import os
import sys
import importlib.util
from pathlib import Path
from datetime import datetime
import textwrap
import pandas as pd


#path
BACKEND_DOCS_DIR = "/Users/krishna/Downloads/DELL_KB_PROJECT-23/docs/dell-data"



# Backend loader 

_BACKEND_MODULE_NAME = "dell_kb_backend_dynamic"

def load_backend_module():
    try:
        import utils.dell_knowledge_query_groq as mod  
        return mod, None
    except Exception as e_normal:
        try:
            THIS_FILE = Path(__file__).resolve()
            backend_path = THIS_FILE.parent / "dell_knowledge_query_groq.py"
            if not backend_path.exists():
                raise FileNotFoundError(f"Backend file not found at {backend_path}")

            spec = importlib.util.spec_from_file_location(_BACKEND_MODULE_NAME, str(backend_path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create spec for backend at {backend_path}")

            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  
            return mod, None
        except Exception as e_path:
            
            return None, (e_normal, e_path)


_backend_module, _backend_error = load_backend_module()


if _backend_module:
    run_rag = getattr(_backend_module, "run_rag", None)
else:
    run_rag = None

#backend call
def query_rag_api(query: str):
    if run_rag is None:

        if _backend_error is None:
            st.error("RAG backend not found or 'run_rag' missing in backend module.")
        else:
            
            if isinstance(_backend_error, tuple):
                normal_err, path_err = _backend_error
                st.error("RAG backend import failed.\nFirst import attempt (normal package import) raised:\n\n"
                         f"{normal_err}\n\nSecond attempt (dynamic path import) raised:\n\n{path_err}")
            else:
                st.error(f"RAG backend import failed: {_backend_error}")
        return None

    if not query or not isinstance(query, str):
        st.warning("Empty query provided to RAG.")
        return None

    
    try:
        with st.spinner("Querying RAG backend... (this may take a few seconds while embeddings/LLM runs)"):
            result = run_rag(query)
        return result
    except Exception as e:
        st.error(f"RAG runtime error: {e}")
        return None


CREDENTIALS = {
    "agent1": {"password": "pass1", "role": "Agent"},
    "agent2": {"password": "pass2", "role": "Agent"},
    "agent3": {"password": "pass3", "role": "Agent"},
    "agent4": {"password": "pass4", "role": "Agent"},
    "agent5": {"password": "pass5", "role": "Agent"},
    "cm": {"password": "cm_pass", "role": "Content Manager"},
}

st.set_page_config(page_title="Dell Support System", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    :root{
        --accent: #0b6cf2;
        --accent-2: #0f76ff;
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
    .big-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .muted { color: var(--muted); font-size: 14px; }
    .card {
        background: var(--card);
        padding: 18px;
        border-radius: var(--panel-radius);
        box-shadow: 0 6px 18px rgba(12, 24, 48, 0.06);
    }
    .ticket-box-assigned { border-left: 4px solid #10b981; padding: 14px; border-radius: 8px; background: #f0fff4; } 
    .ticket-box { border-left: 4px solid var(--accent); padding: 14px; border-radius: 8px; background: #fff; }
    .kbd { background:#f3f6fb; padding:6px 10px; border-radius:6px; font-size:13px; color:#0b6cf2; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state():
    if "tickets" not in st.session_state:
        st.session_state.tickets = [
            {"id": "TCK-001", "email": "test@user.com", "message": "My laptop won't turn on after the update.", "category": "Hardware Issue", "priority": "High", "status": "Open", "agent": "Not Assigned", "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        ]
    if "chat" not in st.session_state:
        st.session_state.chat = {}
    if "rag_chat" not in st.session_state:
        st.session_state.rag_chat = {}
    if "agent_feedback" not in st.session_state:
        st.session_state.agent_feedback = []
    if "logged_in_role" not in st.session_state:
        st.session_state.logged_in_role = None
    if "pending_role" not in st.session_state:
        st.session_state.pending_role = None
    if "agent_name" not in st.session_state:
        st.session_state.agent_name = "Default Agent"
    if "selected_ticket" not in st.session_state:
        st.session_state.selected_ticket = None

init_state()

def create_ticket_id():
    return f"TCK-{len(st.session_state.tickets)+1:03}"

def save_uploaded_file(uploaded_file, target_dir=BACKEND_DOCS_DIR):
    """
    Save an uploaded file to the backend docs folder.
    Returns the saved path (string) on success, or raises an exception.
    """
    # Create target dir if it doesn't exist
    try:
        os.makedirs(target_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Could not create target directory {target_dir}: {e}")

    save_path = os.path.join(target_dir, uploaded_file.name)
    try:
        # uploaded_file is a UploadedFile object from Streamlit
        # use getbuffer() which works reliably for binary writes
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except Exception as e:
        raise RuntimeError(f"Failed to write uploaded file to {save_path}: {e}")

    return save_path

def logout():
    st.session_state.logged_in_role = None
    st.session_state.pending_role = None
    st.session_state.selected_ticket = None
    st.session_state.agent_name = "Default Agent"

# Sidebar
with st.sidebar:
    st.markdown("<div style='padding:14px 6px'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:800; font-size:18px; margin-bottom:8px'>Dell Support System</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.logged_in_role:
        st.markdown(f"**Logged in as:** `{st.session_state.logged_in_role}`")
        if st.session_state.logged_in_role == "Agent":
             st.markdown(f"**Agent Name:** `{st.session_state.agent_name}`")
        st.markdown("---")
        st.button("Logout", on_click=logout)
    else:
        st.markdown("**Select your access point:**")

        if st.button("User Login "):
            st.session_state.logged_in_role = "User"
            st.session_state.pending_role = None

        st.markdown("---")

        st.markdown("##### Staff Login")
        internal_role = st.radio(
            "Select Internal Role to Login",
            ["Agent", "Content Manager"],
            index=0,
            key="internal_role_selector_sidebar"
        )
        if st.button(f"Proceed to {internal_role} Login"):
            st.session_state.pending_role = internal_role

    st.markdown("---")

# Internal login
def internal_login_page():
    role = st.session_state.pending_role
    if not role:
        st.error("Error: No internal role selected. Please select a role in the sidebar.")
        return

    st.markdown(f"<div class='big-title' style='margin-bottom: 20px;'>{role} Login</div>", unsafe_allow_html=True)
    st.markdown("<div class='card' style='max-width: 400px; margin: 0 auto; padding: 30px;'>", unsafe_allow_html=True)
    st.markdown(f"**Accessing the internal system as a {role}.**")

    with st.form("internal_login_form_page", clear_on_submit=False):
        username = st.text_input("Username", value="")
        password = st.text_input("Password", type="password", value="")

        login_submitted = st.form_submit_button(f"Login as {role}")

        if login_submitted:
            if (username in CREDENTIALS and
                CREDENTIALS[username]["password"] == password and
                CREDENTIALS[username]["role"] == role):

                st.session_state.logged_in_role = role
                st.session_state.pending_role = None

                if role == "Agent":
                    st.session_state.agent_name = username
            else:
                st.error("Invalid credentials or role mismatch. Please check your username, password, and the selected role.")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Cancel & Go Back"):
        st.session_state.pending_role = None

# --- Page functions

def user_dashboard():
    st.markdown(f"<div class='big-title'>User Dashboard</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("<div class='card'><strong>Raise a Ticket</strong><div class='muted'>Report an issue</div></div>", unsafe_allow_html=True)
        with st.form("raise_ticket_form", clear_on_submit=True):
            email = st.text_input("Your email", value="test@user.com")
            category = st.selectbox("Category", ["Login Issue", "Hardware Issue", "Software Bug", "Warranty", "Other"])
            message = st.text_area("Describe the issue", height=120)
            submitted = st.form_submit_button("Raise Query!")
            if submitted:
                if not email or not message.strip():
                    st.warning("Please provide your email and a short description.")
                else:
                    tid = create_ticket_id()
                    st.session_state.tickets.append({
                        "id": tid,
                        "email": email,
                        "message": message.strip(),
                        "category": category,

                        "status": "Open",
                        "agent": "Not Assigned",
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.success(f"Query {tid} Raised. Track it below.")

    with col2:
        st.markdown("<div class='card'><strong>Your Query</strong><div class='muted'>Track status & open chat</div></div>", unsafe_allow_html=True)
        user_email = st.text_input("Filter by your email", value="test@user.com", key="user_email_filter")
        tickets = st.session_state.tickets
        filtered = [t for t in tickets if (user_email.strip() == "" or t["email"].strip().lower() == user_email.strip().lower())]

        if not filtered:
            st.info("No tickets found for this email.")
        else:
            for t in filtered:
                is_selected = st.session_state.selected_ticket == t['id']
                box_style = "border-left: 4px solid #f97316 !important; background: #fffbeb;" if is_selected else ""
                box_class = "ticket-box"

                st.markdown(f"<div class='{box_class}' style='margin-top:10px; {box_style}'>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top:8px'>{t['message']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top:8px' class='muted'>Status: {t['status']}  ·  Agent: {t['agent']}</div>", unsafe_allow_html=True)

                btn_col1, _ = st.columns([1,2])
                with btn_col1:
                    if st.button("Open Chat", key=f"user_openchat_{t['id']}"):
                        st.session_state.selected_ticket = t['id']
                st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.selected_ticket:
            sel = st.session_state.selected_ticket
            if next((t for t in filtered if t['id'] == sel), None):
                st.markdown(f"<div class='card' style='margin-top:20px;'><strong>Chat — {sel}</strong><div class='muted'>Two-way messages </div></div>", unsafe_allow_html=True)

                if sel not in st.session_state.chat:
                    st.session_state.chat[sel] = []

                chat_container = st.container(height=300)
                with chat_container:
                    for m in st.session_state.chat[sel]:
                        st.markdown(m, unsafe_allow_html=True)

                ticket_status = next((t['status'] for t in st.session_state.tickets if t['id'] == sel), "Resolved")
                is_resolved = ticket_status == "Resolved"

                if not is_resolved:
                    user_msg = st.text_input("Type a message to the agent", key=f"usermsg_{sel}")
                    if st.button("Send", key=f"send_user_{sel}"):
                        if user_msg.strip():
                            st.session_state.chat[sel].append(f"<div style='padding:8px; border-radius:8px; background:#eef6ff; margin-bottom:6px'><strong>You:</strong> {user_msg}</div>")
                            st.session_state.chat[sel].append(f"<div style='padding:8px; border-radius:8px; background:#f7f7fb; margin-bottom:6px'><strong>Support Team:</strong> Thanks — we'll check and update you shortly.</div>")
                        else:
                            st.warning("Message is empty.")
                else:
                    st.info("This ticket is **Resolved**. Chat is closed.")
            else:
                st.session_state.selected_ticket = None

def agent_console():
    st.markdown(f"<div class='big-title'>Agent Console</div>", unsafe_allow_html=True)
    agent_name = st.session_state.agent_name
    left_col, right_col = st.columns([2, 3])

    with left_col:
        st.markdown(f"**Agent:** `{agent_name}`", unsafe_allow_html=True)
        st.markdown("<div class='card' style='margin-top:8px;'><strong>Ticket Queue</strong><div class='muted'>Open, assigned, or pending resolution</div></div>", unsafe_allow_html=True)

        queue = [t for t in st.session_state.tickets if t["status"] in ("Open", "In Progress", "Waiting for User")]

        if not queue:
            st.info("No active tickets in queue.")
        else:
            for t in queue:
                is_assigned_to_me = t["agent"] == agent_name
                is_selected = st.session_state.selected_ticket == t['id']
                box_class = "ticket-box-assigned" if is_assigned_to_me else "ticket-box"
                box_style = f"margin-top:10px; {'border: 2px solid #f97316 !important;' if is_selected else ''}"

                st.markdown(f"<div class='{box_class}' style='{box_style}'>", unsafe_allow_html=True)
            
                st.markdown(f"<div style='margin-top:8px' class='muted'>From: {t['email']} · Agent: {t['agent']}</div>", unsafe_allow_html=True)

                col_a, col_b, col_c = st.columns([1,1,1])
                with col_a:
                    if t["agent"] == "Not Assigned":
                        if st.button("Assign to me", key=f"assign_{t['id']}"):
                            t["agent"] = agent_name
                            t["status"] = "In Progress"
                            st.session_state.selected_ticket = t['id']
                            st.success(f"{t['id']} assigned to you")
                    elif is_assigned_to_me:
                        st.markdown(f"<span class='kbd' style='color:#10b981; background:#e0ffe5'>Assigned to You</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span class='kbd' style='color:#6b7280; background:#f3f6fb'>Assigned to {t['agent']}</span>", unsafe_allow_html=True)

                with col_b:
                    if st.button("Open", key=f"open_{t['id']}"):
                        st.session_state.selected_ticket = t['id']

                with col_c:
                    if st.button("Mark Resolved", key=f"resolve_{t['id']}"):
                        t["status"] = "Resolved"
                        if st.session_state.selected_ticket == t['id']:
                            st.session_state.selected_ticket = None
                        st.success(f"{t['id']} marked resolved")

                st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='card'><strong>Handle Ticket</strong><div class='muted'>Chat with user, consult RAG, and give feedback</div></div>", unsafe_allow_html=True)

        sel = st.session_state.selected_ticket
        ticket = next((x for x in st.session_state.tickets if x["id"] == sel), None)

        if not sel or not ticket:
            st.info("Select a ticket from the left queue (Assign or Open) to view details.")
            return


        st.markdown(f"<div class='card' style='margin-bottom:10px'><strong>User Query</strong><div style='margin-top:8px'>{ticket['message']}</div></div>", unsafe_allow_html=True)

        c1, c2 = st.columns([1.6, 1])
        with c1:
            st.markdown("<div class='card'><strong>Chat with User</strong><div class='muted'>Two-way messages</div></div>", unsafe_allow_html=True)
            if sel not in st.session_state.chat:
                st.session_state.chat[sel] = []

            chat_container = st.container(height=300)
            with chat_container:
                for m in st.session_state.chat[sel]:
                    st.markdown(m, unsafe_allow_html=True)

            is_resolved = ticket["status"] == "Resolved"

            if not is_resolved:
                agent_msg = st.text_input("Type message to user", key=f"agent_msg_{sel}")
                if st.button("Send to user", key=f"send_agent_{sel}"):
                    if agent_msg.strip():
                        st.session_state.chat[sel].append(f"<div style='padding:8px; border-radius:8px; background:#eef9f5; margin-bottom:6px'><strong>{st.session_state.agent_name}:</strong> {agent_msg}</div>")
                        st.success("Message sent to user ")
                    else:
                        st.warning("Message empty!")
            else:
                st.info("Ticket is **Resolved**. Agent chat is closed.")

        with c2:
            st.markdown("<div class='card'><strong>RAG Assistant</strong><div class='muted'>Ask the retrieval assistant for suggestions</div></div>", unsafe_allow_html=True)
            if sel not in st.session_state.rag_chat:
                st.session_state.rag_chat[sel] = []

            rag_chat_container = st.container(height=300)
            with rag_chat_container:
                for m in st.session_state.rag_chat[sel]:
                    st.markdown(m, unsafe_allow_html=True)

            # RAG integration (uses query_rag_api wrapper)
            rag_q = st.text_input("Ask RAG (e.g., recommended fix?)", key=f"rag_q_{sel}")
            if st.button("Query RAG", key=f"query_rag_{sel}"):
                query_text = rag_q.strip()
                if query_text:
                    st.session_state.rag_chat[sel].append(f"<div style='padding:8px; border-radius:8px; background:#eef6ff; margin-bottom:6px'><strong>Agent → RAG:</strong> {query_text}</div>")

                    rag_result = query_rag_api(query_text)

                    if rag_result:
                        ai_response = rag_result.get('rag_response', 'No AI response received.')
                        doc_context = rag_result.get('doc_context', [])

                        context_list = "\n".join([f"<li>Context Match: {textwrap.shorten(doc, width=80, placeholder='...')}</li>" for doc in doc_context])

                        reply = textwrap.dedent(f"""
                        <div style='padding:8px; border-radius:8px; background:#f7fbff; margin-bottom:6px'>
                        <strong>RAG Response:</strong>
                        <p style='margin-top:6px; margin-bottom:10px; font-weight: 500;'>{ai_response}</p>

                        <strong>Knowledge Base Context Found:</strong>
                        <ul style='margin-top:6px; margin-left:15px; padding-left:0;'>
                        {context_list if doc_context else "<li>No relevant documents found.</li>"}
                        </ul>
                        </div>
                        """)
                        st.session_state.rag_chat[sel].append(reply)
    
                        st.rerun()   
                else:
                    st.warning("Empty query!")

        st.markdown("---")
        st.markdown("<div class='card'><strong>Provide Feedback & Update Ticket</strong><div class='muted'>Help Content Manager improve KB</div></div>", unsafe_allow_html=True)

        current_status_options = ["In Progress", "Waiting for User", "Resolved"]
        current_status_index = current_status_options.index(ticket["status"]) if ticket["status"] in current_status_options else 0

        colf1, colf2 = st.columns([2,1])
        with colf1:
            usefulness = st.selectbox("RAG Usefulness", ["Very Useful", "Somewhat Useful", "Not Useful"], key=f"useful_{sel}")
            missing_suggest = st.text_area("Suggest missing KB article (optional)", key=f"missing_{sel}", height=80)
        with colf2:
            new_status = st.selectbox("Update Ticket Status", current_status_options, index=current_status_index, key=f"status_{sel}")

            if st.button("Submit Feedback & Update", key=f"submit_feedback_{sel}"):
                ticket_index = next((i for i, x in enumerate(st.session_state.tickets) if x["id"] == sel), -1)
                if ticket_index != -1:
                    st.session_state.tickets[ticket_index]["status"] = new_status

                st.session_state.agent_feedback.append({
                    "ticket_id": sel,
                    "agent": st.session_state.agent_name,
                    "usefulness": usefulness,
                    "missing_kb": missing_suggest,
                    "status": new_status,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

                if new_status == "Resolved":
                    st.session_state.selected_ticket = None

                st.success("Feedback submitted & ticket updated.")

def content_manager_hub():
    st.markdown(f"<div class='big-title'>Content Manager Hub</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'><strong>Review feedback and manage KB documents.</strong></div>", unsafe_allow_html=True)
    tabs = st.tabs(["Agent Feedback Overview", "Upload Documents", "Knowledge Base Viewer"])

    with tabs[0]:
        st.subheader("Agent Feedback Overview")
        feedback = st.session_state.agent_feedback
        if not feedback:
            st.info("No feedback submitted yet.")
        else:
            try:
                df = pd.DataFrame(feedback)
                total = len(df)
                helpful = df[df["usefulness"] == "Very Useful"].shape[0] if "usefulness" in df and not df.empty else 0
                col1, col2 = st.columns(2)
                col1.metric("Total Feedbacks", total)
                col2.metric("Very Useful", helpful)

                st.markdown("---")
                if not df.empty and "usefulness" in df:
                    benefit = df["usefulness"].value_counts()
                    st.markdown("Feedback Distribution by Usefulness:")
                    st.bar_chart(benefit)

                st.markdown("---")
                st.markdown("#### Detailed Feedback List")
                for _, row in df.sort_values(by="timestamp", ascending=False).iterrows():
                    st.markdown(f"<div class='card' style='margin-bottom:10px'><strong>Ticket {row['ticket_id']}</strong><div class='muted'>By {row['agent']} · {row['timestamp']}</div><div style='margin-top:8px'><b>RAG Usefulness:</b> {row['usefulness']}<br/><b>Missing KB Suggestion:</b> {row['missing_kb'] if row['missing_kb'] else '—'}<br/><b>Final Status:</b> {row['status']}</div></div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing feedback data: {e}")

    with tabs[1]:
        st.subheader("Upload New KB Document")
        st.markdown("<div class='muted'>Upload .docx files to improve the knowledge base (will be saved into backend docs folder).</div>", unsafe_allow_html=True)
        upload = st.file_uploader("Upload .docx (multiple allowed)", type=["docx"], accept_multiple_files=True)
        if upload:
            for f in upload:
                try:
                    save_path = save_uploaded_file(f, target_dir=BACKEND_DOCS_DIR)
                    st.success(f"Saved: To KB")
                except Exception as e:
                    st.error(f"Failed to save {f.name}: {e}")
        st.markdown("")
        if st.button("Trigger KB Rebuild (Simulated)"):
            st.info("Initiating knowledge base embedding and indexing...")
            st.success("Knowledge Base successfully rebuilt!")

    with tabs[2]:
        st.subheader("Knowledge Base Snapshot")
        kb_dir = BACKEND_DOCS_DIR
        st.markdown(
            f"<div class='muted'>Displaying contents of Uploaded Docx",
            unsafe_allow_html=True
        )

        if not os.path.exists(kb_dir):
            st.error(f"Directory not found: {kb_dir}")
        else:
            files = sorted(os.listdir(kb_dir))
            if not files:
                st.info("No documents found in the knowledge base directory.")
            else:
                for file in files:
                    file_path = os.path.join(kb_dir, file)
                    if os.path.isfile(file_path):
                        try:
                            last_updated = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
                        except Exception:
                            last_updated = "Unknown"
                        st.markdown(
                            f"""
                            <div class='card' style='margin-bottom:8px'>
                                 <strong>{file}</strong>
                                 <div class='muted'>Last Updated: {last_updated}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

# Router
current_role = st.session_state.logged_in_role
pending_role = st.session_state.pending_role

if current_role == "User":
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
    st.markdown("<h3 class='muted'>Please select your user role from the sidebar to begin.</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

