import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environmental variables
load_dotenv()

# Import logic functions and constants from app.py
from app import (
    ALL_TYPES,
    IMAGE_TYPES,
    ask,
    get_upload_dir,
    ingest_all,
)

# Page configuration
st.set_page_config(
    page_title="DocMe",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state initialization
for _k, _v in {
    "ready":         False,
    "vector_store":  None,
    "llm":           None,
    "messages":      [],
    "doc_names":     [],
    "process_log":   [],
    "session_id":    "",
    "chunk_size":    1000,
    "chunk_overlap": 200,
    "top_k":         6,
    "llm_model":     "llama-3.3-70b-versatile",
}.items():
    if _k not in st.session_state:
        if _k == "session_id":
            st.session_state[_k] = str(uuid.uuid4())
        else:
            st.session_state[_k] = _v

# Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] {
    background: #0d0f17;
    border-right: 1px solid #1c1f2e;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
header[data-testid="stHeader"] { display: none; }
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: #fff !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.55rem 1rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 0.5rem 0.8rem;
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# Sidebar layout
with st.sidebar:
    st.markdown("## 📄 DocMind")
    st.caption("PDFs and images · Answers grounded in your files")

    # API Keys Validation
    openai_ready = bool(os.getenv("OPENAI_API_KEY"))
    groq_ready = bool(os.getenv("GROQ_API_KEY"))
    if not openai_ready or not groq_ready:
        st.error("⚠️ Missing API Keys in Environment", icon="🔑")
        if not openai_ready:
            st.info("Missing `OPENAI_API_KEY` for Embeddings", icon="ℹ️")
        if not groq_ready:
            st.info("Missing `GROQ_API_KEY` for Chat LLM", icon="ℹ️")
        st.divider()

    try:
        import paddleocr as _poc
        st.success(f"PaddleOCR {getattr(_poc,'__version__','v3')} ready", icon="🔍")
    except Exception:
        st.error("PaddleOCR not installed", icon="❌")

    st.divider()

    # Advanced Settings Slider/Inputs
    with st.expander("⚙️ Advanced Settings"):
        chunk_size = st.number_input("Chunk Size", min_value=100, max_value=5000, value=st.session_state.chunk_size, step=100)
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=2000, value=st.session_state.chunk_overlap, step=50)
        top_k = st.slider("Retrieve Top-K", min_value=1, max_value=20, value=st.session_state.top_k)
        llm_model = st.selectbox(
            "LLM Model",
            options=["llama-3.3-70b-versatile", "llama-3-1-8b-instant", "mixtral-8x7b-32768"],
            index=["llama-3.3-70b-versatile", "llama-3-1-8b-instant", "mixtral-8x7b-32768"].index(st.session_state.llm_model) if st.session_state.llm_model in ["llama-3.3-70b-versatile", "llama-3-1-8b-instant", "mixtral-8x7b-32768"] else 0
        )
        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap
        st.session_state.top_k = top_k
        st.session_state.llm_model = llm_model

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDFs or images",
        type=ALL_TYPES,
        accept_multiple_files=True,
        label_visibility="visible",
        help="Supported: PDF, PNG, JPG, JPEG, TIFF, BMP, WEBP",
    )

    if st.button("⚡  Process Files"):
        if not uploaded_files:
            st.warning("Upload at least one file first.")
        elif not openai_ready or not groq_ready:
            st.error("Please configure the required environment API keys before processing.")
        else:
            session_upload_dir = get_upload_dir()
            for f in os.listdir(session_upload_dir):
                os.remove(os.path.join(session_upload_dir, f))
            for f in uploaded_files:
                dest = os.path.join(session_upload_dir, f.name)
                with open(dest, "wb") as fh:
                    fh.write(f.getvalue())

            st.session_state.doc_names = [f.name for f in uploaded_files]

            ok = ingest_all(session_upload_dir)
            if ok:
                st.session_state.messages = []
                st.rerun()

    if st.session_state.process_log:
        st.divider()
        st.markdown("**Ingestion log**")
        for entry in st.session_state.process_log:
            st.markdown(f"<small>{entry}</small>", unsafe_allow_html=True)

    if st.session_state.doc_names:
        st.divider()
        st.markdown("**Loaded files**")
        for name in st.session_state.doc_names:
            ext = Path(name).suffix.lower()
            icon = "🖼️" if ext.lstrip(".") in IMAGE_TYPES else "📄"
            st.markdown(f"<small>{icon} {name}</small>", unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️  Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Groq · LangChain · OpenAI Embeddings · PaddleOCR v3")

# Main Chat Interface
st.markdown("# DocMe")
st.caption("Ask questions about your uploaded PDFs and images.")
st.divider()

if not st.session_state.ready:
    st.info(
        "👈  Upload **PDFs or images** in the sidebar and click **Process Files**.",
        icon="📂",
    )
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask anything about your files…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = ask(user_input)
            except Exception as exc:
                answer = f"⚠️ Error: {exc}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
