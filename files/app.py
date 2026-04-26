# =============================================================================
#  DocMind — RAG Chatbot
#  Accepts: PDF files  +  Images (PNG, JPG, JPEG, TIFF, BMP, WEBP)
#  Stack:   Streamlit · PaddleOCR v3 · LangChain · Groq · OpenAI Embeddings
#
#  pip install streamlit python-dotenv pdf2image numpy pypdf Pillow \
#              paddleocr langchain langchain-core langchain-community \
#              langchain-openai langchain-groq
#
#  .env → OPENAI_API_KEY=...   GROQ_API_KEY=...
# =============================================================================

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

st.set_page_config(
    page_title="DocMe",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paddleocr import PaddleOCR
from pdf2image import convert_from_path


# =============================================================================
#  Config
# =============================================================================

UPLOAD_DIR    = "./doc_files/"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
TOP_K         = 6
OCR_DPI       = 200
MIN_TEXT_LEN  = 50
LLM_MODEL     = "llama-3.3-70b-versatile"
EMBED_MODEL   = "text-embedding-3-large"

PDF_TYPES   = ["pdf"]
IMAGE_TYPES = ["png", "jpg", "jpeg", "tiff", "bmp", "webp"]
ALL_TYPES   = PDF_TYPES + IMAGE_TYPES


# =============================================================================
#  Session state
# =============================================================================

for _k, _v in {
    "ready":        False,
    "vector_store": None,
    "llm":          None,
    "messages":     [],
    "doc_names":    [],
    "process_log":  [],
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# =============================================================================
#  PaddleOCR v3  —  cached singleton, predict() API
# =============================================================================

@st.cache_resource(show_spinner="Loading OCR model…")
def load_ocr() -> PaddleOCR:
    return PaddleOCR(lang="en")


def _parse_result(result_obj) -> str:
    """Extract text from one PaddleOCR v3 Result object (.predict() output)."""
    try:
        payload = result_obj.json
        res     = payload.get("res", payload)
        texts   = res.get("rec_texts",  []) or []
        scores  = res.get("rec_scores", []) or []
    except Exception:
        return ""

    lines = []
    for text, score in zip(texts, scores):
        try:
            if float(score) >= 0.5 and str(text).strip():
                lines.append(str(text).strip())
        except (TypeError, ValueError):
            if str(text).strip():
                lines.append(str(text).strip())
    return " ".join(lines)


def ocr_numpy(img_np: np.ndarray) -> str:
    """Run OCR on a numpy image array and return extracted text."""
    results = load_ocr().predict(img_np)
    return _parse_result(results[0]) if results else ""


def ocr_pdf_page(pdf_path: str, page_index: int) -> str:
    """OCR a single page (0-based) of a PDF."""
    images = convert_from_path(
        pdf_path,
        first_page=page_index + 1,
        last_page=page_index + 1,
        dpi=OCR_DPI,
    )
    return ocr_numpy(np.array(images[0])) if images else ""


def ocr_image_file(image_path: str) -> str:
    """OCR a standalone image file (PNG, JPG, TIFF, etc.)."""
    img = Image.open(image_path).convert("RGB")
    return ocr_numpy(np.array(img))


# =============================================================================
#  Ingestion helpers
# =============================================================================

def _ingest_pdf(path: str, log: list[str]) -> list[Document]:
    """Load a PDF; OCR any scanned/image-only pages."""
    docs: list[Document] = []
    pages = PyPDFLoader(path).load()

    for page in pages:
        text     = page.page_content.strip()
        page_idx = page.metadata.get("page", 0)
        label    = f"{Path(path).name} p.{page_idx + 1}"

        if len(text) >= MIN_TEXT_LEN:
            docs.append(page)
            log.append(f"PDF text ✓  {label}")
        else:
            ocr_text = ocr_pdf_page(path, page_idx)
            if len(ocr_text.strip()) > 10:
                docs.append(Document(
                    page_content=ocr_text,
                    metadata={**page.metadata, "extraction": "ocr"},
                ))
                log.append(f"PDF OCR  ✓  {label}")
            else:
                log.append(f"PDF OCR  ✗  {label}  (blank)")

    return docs


def _ingest_image(path: str, log: list[str]) -> list[Document]:
    """OCR a standalone image file and return it as a single Document."""
    name     = Path(path).name
    ocr_text = ocr_image_file(path)

    if len(ocr_text.strip()) > 10:
        log.append(f"Image OCR ✓  {name}")
        return [Document(
            page_content=ocr_text,
            metadata={"source": path, "page": 0, "extraction": "ocr"},
        )]
    else:
        log.append(f"Image OCR ✗  {name}  (no text found)")
        return []


def ingest_all(directory: str) -> bool:
    """
    Ingest every file in `directory`.

    PDFs  → PyPDF text layer, OCR fallback per page.
    Images → OCR directly with PaddleOCR v3.
    Then chunk → embed → store in InMemoryVectorStore.
    """
    log: list[str]      = []
    all_docs: list[Document] = []

    files = list(Path(directory).iterdir())
    if not files:
        st.error("No files found in the upload folder.")
        return False

    with st.spinner("Processing files…"):
        for file in files:
            ext = file.suffix.lower().lstrip(".")
            if ext == "pdf":
                all_docs.extend(_ingest_pdf(str(file), log))
            elif ext in IMAGE_TYPES:
                all_docs.extend(_ingest_image(str(file), log))
            else:
                log.append(f"Skipped  {file.name}  (unsupported type)")

    usable = [d for d in all_docs if len(d.page_content.strip()) > 10]
    if not usable:
        st.error("No usable text extracted — try a clearer image or a text-layer PDF.")
        return False

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    ).split_documents(usable)

    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model=EMBED_MODEL),
    )

    st.session_state.vector_store = vector_store
    st.session_state.llm          = ChatGroq(model=LLM_MODEL)
    st.session_state.ready        = True

    n_pdf = sum(1 for d in all_docs if d.metadata.get("source", "").endswith(".pdf")
                or Path(d.metadata.get("source","")).suffix.lower() == ".pdf")
    n_img = len(all_docs) - n_pdf
    st.session_state.process_log = [
        f"✅ {len(usable)} usable page(s) — "
        f"{n_pdf} from PDF · {n_img} from images"
    ] + log

    return True


# =============================================================================
#  RAG query
# =============================================================================

SYSTEM_PROMPT = """\
You are DocMind, a precise document-analysis assistant.

Every user message contains DOCUMENT CONTEXT — passages extracted from the \
user's uploaded files (PDFs and/or images).

Rules:
• Answer ONLY from the provided DOCUMENT CONTEXT. Never use outside knowledge.
• Cite the source filename and page number for every claim.
• If the context does not contain the answer, say so clearly and describe \
  what the context does contain.
• For summarisation requests, write a structured summary covering ALL chunks.
• Be concise and factual. Use bullet points where they help clarity.\
"""


def ask(question: str) -> str:
    hits = st.session_state.vector_store.similarity_search(question, k=TOP_K)
    if not hits:
        return "No relevant passages found in the uploaded files."

    context = "\n\n---\n\n".join(
        "[{src}, page {pg} | {ext}]\n{text}".format(
            src=Path(d.metadata.get("source", "?")).name,
            pg=int(d.metadata.get("page", 0)) + 1,
            ext=d.metadata.get("extraction", "text-layer"),
            text=d.page_content,
        )
        for d in hits
    )

    history = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in st.session_state.messages[-6:]
    ]

    response = st.session_state.llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        *history,
        HumanMessage(content=f"DOCUMENT CONTEXT:\n\n{context}\n\n---\n\nQUESTION: {question}"),
    ])
    return response.content


# =============================================================================
#  UI
# =============================================================================

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


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 DocMind")
    st.caption("PDFs and images · Answers grounded in your files")

    try:
        import paddleocr as _poc
        st.success(f"PaddleOCR {getattr(_poc,'__version__','v3')} ready", icon="🔍")
    except Exception:
        st.error("PaddleOCR not installed", icon="❌")

    st.divider()

    # ── uploader accepts both PDFs and images ─────────────────────────────────
    files = st.file_uploader(
        "Upload PDFs or images",
        type=ALL_TYPES,
        accept_multiple_files=True,
        label_visibility="visible",
        help="Supported: PDF, PNG, JPG, JPEG, TIFF, BMP, WEBP",
    )

    if st.button("⚡  Process Files"):
        if not files:
            st.warning("Upload at least one file first.")
        else:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            for f in os.listdir(UPLOAD_DIR):
                os.remove(os.path.join(UPLOAD_DIR, f))
            for f in files:
                dest = os.path.join(UPLOAD_DIR, f.name)
                with open(dest, "wb") as fh:
                    fh.write(f.getvalue())

            st.session_state.doc_names = [f.name for f in files]

            ok = ingest_all(UPLOAD_DIR)
            if ok:
                st.session_state.messages = []
                st.rerun()

    # ── ingestion log ─────────────────────────────────────────────────────────
    if st.session_state.process_log:
        st.divider()
        st.markdown("**Ingestion log**")
        for entry in st.session_state.process_log:
            st.markdown(f"<small>{entry}</small>", unsafe_allow_html=True)

    # ── loaded files ──────────────────────────────────────────────────────────
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


# ── main ──────────────────────────────────────────────────────────────────────
st.markdown("# DocMind")
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