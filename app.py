from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paddleocr import PaddleOCR
from pdf2image import convert_from_path

UPLOAD_DIR    = "./files/doc_files/"
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


def get_upload_dir() -> str:
    path = os.path.join(UPLOAD_DIR, st.session_state.session_id)
    os.makedirs(path, exist_ok=True)
    return path


@st.cache_resource(show_spinner="Loading OCR model…")
def load_ocr() -> PaddleOCR:
    return PaddleOCR(lang="en", show_log=False, use_angle_cls=True)


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


def _ingest_pdf(path: str, log: list[str]) -> list[Document]:
    """Load a PDF; OCR any scanned/image-only pages."""
    docs: list[Document] = []
    pages = PyPDFLoader(path).load()

    total_pages = len(pages)
    if total_pages > 1:
        progress_text = f"Processing {Path(path).name}..."
        progress_bar = st.progress(0.0, text=progress_text)
    else:
        progress_bar = None

    for idx, page in enumerate(pages):
        if progress_bar:
            progress_bar.progress((idx) / total_pages, text=f"{progress_text} (page {idx+1}/{total_pages})")
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

    if progress_bar:
        progress_bar.empty()

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
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap,
    ).split_documents(usable)

    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model=EMBED_MODEL),
    )

    st.session_state.vector_store = vector_store
    st.session_state.llm          = ChatGroq(model=st.session_state.llm_model)
    st.session_state.ready        = True

    n_pdf = sum(1 for d in all_docs if d.metadata.get("source", "").endswith(".pdf")
                or Path(d.metadata.get("source","")).suffix.lower() == ".pdf")
    n_img = len(all_docs) - n_pdf
    st.session_state.process_log = [
        f"✅ {len(usable)} usable page(s) — "
        f"{n_pdf} from PDF · {n_img} from images"
    ] + log

    return True


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
    hits = st.session_state.vector_store.similarity_search(question, k=st.session_state.top_k)
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
