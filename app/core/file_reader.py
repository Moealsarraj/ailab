"""File text extractor — supports .docx, .pdf, .txt.

Reusable: copy this file to any Flask project's app/core/ directory.
Dependencies: pypdf>=4.0  (for PDF support — add to requirements.txt)
DOCX and TXT use Python built-ins only (no extra packages needed).
"""
import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def extract_text(file_storage) -> str:
    """Extract plain text from a Werkzeug FileStorage object.

    Supports .pdf, .docx, .txt files up to 10 MB.
    Returns extracted text as a string.
    Raises ValueError for unsupported types, oversized files, or parse errors.
    """
    filename = file_storage.filename or ""
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext or '(none)'}'. Allowed: PDF, DOCX, TXT"
        )

    data = file_storage.read()
    if len(data) > MAX_FILE_SIZE:
        raise ValueError("File too large (max 10 MB)")
    if not data:
        raise ValueError("File is empty")

    if ext == ".txt":
        return data.decode("utf-8", errors="replace").strip()
    if ext == ".docx":
        return _read_docx(io.BytesIO(data))
    if ext == ".pdf":
        return _read_pdf(io.BytesIO(data))

    raise ValueError(f"Unhandled extension: {ext}")


def _read_docx(stream: io.BytesIO) -> str:
    """Extract text from a .docx file using built-in zipfile + xml.etree (no deps)."""
    try:
        with zipfile.ZipFile(stream) as z:
            with z.open("word/document.xml") as f:
                tree = ET.parse(f)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError(f"Could not read Word document: {exc}")

    root = tree.getroot()
    paragraphs = []
    for para in root.iter(f"{{{_WORD_NS}}}p"):
        # Collect all text runs, preserving spaces
        parts = []
        for node in para.iter():
            if node.tag == f"{{{_WORD_NS}}}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{{{_WORD_NS}}}br":
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)

    text = "\n\n".join(paragraphs)
    if not text.strip():
        raise ValueError("No readable text found in the Word document")
    return text


def _read_pdf(stream: io.BytesIO) -> str:
    """Extract text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ValueError("pypdf not installed — run: pip install pypdf")

    try:
        reader = PdfReader(stream)
    except Exception as exc:
        raise ValueError(f"Could not read PDF: {exc}")

    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())

    text = "\n\n".join(pages)
    if not text.strip():
        raise ValueError("No readable text found in the PDF (may be image-based)")
    return text
