"""Validate, normalize, and extract text from uploaded files.

File types are detected from content, never from the name or Content-Type.
Every step is a span under the upload's ``POST /attachments`` server span.
Blocking; call from a worker thread.
"""

from __future__ import annotations

import hashlib
import io
import time
import uuid
import zipfile
from pathlib import Path

from opentelemetry import trace
from opentelemetry.trace import Span
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import get_settings
from . import ocr
from .store import Attachment, AttachmentError, get_store

_tracer = trace.get_tracer("agent_chat.attachments")

_IMAGE_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_TEXT_MIME = {".md": "text/markdown", ".markdown": "text/markdown", ".csv": "text/csv"}
# A PDF page with less text than this is treated as scanned and sent to OCR.
_MIN_PAGE_TEXT = 20
_OCR_OUTPUT_PREVIEW = 4096

ACCEPTED_TYPES = {
    "image": [".png", ".jpg", ".jpeg", ".webp"],
    "document": [".pdf", ".docx", ".txt", ".md", ".csv"],
}


def _sniff(data: bytes) -> str:
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if "word/document.xml" in archive.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            pass
        raise AttachmentError(415, "Unsupported archive; only .docx Word files are accepted.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
    except UnidentifiedImageError:
        fmt = None
    except Image.DecompressionBombError as exc:
        raise AttachmentError(413, f"Image is too large to process: {exc}") from exc
    if fmt:
        if fmt not in _IMAGE_FORMATS:
            raise AttachmentError(415, f"Unsupported image format {fmt}; use PNG, JPEG, or WebP.")
        return "image"
    head = data[:8192]
    if b"\x00" in head:
        raise AttachmentError(415, "Unsupported binary file type.")
    try:
        data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AttachmentError(415, "Text files must be UTF-8 encoded.") from exc
    return "text"


def _set_ocr_attributes(span: Span, result: ocr.OcrResult) -> None:
    span.set_attribute("ocr.engine", ocr.ENGINE_NAME)
    span.set_attribute("ocr.engine.version", ocr.engine_version())
    span.set_attribute("ocr.languages", ocr.LANGUAGES)
    span.set_attribute("ocr.lines", result.lines)
    span.set_attribute("ocr.chars", result.chars)
    span.set_attribute("ocr.mean_confidence", result.mean_confidence)
    span.set_attribute("output.value", result.text[:_OCR_OUTPUT_PREVIEW])


def _ocr_image(image: Image.Image, span_name: str, **attributes) -> ocr.OcrResult:
    with _tracer.start_as_current_span(span_name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        span.set_attribute("ocr.image.width", image.width)
        span.set_attribute("ocr.image.height", image.height)
        result = ocr.recognize(image)
        _set_ocr_attributes(span, result)
        return result


def _process_image(data: bytes) -> tuple[bytes, str, str, dict]:
    settings = get_settings()
    with _tracer.start_as_current_span("normalize image") as span:
        Image.MAX_IMAGE_PIXELS = settings.attachment_image_max_pixels
        try:
            with Image.open(io.BytesIO(data)) as opened:
                fmt = opened.format
                if opened.width * opened.height > settings.attachment_image_max_pixels:
                    raise AttachmentError(
                        413,
                        f"Image is {opened.width}x{opened.height}; the limit is "
                        f"{settings.attachment_image_max_pixels // 1_000_000} megapixels.",
                    )
                opened.load()
                # Apply the EXIF orientation; re-encoding below drops all metadata (incl. GPS).
                image = ImageOps.exif_transpose(opened)
        except (Image.DecompressionBombError, OSError) as exc:
            raise AttachmentError(422, f"Could not read the image: {exc}") from exc

        has_alpha = image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        )
        original_size = image.size
        ocr_copy = image.copy()
        ocr_copy.thumbnail((settings.attachment_ocr_max_px,) * 2)
        model_copy = image.copy()
        model_copy.thumbnail((settings.attachment_image_max_px,) * 2)
        buffer = io.BytesIO()
        if fmt == "PNG" or has_alpha:
            mime = "image/png"
            model_copy.save(buffer, "PNG", optimize=True)
        else:
            mime = "image/jpeg"
            model_copy.convert("RGB").save(buffer, "JPEG", quality=85)
        meta = {
            "width": original_size[0],
            "height": original_size[1],
            "model_width": model_copy.width,
            "model_height": model_copy.height,
        }
        span.set_attribute("image.source_format", fmt or "unknown")
        span.set_attribute("image.width", original_size[0])
        span.set_attribute("image.height", original_size[1])
        span.set_attribute("image.output_mime_type", mime)
        span.set_attribute("image.output_bytes", buffer.tell())

    text = ""
    meta["ocr"] = None
    if settings.attachment_ocr_enabled:
        result = _ocr_image(ocr_copy, "ocr image")
        text = result.text
        meta["ocr"] = {
            "lines": result.lines,
            "chars": result.chars,
            "mean_confidence": result.mean_confidence,
        }
    return buffer.getvalue(), mime, text, meta


def _process_pdf(data: bytes) -> tuple[str, dict]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    settings = get_settings()
    with _tracer.start_as_current_span("extract text") as span:
        span.set_attribute("document.format", "pdf")
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted and not reader.decrypt(""):
                raise AttachmentError(422, "Password-protected PDFs are not supported.")
            page_count = len(reader.pages)
            if page_count > settings.attachment_max_pdf_pages:
                raise AttachmentError(
                    413,
                    f"PDF has {page_count} pages; the limit is "
                    f"{settings.attachment_max_pdf_pages}.",
                )
            page_texts = [(page.extract_text() or "").strip() for page in reader.pages]
        except PdfReadError as exc:
            raise AttachmentError(422, f"Could not read the PDF: {exc}") from exc
        span.set_attribute("document.pages", page_count)

    scanned = [i for i, text in enumerate(page_texts) if len(text) < _MIN_PAGE_TEXT]
    ocr_meta = None
    if scanned and settings.attachment_ocr_enabled:
        import pypdfium2

        to_ocr = scanned[: settings.attachment_ocr_max_pages]
        deadline = time.monotonic() + settings.attachment_ocr_timeout_s
        results: list[ocr.OcrResult] = []
        stopped_for_time = False
        with _tracer.start_as_current_span("ocr pdf") as span:
            span.set_attribute("ocr.pages.scanned", len(scanned))
            pdf = pypdfium2.PdfDocument(data)
            try:
                for index in to_ocr:
                    if time.monotonic() > deadline:
                        stopped_for_time = True
                        break
                    bitmap = pdf[index].render(scale=settings.attachment_ocr_pdf_dpi / 72)
                    image = bitmap.to_pil()
                    image.thumbnail((settings.attachment_ocr_max_px,) * 2)
                    result = _ocr_image(image, "ocr pdf page", **{"ocr.page": index + 1})
                    page_texts[index] = result.text
                    results.append(result)
            finally:
                pdf.close()
            recognized = [r for r in results if r.lines]
            ocr_meta = {
                "pages": len(results),
                "lines": sum(r.lines for r in results),
                "chars": sum(r.chars for r in results),
                "mean_confidence": round(
                    sum(r.mean_confidence for r in recognized) / len(recognized), 4
                )
                if recognized
                else 0.0,
                "skipped_pages": len(scanned) - len(results),
                "stopped_for_time": stopped_for_time,
            }
            span.set_attribute("ocr.engine", ocr.ENGINE_NAME)
            span.set_attribute("ocr.languages", ocr.LANGUAGES)
            span.set_attribute("ocr.pages", ocr_meta["pages"])
            span.set_attribute("ocr.chars", ocr_meta["chars"])
            span.set_attribute("ocr.mean_confidence", ocr_meta["mean_confidence"])
            span.set_attribute("ocr.skipped_pages", ocr_meta["skipped_pages"])

    if page_count > 1:
        text = "\n\n".join(
            f"--- page {i + 1} ---\n{t}" for i, t in enumerate(page_texts) if t
        )
    else:
        text = page_texts[0] if page_texts else ""
    if not text.strip():
        hint = (
            "" if settings.attachment_ocr_enabled else " (scanned PDF? text recognition is off)"
        )
        raise AttachmentError(422, f"No readable text found in the PDF{hint}.")
    return text, {"pages": page_count, "ocr": ocr_meta}


def _process_docx(data: bytes) -> tuple[str, dict]:
    import docx

    with _tracer.start_as_current_span("extract text") as span:
        span.set_attribute("document.format", "docx")
        document = docx.Document(io.BytesIO(data))
        parts = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        text = "\n".join(parts).strip()
    if not text:
        raise AttachmentError(422, "No readable text found in the Word document.")
    return text, {"ocr": None}


def process_upload(conversation_id: str, filename: str, data: bytes) -> Attachment:
    """Turn raw upload bytes into a stored attachment."""
    name = Path(filename or "upload").name[:200]
    with _tracer.start_as_current_span("validate attachment") as span:
        if not data:
            raise AttachmentError(422, "The file is empty.")
        kind = _sniff(data)
        span.set_attribute("attachment.detected_type", kind)

    image_bytes: bytes | None = None
    if kind == "image":
        image_bytes, mime, text, meta = _process_image(data)
    elif kind == "pdf":
        mime = "application/pdf"
        text, meta = _process_pdf(data)
    elif kind == "docx":
        mime = _DOCX_MIME
        text, meta = _process_docx(data)
    else:
        mime = _TEXT_MIME.get(Path(name).suffix.lower(), "text/plain")
        with _tracer.start_as_current_span("extract text") as span:
            span.set_attribute("document.format", "text")
            text = data.decode("utf-8-sig").strip()
        if not text:
            raise AttachmentError(422, "The file has no text.")
        meta = {"ocr": None}
    settings = get_settings()
    if len(text) > settings.attachment_max_text_chars:
        # Only the prompt budget is ever used; don't hold huge texts in memory.
        meta["text_truncated_from"] = len(text)
        text = text[: settings.attachment_max_text_chars]
    meta["chars"] = len(text)

    store = get_store()
    attachment = Attachment(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        name=name,
        kind="image" if kind == "image" else "document",
        mime_type=mime,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        cite_id=store.unique_cite_id(conversation_id, name),
        data=image_bytes,
        text=text,
        meta=meta,
    )
    store.add(attachment)
    return attachment
