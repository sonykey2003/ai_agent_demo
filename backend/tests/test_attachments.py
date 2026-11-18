import asyncio
import io
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from PIL import Image, ImageDraw, ImageFont
from starlette.datastructures import UploadFile

sys.path.insert(0, str(Path(__file__).parents[1]))

from app import main
from app.agent import graph
from app.attachments.processing import process_upload
from app.attachments.store import Attachment, AttachmentError, get_store
from app.config import ProviderConfig, get_settings
from app.main import ChatRequest
from app.telemetry import otel

CONV = "conv-attachments"
_FONT_DIRS = ["/System/Library/Fonts", "/usr/share/fonts", "/Library/Fonts"]


@pytest.fixture(autouse=True)
def _clean_store():
    get_store().clear()
    yield
    get_store().clear()


def _font(names: list[str], size: int):
    for directory in _FONT_DIRS:
        for path in Path(directory).rglob("*"):
            if path.name in names:
                return ImageFont.truetype(str(path), size)
    return None


def _png(lines: list[str], font=None) -> bytes:
    image = Image.new("RGB", (900, 120 + 90 * len(lines)), "white")
    draw = ImageDraw.Draw(image)
    font = font or _font(["Arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"], 44)
    for i, line in enumerate(lines):
        draw.text((40, 50 + 90 * i), line, fill="black", font=font)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _text_pdf(text: str) -> bytes:
    """Minimal one-page PDF with a real text layer."""
    stream = f"BT /F1 18 Tf 72 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return out


def _no_ocr(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "attachment_ocr_enabled", False)


def test_type_is_detected_from_content_not_name(monkeypatch) -> None:
    _no_ocr(monkeypatch)
    attachment = process_upload(CONV, "report.pdf", _png(["hello"]))
    assert attachment.kind == "image"
    assert attachment.mime_type == "image/png"

    with pytest.raises(AttachmentError) as binary:
        process_upload(CONV, "photo.png", b"\x00\x01\x02 not an image")
    assert binary.value.status_code == 415

    with pytest.raises(AttachmentError) as broken:
        process_upload(CONV, "x.pdf", b"%PDF-1.4 garbage")
    assert broken.value.status_code == 422


def test_image_is_rotated_downscaled_and_stripped_of_exif(monkeypatch) -> None:
    _no_ocr(monkeypatch)
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° on display
    exif[0x010F] = "SpyCam"
    buffer = io.BytesIO()
    Image.new("RGB", (4000, 2000), "red").save(buffer, "JPEG", exif=exif)

    attachment = process_upload(CONV, "photo.jpg", buffer.getvalue())

    stored = Image.open(io.BytesIO(attachment.data))
    assert stored.format == "JPEG"
    assert stored.size == (784, 1568)
    assert not stored.getexif()
    assert (attachment.meta["width"], attachment.meta["height"]) == (2000, 4000)


def test_pdf_and_docx_text_is_extracted(monkeypatch) -> None:
    import docx

    _no_ocr(monkeypatch)
    pdf = process_upload(CONV, "statement.pdf", _text_pdf("Balance due 512.40 USD"))
    assert pdf.kind == "document"
    assert "Balance due 512.40" in pdf.text
    assert pdf.meta["pages"] == 1

    document = docx.Document()
    document.add_paragraph("Claim number 77-31")
    buffer = io.BytesIO()
    document.save(buffer)
    word = process_upload(CONV, "claim.docx", buffer.getvalue())
    assert word.text == "Claim number 77-31"
    assert word.cite_id == "claim"


def test_scanned_pdf_without_ocr_is_rejected(monkeypatch) -> None:
    _no_ocr(monkeypatch)
    buffer = io.BytesIO()
    Image.open(io.BytesIO(_png(["TOTAL 99.00"]))).save(buffer, "PDF")
    with pytest.raises(AttachmentError) as exc:
        process_upload(CONV, "scan.pdf", buffer.getvalue())
    assert exc.value.status_code == 422


def test_upload_stops_reading_past_the_size_limit() -> None:
    upload = UploadFile(file=io.BytesIO(b"x" * (3 * 1024 * 1024)), filename="big.txt")
    with pytest.raises(AttachmentError) as exc:
        asyncio.run(main._read_limited(upload, 2 * 1024 * 1024))
    assert exc.value.status_code == 413


def test_upload_endpoint_returns_descriptor_and_errors() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(main.app)
    ok = client.post(
        "/attachments",
        data={"conversation_id": CONV},
        files={"file": ("notes.md", b"# Notes\nRefund window is 30 days.", "text/markdown")},
    )
    assert ok.status_code == 200
    assert ok.json()["kind"] == "document"
    assert ok.json()["mime_type"] == "text/markdown"

    spoofed = client.post(
        "/attachments",
        data={"conversation_id": CONV},
        files={"file": ("cat.png", b"\x00\x00binary", "image/png")},
    )
    assert spoofed.status_code == 415
    assert "error" in spoofed.json()

    removed = client.delete(f"/attachments/{ok.json()['id']}", params={"conversation_id": CONV})
    assert removed.status_code == 200
    assert get_store().get(CONV, ok.json()["id"]) is None


def _ocr_available() -> bool:
    try:
        import rapidocr  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _ocr_available(), reason="rapidocr not installed")
def test_ocr_reads_english_chinese_and_scanned_pdf_pages() -> None:
    receipt = process_upload(CONV, "receipt.png", _png(["TOTAL DUE 1234.56", "Coffee Beans"]))
    assert "TOTAL DUE 1234.56" in receipt.text
    assert receipt.meta["ocr"]["lines"] == 2

    cjk = _font(["STHeiti Medium.ttc", "Hiragino Sans GB.ttc", "NotoSansCJK-Regular.ttc"], 44)
    if cjk is not None:
        invoice = process_upload(CONV, "invoice.png", _png(["合计 356.00"], font=cjk))
        assert "合计" in invoice.text

    page = Image.open(io.BytesIO(_png(["Statement total 88.10"])))
    buffer = io.BytesIO()
    page.save(buffer, "PDF", resolution=72)
    scan = process_upload(CONV, "scan.pdf", buffer.getvalue())
    assert "88.10" in scan.text
    assert scan.meta["ocr"]["pages"] == 1


def _stored(kind: str, text: str, name: str, data: bytes | None = None) -> Attachment:
    attachment = Attachment(
        id=f"id-{name}",
        conversation_id=CONV,
        name=name,
        kind=kind,
        mime_type="image/png" if kind == "image" else "text/plain",
        size_bytes=10,
        sha256="x",
        cite_id=name.split(".")[0],
        data=data,
        text=text,
    )
    get_store().add(attachment)
    return attachment


def _provider(vision: bool) -> ProviderConfig:
    return ProviderConfig(
        name="test",
        label="Test",
        base_url="http://unused",
        api_key="x",
        model="test-model",
        display="Test",
        genai_system="openai",
        max_tokens=10,
        frequency_penalty=0.0,
        supports_vision=vision,
    )


def test_text_only_model_gets_ocr_text_or_a_clear_error() -> None:
    _stored("image", "TOTAL 12.00", "receipt.png", b"png")
    _stored("image", "", "sunset.png", b"png")
    request = ChatRequest(message="total?", conversation_id=CONV, attachments=["id-receipt.png"])

    items, notice = main._prepare_attachments(request, _provider(vision=False))
    assert [a.name for a in items] == ["receipt.png"]
    assert notice["items"][0]["delivery"] == "ocr_only"

    blank = request.model_copy(update={"attachments": ["id-sunset.png"]})
    with pytest.raises(AttachmentError) as exc:
        main._prepare_attachments(blank, _provider(vision=False))
    assert exc.value.status_code == 422

    _, vision_notice = main._prepare_attachments(blank, _provider(vision=True))
    assert vision_notice["items"][0]["delivery"] == "image+ocr"


def test_unknown_attachment_id_ends_the_turn_with_an_error() -> None:
    request = ChatRequest(message="hi", provider="local", conversation_id=CONV, attachments=["nope"])
    events = asyncio.run(_collect(main._event_stream(request)))
    assert events[0]["type"] == "error"
    assert events[-1] == {"type": "done"}


async def _collect(stream) -> list[dict]:
    import json

    return [json.loads(chunk.removeprefix("data: ").strip()) async for chunk in stream]


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[list] = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="ok")


def _run_turn(agent, text: str, attachment_ids: list[str]) -> None:
    get_store().begin_turn(CONV, attachment_ids)
    asyncio.run(
        agent.ainvoke(
            {"messages": [("user", text)]},
            config={"configurable": {"thread_id": CONV}},
        )
    )


@pytest.mark.parametrize("vision", [True, False])
def test_agent_sends_images_only_to_vision_models_and_keeps_history_plain(
    monkeypatch, vision
) -> None:
    fake = _FakeModel()
    monkeypatch.setattr(graph, "make_chat_model", lambda **kwargs: fake)
    monkeypatch.setattr(graph, "provider_supports_vision", lambda cfg: vision)
    monkeypatch.setattr(graph, "retrieve_context", lambda query, collection=None: ("", []))
    monkeypatch.setattr(get_settings(), "attachment_image_carryover_turns", 2)
    graph.build_agent.cache_clear()
    graph._CHECKPOINTER.delete_thread(CONV)
    agent = graph.build_agent(provider="openai", temperature=0.0, domain=None)
    _stored("image", "TOTAL 12.00", "receipt.png", b"\x89PNG")

    _run_turn(agent, "What is the total?", ["id-receipt.png"])
    _run_turn(agent, "And the tax?", [])
    _run_turn(agent, "Thanks", [])

    def last_human(call):
        return [m for m in call if isinstance(m, HumanMessage)][-1]

    def text_of(message):
        if isinstance(message.content, str):
            return message.content
        return " ".join(b["text"] for b in message.content if b["type"] == "text")

    first, second, third = fake.calls
    assert "[receipt] receipt.png" in first[0].content
    assert "TOTAL 12.00" in third[0].content
    assert "(Attached to this message: receipt.png [receipt])" in text_of(last_human(first))
    assert "Attached to this message" not in text_of(last_human(second))
    if vision:
        images = [b for b in last_human(first).content if b["type"] == "image_url"]
        assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")
        # Carried over one turn, labelled as earlier; dropped after that.
        assert "receipt.png (from an earlier message)" in text_of(last_human(second))
        assert isinstance(last_human(third).content, str)
    else:
        assert "You cannot see images" in first[0].content
        assert all(isinstance(last_human(c).content, str) for c in fake.calls)

    history = agent.get_state({"configurable": {"thread_id": CONV}}).values["messages"]
    assert all(isinstance(m.content, str) for m in history)
    graph.build_agent.cache_clear()


def test_exported_spans_never_carry_base64_media() -> None:
    sink = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(otel.MediaScrubbingSpanExporter(sink)))
    payload = "iVBORw0KGgo" * 400
    with provider.get_tracer("t").start_as_current_span("llm") as span:
        span.set_attribute("gen_ai.prompt.0.content", f'[{{"url": "data:image/png;base64,{payload}"}}]')
        span.add_event("prompt", {"content": f"data:image/jpeg;base64,{payload}"})
        span.set_attribute("plain", "hello")

    (exported,) = sink.get_finished_spans()
    assert payload not in exported.attributes["gen_ai.prompt.0.content"]
    assert "<omitted" in exported.attributes["gen_ai.prompt.0.content"]
    assert payload not in exported.events[0].attributes["content"]
    assert exported.attributes["plain"] == "hello"
    provider.shutdown()


def test_galileo_llm_message_uses_placeholder_for_images() -> None:
    message = HumanMessage(
        content=[
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
    )
    assert main._llm_message(message)["content"] == "What is this?\n[image]"
