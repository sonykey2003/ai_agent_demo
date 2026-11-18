"""Turn a conversation's attachments into model input."""

from __future__ import annotations

import base64

from .store import Attachment

ATTACHMENT_POLICY = (
    "\n\nAttachments: the user attached files to this conversation. Their text is "
    "below under 'Attached files'. Answer questions about them from that text and cite "
    "the file id in brackets (e.g. [receipt]). Recognized text from images comes from "
    "OCR and may contain errors. If the attachments don't cover the question, say so."
)

VISION_NOTE = "The images themselves are attached to the user's message."
TEXT_ONLY_NOTE = (
    "You cannot see images: for each image you only have its recognized (OCR) text. "
    "If a question needs visual detail beyond that text, say you can't see the image."
)


def _label(attachment: Attachment) -> str:
    if attachment.kind == "image":
        return f"{attachment.name} — image; recognized text (OCR)"
    pages = attachment.meta.get("pages")
    suffix = f", {pages} pages" if pages else ""
    return f"{attachment.name} — document{suffix}"


def build_context(
    attachments: list[Attachment], max_chars: int, *, vision: bool
) -> tuple[str, dict[str, bool]]:
    """Attachment text for the system prompt, newest first within the budget.

    Returns the context block and, per attachment id, whether its text was cut.
    """
    budget = max_chars
    sections: list[str] = []
    truncated: dict[str, bool] = {}
    for attachment in reversed(attachments):
        text = attachment.text.strip()
        if attachment.kind == "image" and not text:
            body = "(no text recognized)"
        elif budget <= 0:
            body = "(omitted: attachment text budget used up)"
            truncated[attachment.id] = True
        elif len(text) > budget:
            body = f"{text[:budget]}\n[… truncated {len(text) - budget} more characters]"
            budget = 0
            truncated[attachment.id] = True
        else:
            body = text
            budget -= len(text)
            truncated.setdefault(attachment.id, False)
        sections.append(f"[{attachment.cite_id}] {_label(attachment)}:\n{body}")
    has_images = any(a.kind == "image" for a in attachments)
    note = ""
    if has_images:
        note = "\n" + (VISION_NOTE if vision else TEXT_ONLY_NOTE)
    return "Attached files:" + note + "\n\n" + "\n\n".join(reversed(sections)), truncated


def image_blocks(
    attachments: list[Attachment],
    current_turn: int,
    carryover_turns: int,
    detail: str,
) -> list[tuple[Attachment, dict]]:
    """OpenAI-style image_url content blocks for recently sent images."""
    first_turn = current_turn - max(carryover_turns, 1) + 1
    blocks = []
    for attachment in attachments:
        if attachment.kind != "image" or not attachment.data:
            continue
        if attachment.turn is None or attachment.turn < first_turn:
            continue
        encoded = base64.b64encode(attachment.data).decode()
        blocks.append(
            (
                attachment,
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{attachment.mime_type};base64,{encoded}",
                        "detail": detail,
                    },
                },
            )
        )
    return blocks
