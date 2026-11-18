"""In-memory, per-conversation attachment store.

Conversation history lives in LangGraph's in-memory checkpointer, so attachments
deliberately have the same lifetime: they disappear on restart along with the
conversations that reference them.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from ..config import get_settings


class AttachmentError(Exception):
    """A user-facing attachment problem, carrying the HTTP status to return."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass
class Attachment:
    id: str
    conversation_id: str
    name: str
    kind: Literal["image", "document"]
    mime_type: str
    size_bytes: int
    sha256: str
    cite_id: str
    # Normalized image sent to vision models; None for documents.
    data: bytes | None
    # Extracted document text or recognized (OCR) image text.
    text: str
    meta: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    # Chat turn the attachment was sent with; None until it is sent.
    turn: int | None = None

    def descriptor(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "cite_id": self.cite_id,
            **self.meta,
        }


def _slug(name: str) -> str:
    stem = Path(name).stem.lower()
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-")[:40] or "file"


class AttachmentStore:
    def __init__(self) -> None:
        self._items: dict[str, Attachment] = {}
        self._turns: dict[str, int] = {}
        self._lock = threading.Lock()

    def unique_cite_id(self, conversation_id: str, name: str) -> str:
        base = _slug(name)
        with self._lock:
            taken = {
                a.cite_id
                for a in self._items.values()
                if a.conversation_id == conversation_id
            }
        cite, n = base, 2
        while cite in taken:
            cite, n = f"{base}-{n}", n + 1
        return cite

    def add(self, attachment: Attachment) -> None:
        settings = get_settings()
        self.purge_expired(settings.attachment_ttl_hours * 3600)
        with self._lock:
            count = sum(
                1
                for a in self._items.values()
                if a.conversation_id == attachment.conversation_id
            )
            if count >= settings.attachment_max_per_conversation:
                raise AttachmentError(
                    409,
                    f"This conversation already has {count} attachments "
                    f"(limit {settings.attachment_max_per_conversation}).",
                )
            self._items[attachment.id] = attachment

    def get(self, conversation_id: str, attachment_id: str) -> Attachment | None:
        with self._lock:
            item = self._items.get(attachment_id)
        if item is None or item.conversation_id != conversation_id:
            return None
        return item

    def delete(self, conversation_id: str, attachment_id: str) -> bool:
        with self._lock:
            item = self._items.get(attachment_id)
            if item is None or item.conversation_id != conversation_id:
                return False
            del self._items[attachment_id]
            return True

    def begin_turn(
        self, conversation_id: str, attachment_ids: list[str]
    ) -> list[Attachment]:
        """Advance the conversation's turn and mark attachments as sent with it.

        Called for every chat turn (with or without attachments) so image
        carry-over counts real turns.
        """
        with self._lock:
            items = []
            for attachment_id in attachment_ids:
                item = self._items.get(attachment_id)
                if item is None or item.conversation_id != conversation_id:
                    raise AttachmentError(
                        404, f"Attachment {attachment_id} not found for this conversation."
                    )
                items.append(item)
            turn = self._turns.get(conversation_id, 0) + 1
            self._turns[conversation_id] = turn
            for item in items:
                item.turn = turn
            return items

    def current_turn(self, conversation_id: str) -> int:
        with self._lock:
            return self._turns.get(conversation_id, 0)

    def for_conversation(self, conversation_id: str) -> list[Attachment]:
        """Attachments already sent in this conversation, oldest first."""
        with self._lock:
            items = [
                a
                for a in self._items.values()
                if a.conversation_id == conversation_id and a.turn is not None
            ]
        return sorted(items, key=lambda a: (a.turn, a.created_at))

    def purge_expired(self, ttl_seconds: float) -> None:
        cutoff = time.time() - ttl_seconds
        with self._lock:
            for key in [k for k, a in self._items.items() if a.created_at < cutoff]:
                del self._items[key]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._turns.clear()


@lru_cache
def get_store() -> AttachmentStore:
    return AttachmentStore()
