"""Text recognition (OCR) for images and scanned PDF pages via RapidOCR.

RapidOCR runs PP-OCR models on ONNX Runtime; the default models (bundled in the
wheel) read Chinese and English, so no system packages or downloads are needed.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from functools import lru_cache

from PIL import Image

from ..config import get_settings

log = logging.getLogger(__name__)

ENGINE_NAME = "rapidocr"
LANGUAGES = "en,zh"

# OCR is CPU-bound; serialize it so concurrent uploads don't starve the chat turns.
_OCR_SLOTS = threading.BoundedSemaphore(1)
_ENGINE_LOCK = threading.Lock()


@dataclass
class OcrResult:
    text: str
    lines: int
    mean_confidence: float

    @property
    def chars(self) -> int:
        return len(self.text)


def engine_version() -> str:
    try:
        from importlib.metadata import version

        return version("rapidocr")
    except Exception:  # noqa: BLE001
        return "unknown"


@lru_cache(maxsize=1)
def _engine():
    from rapidocr import RapidOCR

    return RapidOCR()


def get_engine():
    with _ENGINE_LOCK:
        return _engine()


def _ordered_lines(boxes, texts, scores, min_confidence: float) -> list[tuple[str, float]]:
    """Order recognized lines top-to-bottom, then left-to-right."""
    rows = []
    for box, text, score in zip(boxes, texts, scores, strict=False):
        if score < min_confidence or not str(text).strip():
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        rows.append((min(ys), max(ys) - min(ys), min(xs), str(text).strip(), float(score)))
    rows.sort(key=lambda r: (r[0], r[2]))
    lines: list[list] = []
    for top, height, left, text, score in rows:
        # Same visual line when the tops are within half a line height.
        if lines and abs(lines[-1][0] - top) <= max(height, 1) / 2:
            lines[-1][1].append((left, text, score))
        else:
            lines.append([top, [(left, text, score)]])
    out = []
    for _, parts in lines:
        parts.sort()
        out.append(
            (
                " ".join(text for _, text, _ in parts),
                sum(score for _, _, score in parts) / len(parts),
            )
        )
    return out


def recognize(image: Image.Image) -> OcrResult:
    """Recognize text in an image. Blocking; call from a worker thread."""
    import numpy as np

    settings = get_settings()
    rgb = image.convert("RGB")
    with _OCR_SLOTS:
        result = get_engine()(np.array(rgb))
    if result is None or result.txts is None or result.boxes is None:
        return OcrResult(text="", lines=0, mean_confidence=0.0)
    lines = _ordered_lines(
        result.boxes.tolist(),
        result.txts,
        result.scores,
        settings.attachment_ocr_min_confidence,
    )
    if not lines:
        return OcrResult(text="", lines=0, mean_confidence=0.0)
    return OcrResult(
        text="\n".join(text for text, _ in lines),
        lines=len(lines),
        mean_confidence=round(sum(score for _, score in lines) / len(lines), 4),
    )
