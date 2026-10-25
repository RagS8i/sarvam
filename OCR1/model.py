"""Sarvam Vision document digitisation for CareFlow AI.

Sarvam Vision accepts PDFs, PNG/JPG images and ZIP archives, with a maximum of
10 pages per job.  This module keeps the same internal contract the organizer
already consumes: one normalized page with text and labeled blocks plus page
provenance.  Large PDFs are rendered page-by-page and sent in batches of ten.

The module is imported lazily by intake so the web app can start without a
Sarvam API key unless document intake is used.
"""
from __future__ import annotations

import html
import io
import json
import os
import re
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

import requests

__all__ = ["SUPPORTED_EXTENSIONS", "SarvamVisionError", "extract_document", "get_ocr_pipeline"]

DEFAULT_OCR_MODEL = "sarvam-vision"
DEFAULT_SARVAM_BASE_URL = "https://api.sarvam.ai"
DEFAULT_OCR_TIMEOUT_S = 300.0
DEFAULT_OCR_POLL_S = 2.0
DEFAULT_OCR_REQUEST_TIMEOUT_S = 120.0
MAX_PAGES_PER_JOB = 10
MAX_VISION_BYTES = 200 * 1024 * 1024

SUPPORTED_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    }
)

_TERMINAL_FAILURE = {"failed", "error", "cancelled", "canceled", "rejected"}
_TERMINAL_SUCCESS = {
    "completed",
    "complete",
    "partially_completed",
    "succeeded",
    "success",
    "done",
    "processed",
}


class SarvamVisionError(RuntimeError):
    """A safe Sarvam Vision failure that never contains patient text."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _strip_fences(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _parse_json(value: str) -> Any | None:
    cleaned = _strip_fences(value)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Vision sometimes wraps a JSON page in a short explanation.
        decoder = json.JSONDecoder()
        for index, character in enumerate(cleaned):
            if character not in "[{":
                continue
            try:
                candidate, _ = decoder.raw_decode(cleaned[index:])
            except json.JSONDecodeError:
                continue
            return candidate
    return None


def _html_to_text(value: Any) -> str:
    """Flatten HTML/Markdown-ish markup into readable plain text."""
    if not isinstance(value, str) or not value.strip():
        return ""
    text = re.sub(
        r"(?i)<br\s*/?>|</(?:p|div|li|tr|h[1-6]|table|pre|blockquote)>",
        "\n",
        value,
    )
    text = re.sub(r"(?i)</t[dh]>", "\t", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("\xa0", " ")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _text_blocks(text: str) -> list[dict[str, Any]]:
    return [
        {"label": "text", "text": line.strip()}
        for line in (text or "").splitlines()
        if line.strip()
    ]


def _table_to_text(table: Any) -> str:
    if not isinstance(table, list):
        return ""
    rows: list[str] = []
    for row in table:
        if isinstance(row, dict):
            cells = [str(value) for value in row.values()]
        elif isinstance(row, list):
            cells = [str(value) for value in row]
        else:
            cells = [str(row)]
        if any(cell.strip() for cell in cells):
            rows.append("\t".join(cells))
    return "\n".join(rows)


def _normalize_bbox(value: Any) -> Optional[list[float]]:
    if value is None:
        return None
    if isinstance(value, dict):
        try:
            if {"x0", "y0", "x1", "y1"} <= set(value):
                return [float(value["x0"]), float(value["y0"]), float(value["x1"]), float(value["y1"])]
            if {"x1", "y1", "x2", "y2"} <= set(value):
                return [float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"])]
            if {"x", "y", "width", "height"} <= set(value):
                x, y = float(value["x"]), float(value["y"])
                return [x, y, x + float(value["width"]), y + float(value["height"])]
        except (TypeError, ValueError):
            return None
        return None
    if isinstance(value, (list, tuple)):
        try:
            numbers = [float(item) for item in value]
        except (TypeError, ValueError):
            return None
        if len(numbers) == 4:
            return numbers
        if len(numbers) >= 8 and len(numbers) % 2 == 0:
            xs = numbers[0::2]
            ys = numbers[1::2]
            return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _normalize_block(
    value: Any,
    *,
    default_label: str = "text",
) -> Optional[dict[str, Any]]:
    if isinstance(value, str):
        text = _html_to_text(value)
        return {"label": default_label, "text": text} if text else None
    if not isinstance(value, dict):
        return None

    label = str(
        value.get("label")
        or value.get("layout_tag")
        or value.get("type")
        or value.get("tag")
        or value.get("block_label")
        or value.get("block_type")
        or default_label
    ).strip().lower() or default_label

    text_value = (
        value.get("text")
        or value.get("block_content")
        or value.get("content")
        or value.get("value")
        or value.get("markdown")
        or _table_to_text(value.get("table"))
        or ""
    )
    text = _html_to_text(text_value) if isinstance(text_value, str) else ""
    if not text:
        return None

    block: dict[str, Any] = {"label": label, "text": text}
    bbox = _normalize_bbox(
        value.get("bbox")
        or value.get("box")
        or value.get("bounding_box")
        or value.get("coordinates")
    )
    if bbox is not None:
        block["bbox"] = bbox
    confidence = value.get("confidence", value.get("score"))
    if confidence is not None:
        try:
            block["confidence"] = float(confidence)
        except (TypeError, ValueError):
            pass
    return block


_CHILD_KEYS = (
    "blocks",
    "children",
    "elements",
    "lines",
    "paragraphs",
    "cells",
    "items",
    "regions",
)


def _blocks_from_node(
    node: Any,
    *,
    default_label: str = "text",
    output: list[dict[str, Any]],
) -> None:
    if node is None:
        return
    if isinstance(node, str):
        block = _normalize_block(node, default_label=default_label)
        if block:
            output.append(block)
        return
    if isinstance(node, list):
        for item in node:
            _blocks_from_node(item, default_label=default_label, output=output)
        return
    if not isinstance(node, dict):
        return

    child_blocks: list[dict[str, Any]] = []
    for key in _CHILD_KEYS:
        child = node.get(key)
        if isinstance(child, (list, dict)):
            _blocks_from_node(
                child,
                default_label=default_label,
                output=child_blocks,
            )
    if child_blocks:
        output.extend(child_blocks)
        return

    block = _normalize_block(node, default_label=default_label)
    if block:
        output.append(block)


def _page_payload(payload: Any) -> tuple[str, list[dict[str, Any]]]:
    """Normalize one Vision page's JSON, HTML or markdown content."""
    if isinstance(payload, str):
        cleaned = _strip_fences(payload)
        parsed = _parse_json(cleaned)
        if parsed is None:
            text = _html_to_text(cleaned)
            return text, _text_blocks(text)
        payload = parsed

    blocks: list[dict[str, Any]] = []
    _blocks_from_node(payload, output=blocks)

    text = ""
    if isinstance(payload, dict):
        raw_text = (
            payload.get("text")
            or payload.get("markdown")
            or payload.get("content")
            or payload.get("page_text")
            or ""
        )
        if isinstance(raw_text, str):
            text = _html_to_text(raw_text)

    if not text and blocks:
        text = "\n".join(block["text"] for block in blocks)
    if not blocks and text:
        blocks = _text_blocks(text)

    # Drop exact duplicate blocks while preserving reading order.
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for block in blocks:
        key = (block["label"], block["text"])
        if key not in seen:
            seen.add(key)
            deduped.append(block)
    return text, deduped


def _page_blocks(page: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the block list returned by Document AI digitise results.

    The JSON result normally nests text under ``page["blocks"]``.  Older
    responses and HTML/markdown results instead expose a ``content`` string, so
    both shapes are supported.
    """
    raw_blocks = page.get("blocks")
    if isinstance(raw_blocks, list) and raw_blocks:
        ordered = [block for block in raw_blocks if isinstance(block, dict)]
        ordered.sort(
            key=lambda block: (
                not isinstance(block.get("reading_order"), (int, float)),
                block.get("reading_order")
                if isinstance(block.get("reading_order"), (int, float))
                else 0,
            )
        )
        blocks: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for raw_block in ordered:
            block = _normalize_block(raw_block)
            if block is None:
                continue
            key = (block["label"], block["text"])
            if key in seen:
                continue
            seen.add(key)
            blocks.append(block)
        if blocks:
            return blocks

    content = page.get("content") or page.get("text") or page.get("markdown")
    _text, blocks = _page_payload(content)
    return blocks


class SarvamVisionPipeline:
    """Sarvam Vision job runner with 10-page batching and safe polling.

    Uses the documented Document AI REST lifecycle directly:

    ``POST /job/digitise`` → poll ``GET /job/{id}/status`` → mint a download
    URL → download and parse the result ZIP.  The JSON output keeps individual
    layout blocks, so CareFlow can preserve reading order and bounding boxes for
    provenance.
    """

    def __init__(
        self,
        model: str = DEFAULT_OCR_MODEL,
        timeout_s: float = DEFAULT_OCR_TIMEOUT_S,
        poll_s: float = DEFAULT_OCR_POLL_S,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        language: Optional[str] = None,
        content_type: Optional[str] = None,
        request_timeout_s: float = DEFAULT_OCR_REQUEST_TIMEOUT_S,
    ):
        self.model = model
        self.timeout_s = timeout_s
        self.poll_s = max(0.2, poll_s)
        self.request_timeout_s = max(1.0, request_timeout_s)
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        configured_base = (
            base_url
            or os.getenv("CAREFLOW_SARVAM_BASE_URL", DEFAULT_SARVAM_BASE_URL)
        ).rstrip("/")
        self.base_url = (
            configured_base
            if configured_base.endswith("/doc-ai/v1")
            else f"{configured_base}/doc-ai/v1"
        )
        self.language = (
            language
            if language is not None
            else os.getenv("CAREFLOW_OCR_LANGUAGE", "")
        ).strip()
        configured_content_type = (
            content_type
            if content_type is not None
            else os.getenv("CAREFLOW_OCR_CONTENT_TYPE", "printed")
        ).strip().lower()
        self.content_type = (
            configured_content_type
            if configured_content_type in {"printed", "handwritten", "mixed"}
            else "printed"
        )
        self._session = requests.Session()

    def _api_headers(self) -> dict[str, str]:
        if not self.api_key or not self.api_key.strip():
            raise SarvamVisionError(
                "not_configured",
                "SARVAM_API_KEY is not configured for document OCR.",
            )
        return {"api-subscription-key": self.api_key.strip()}

    def _wait_for_job(self, job_id: str) -> None:
        deadline = time.monotonic() + self.timeout_s
        status_url = f"{self.base_url}/job/{job_id}/status"
        while True:
            try:
                response = self._session.get(
                    status_url,
                    headers=self._api_headers(),
                    timeout=min(30.0, self.request_timeout_s),
                )
                response.raise_for_status()
                status_payload = response.json()
            except Exception as exc:
                raise SarvamVisionError(
                    "status_failed", "Sarvam Vision status check failed."
                ) from exc
            if not isinstance(status_payload, dict):
                raise SarvamVisionError(
                    "status_failed", "Sarvam Vision returned an invalid job status."
                )
            status = str(status_payload.get("status") or "").strip().lower()
            if status in _TERMINAL_SUCCESS:
                return
            if status in _TERMINAL_FAILURE:
                raise SarvamVisionError(
                    "job_failed", "Sarvam Vision could not process the document."
                )
            if time.monotonic() >= deadline:
                raise SarvamVisionError(
                    "timeout", "Sarvam Vision timed out while processing the document."
                )
            time.sleep(self.poll_s)

    @staticmethod
    def _archive_documents(content: bytes) -> list[dict[str, Any]]:
        """Read document JSON payloads from the downloaded ZIP."""
        if not content:
            raise SarvamVisionError(
                "empty_results", "Sarvam Vision returned no OCR output."
            )
        if len(content) > MAX_VISION_BYTES:
            raise SarvamVisionError(
                "too_large", "The OCR result exceeds CareFlow's size limit."
            )

        try:
            archive = zipfile.ZipFile(io.BytesIO(content))
        except zipfile.BadZipFile:
            # Be tolerant of a JSON body from a future API revision.
            try:
                direct = json.loads(content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SarvamVisionError(
                    "malformed_results", "Sarvam Vision returned an invalid result."
                ) from exc
            return [direct] if isinstance(direct, dict) else []

        documents: list[dict[str, Any]] = []
        total_uncompressed = 0
        try:
            with archive:
                for info in archive.infolist():
                    total_uncompressed += max(0, info.file_size)
                    if total_uncompressed > MAX_VISION_BYTES:
                        raise SarvamVisionError(
                            "too_large",
                            "The OCR result exceeds CareFlow's size limit.",
                        )
                    name = info.filename
                    lower_name = name.lower()
                    parts = {part.lower() for part in name.split("/")}
                    if lower_name.endswith("manifest.json"):
                        continue
                    if "metadata" in parts:
                        continue
                    if not lower_name.endswith(".json"):
                        continue
                    try:
                        payload = json.loads(archive.read(info).decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if isinstance(payload, dict):
                        documents.append(payload)
        except SarvamVisionError:
            raise
        except Exception as exc:
            raise SarvamVisionError(
                "malformed_results", "Sarvam Vision returned an invalid result."
            ) from exc
        return documents

    def _parse_archive(
        self, content: bytes, *, page_offset: int
    ) -> list[dict[str, Any]]:
        documents = self._archive_documents(content)
        raw_pages: list[dict[str, Any]] = []
        for document in documents:
            document_pages = document.get("pages")
            if not isinstance(document_pages, list):
                continue
            raw_pages.extend(
                page for page in document_pages if isinstance(page, dict)
            )

        pages: list[dict[str, Any]] = []
        for raw_page in raw_pages:
            page_blocks = _page_blocks(raw_page)
            if not page_blocks:
                continue
            pages.append(
                {
                    "page_number": page_offset + len(pages) + 1,
                    "blocks": page_blocks,
                }
            )

        if not pages:
            raise SarvamVisionError(
                "empty_results", "Sarvam Vision returned no readable pages."
            )
        return pages

    def _download_output(self, job_id: str) -> bytes:
        link_url = f"{self.base_url}/job/{job_id}/download-url"
        try:
            link_response = self._session.get(
                link_url,
                headers=self._api_headers(),
                timeout=min(30.0, self.request_timeout_s),
            )
            link_response.raise_for_status()
            link_payload = link_response.json()
        except Exception as exc:
            raise SarvamVisionError(
                "results_failed", "Sarvam Vision results could not be retrieved."
            ) from exc
        if not isinstance(link_payload, dict):
            raise SarvamVisionError(
                "results_failed", "Sarvam Vision returned an invalid download link."
            )

        download_url = str(link_payload.get("url") or "").strip()
        if not download_url.startswith(("https://", "http://")):
            raise SarvamVisionError(
                "results_failed", "Sarvam Vision returned an invalid download link."
            )
        download_url = urljoin("https://api.sarvam.ai/", download_url)
        download_headers = link_payload.get("headers") or {}
        if not isinstance(download_headers, dict):
            download_headers = {}
        try:
            output = self._session.get(
                download_url,
                headers={str(k): str(v) for k, v in download_headers.items()},
                timeout=max(300.0, self.request_timeout_s),
            )
            output.raise_for_status()
        except Exception as exc:
            raise SarvamVisionError(
                "results_failed", "Sarvam Vision results could not be downloaded."
            ) from exc
        return output.content

    def _run_batch(
        self,
        files: list[tuple[str, bytes, str]],
        *,
        page_offset: int,
    ) -> list[dict[str, Any]]:
        total_bytes = sum(len(content) for _name, content, _mime in files)
        if total_bytes > MAX_VISION_BYTES:
            raise SarvamVisionError(
                "too_large", "The document exceeds Sarvam Vision's 200 MB limit."
            )

        data: dict[str, str] = {
            "output_format": "json",
            "content_type": self.content_type,
            "auto_orient": "true",
        }
        if self.language:
            data["language"] = self.language
        # The public model name ``sarvam-vision`` is not accepted by the
        # digitise pipeline's model override.  Omitting ``model`` lets Sarvam
        # select the pipeline's current default.  An explicit, non-default
        # override is still forwarded for future pipeline models.
        model = (self.model or "").strip()
        if model and model.lower() not in {"auto", "default", "sarvam-vision"}:
            data["model"] = model

        upload_files = [
            ("file", (name, content, mime)) for name, content, mime in files
        ]
        try:
            started = self._session.post(
                f"{self.base_url}/job/digitise",
                headers=self._api_headers(),
                files=upload_files,
                data=data,
                timeout=self.request_timeout_s,
            )
            started.raise_for_status()
            started_payload = started.json()
        except Exception as exc:
            raise SarvamVisionError(
                "start_failed", "Sarvam Vision could not start the OCR job."
            ) from exc

        job_id = (
            started_payload.get("job_id")
            if isinstance(started_payload, dict)
            else None
        )
        if not job_id:
            raise SarvamVisionError(
                "start_failed", "Sarvam Vision did not return a job identifier."
            )
        self._wait_for_job(str(job_id))

        content = self._download_output(str(job_id))
        return self._parse_archive(content, page_offset=page_offset)

    def _render_pdf_batch(self, path: Path, start: int, count: int) -> list[tuple[str, bytes, str]]:
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise SarvamVisionError(
                "dependency_missing",
                "PDF rendering requires pypdfium2.",
            ) from exc

        files: list[tuple[str, bytes, str]] = []
        with pdfium.PdfDocument(str(path)) as pdf:
            for page_index in range(start, min(start + count, len(pdf))):
                page = pdf[page_index]
                bitmap = page.render(scale=2)
                image = bitmap.to_pil().convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                files.append(
                    (
                        f"page_{page_index + 1:04d}.png",
                        buffer.getvalue(),
                        "image/png",
                    )
                )
        return files

    def process(self, path: Path) -> list[dict[str, Any]]:
        suffix = path.suffix.lower()
        pages: list[dict[str, Any]] = []

        if suffix == ".pdf":
            try:
                import pypdfium2 as pdfium
            except ImportError as exc:
                raise SarvamVisionError(
                    "dependency_missing",
                    "PDF support requires pypdfium2.",
                ) from exc
            try:
                with pdfium.PdfDocument(str(path)) as pdf:
                    page_count = len(pdf)
            except Exception as exc:
                raise SarvamVisionError(
                    "invalid_document", "The PDF could not be opened."
                ) from exc
            if page_count < 1:
                raise SarvamVisionError(
                    "empty_document", "The document contains no pages."
                )
            if page_count <= MAX_PAGES_PER_JOB:
                # A small PDF can be sent as-is, preserving its native layout.
                pages.extend(
                    self._run_batch(
                        [(path.name, path.read_bytes(), "application/pdf")],
                        page_offset=0,
                    )
                )
            else:
                for start in range(0, page_count, MAX_PAGES_PER_JOB):
                    batch = self._render_pdf_batch(path, start, MAX_PAGES_PER_JOB)
                    pages.extend(self._run_batch(batch, page_offset=start))
            return pages

        image_bytes = _image_to_png_bytes(path)
        return self._run_batch(
            [(f"{path.stem}.png", image_bytes, "image/png")],
            page_offset=0,
        )


def _image_to_png_bytes(path: Path) -> bytes:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SarvamVisionError(
            "dependency_missing", "Pillow is required to load OCR images."
        ) from exc
    try:
        with Image.open(path) as image:
            converted = image.convert("RGB")
            buffer = io.BytesIO()
            converted.save(buffer, format="PNG")
            return buffer.getvalue()
    except Exception as exc:
        raise SarvamVisionError(
            "invalid_document", "The image could not be opened."
        ) from exc


@lru_cache(maxsize=1)
def get_ocr_pipeline() -> SarvamVisionPipeline:
    """Initialize the Sarvam Vision client configuration once and reuse it."""
    model = os.getenv("CAREFLOW_OCR_MODEL", DEFAULT_OCR_MODEL).strip()
    return SarvamVisionPipeline(
        model=model or DEFAULT_OCR_MODEL,
        timeout_s=_env_float("CAREFLOW_OCR_TIMEOUT_S", DEFAULT_OCR_TIMEOUT_S),
        poll_s=_env_float("CAREFLOW_OCR_POLL_S", DEFAULT_OCR_POLL_S),
        base_url=os.getenv("CAREFLOW_SARVAM_BASE_URL", DEFAULT_SARVAM_BASE_URL),
        language=os.getenv("CAREFLOW_OCR_LANGUAGE", ""),
        content_type=os.getenv("CAREFLOW_OCR_CONTENT_TYPE", "printed"),
        request_timeout_s=_env_float(
            "CAREFLOW_OCR_REQUEST_TIMEOUT_S", DEFAULT_OCR_REQUEST_TIMEOUT_S
        ),
    )


def extract_document(file_path: str | Path) -> dict[str, Any]:
    """Extract text blocks from an image or PDF using Sarvam Vision.

    Each PDF page produces one normalized page.  PDFs over ten pages are
    rendered and submitted in compliant batches; the page numbers remain
    global to the original document.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. Provide a supported image or PDF."
        )
    if path.stat().st_size > MAX_VISION_BYTES:
        raise SarvamVisionError(
            "too_large", "The document exceeds Sarvam Vision's 200 MB limit."
        )

    pipeline = get_ocr_pipeline()
    pages = pipeline.process(path)
    if not pages:
        raise SarvamVisionError("empty_document", "No pages were extracted.")
    return {
        "file_name": path.name,
        "ocr_model": pipeline.model,
        "page_count": len(pages),
        "pages": [
            {
                "page_number": page.get("page_number", index),
                "blocks": page.get("blocks", []),
            }
            for index, page in enumerate(pages, start=1)
        ],
    }
