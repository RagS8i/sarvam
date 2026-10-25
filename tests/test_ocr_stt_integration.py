"""Integration tests for OCR1 and STT_translation modules with CareFlow adapters."""
from __future__ import annotations

import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

import OCR1.organizer as organizer
from app.intake import process_document_locally
from app.pipeline.normalize import (
    InputError,
    adapt_stt,
    sources_from_ocr_pipeline,
)
from OCR1.model import SarvamVisionPipeline
from OCR1.organizer import _merge_organized_sources
from STT_translation.schemas import (
    Language,
    SpeechTranslationInput,
    SpeechTranslationRequest,
    SpeechTranslationResult,
    TranscriptChunk,
)
from STT_translation.api import app as stt_app


class _FakeResponse:
    def __init__(self, payload=None, *, content: bytes = b"", status_code: int = 200):
        self._payload = payload
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _vision_zip(payload: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("document/document.json", json.dumps(payload))
        archive.writestr("manifest.json", json.dumps({"status": "completed"}))
    return buffer.getvalue()


class _FakeSession:
    def __init__(self, archive: bytes):
        self.archive = archive
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return _FakeResponse({"job_id": "job_check"}, status_code=201)

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if url.endswith("/job/job_check/status"):
            return _FakeResponse({"status": "partially_completed"})
        if url.endswith("/job/job_check/download-url"):
            return _FakeResponse(
                {
                    "url": "https://download.example/result.zip",
                    "headers": {"X-Test": "1"},
                }
            )
        if url.startswith("https://download.example/"):
            return _FakeResponse(content=self.archive)
        raise AssertionError(f"unexpected request: {url}")


def test_vision_digitise_uses_pipeline_default_and_parses_blocks():
    """The pipeline uses the documented REST flow and omits the invalid alias."""
    archive_payload = {
        "pages": [
            {
                "blocks": [
                    {
                        "layout_tag": "paragraph",
                        "text": "Second line",
                        "reading_order": 2,
                        "coordinates": {
                            "x1": 10,
                            "y1": 20,
                            "x2": 100,
                            "y2": 50,
                        },
                    },
                    {
                        "layout_tag": "heading",
                        "text": "First line",
                        "reading_order": 1,
                    },
                ]
            }
        ]
    }
    fake = _FakeSession(_vision_zip(archive_payload))
    pipeline = SarvamVisionPipeline(model="sarvam-vision", api_key="test-key")
    pipeline._session = fake

    pages = pipeline._run_batch(
        [("page.png", b"image", "image/png")], page_offset=0
    )

    post = next(call for call in fake.calls if call[0] == "POST")
    assert post[1].endswith("/job/digitise")
    assert post[2]["data"]["output_format"] == "json"
    assert post[2]["data"]["content_type"] == "printed"
    assert "model" not in post[2]["data"]
    assert post[2]["files"] == [
        ("file", ("page.png", b"image", "image/png"))
    ]

    download = next(
        call for call in fake.calls if call[1].startswith("https://download.example/")
    )
    assert download[2]["headers"] == {"X-Test": "1"}
    assert pages == [
        {
            "page_number": 1,
            "blocks": [
                {"label": "heading", "text": "First line"},
                {
                    "label": "paragraph",
                    "text": "Second line",
                    "bbox": [10.0, 20.0, 100.0, 50.0],
                },
            ],
        }
    ]


def test_vision_digitise_forwards_explicit_model_override():
    archive_payload = {
        "pages": [{"content": "<p>Readable text</p>"}],
    }
    fake = _FakeSession(_vision_zip(archive_payload))
    pipeline = SarvamVisionPipeline(model="future-vision", api_key="test-key")
    pipeline._session = fake

    pages = pipeline._run_batch(
        [("page.png", b"image", "image/png")], page_offset=4
    )

    post = next(call for call in fake.calls if call[0] == "POST")
    assert post[2]["data"]["model"] == "future-vision"
    assert pages == [
        {
            "page_number": 5,
            "blocks": [{"label": "text", "text": "Readable text"}],
        }
    ]


def test_stt_translation_schemas():
    """Verify STT_translation schemas validate correctly."""
    req = SpeechTranslationRequest(target_language=Language.ENGLISH)
    assert req.target_language == "en-IN"

    inp = SpeechTranslationInput(filename="cough_audio.wav", target_language=Language.ENGLISH)
    assert inp.filename == "cough_audio.wav"

    chunk = TranscriptChunk(
        index=0,
        start_seconds=1.2,
        end_seconds=4.5,
        translated_text="I have had persistent chest pain and cough for three days.",
        language_code="hi",
        language_probability=0.92,
    )
    result = SpeechTranslationResult(
        translated_text="I have had persistent chest pain and cough for three days.",
        target_language="en-IN",
        chunks=[chunk],
    )
    assert result.chunks[0].translated_text == chunk.translated_text
    assert result.chunks[0].start_seconds == 1.2


def test_adapt_stt_with_stt_translation_result():
    """Verify CareFlow Stage A adapt_stt seamlessly consumes STT_translation results."""
    stt_payload = {
        "translated_text": "I have had fever and chills since yesterday.",
        "target_language": "en-IN",
        "chunks": [
            {
                "index": 0,
                "start_seconds": 0.5,
                "end_seconds": 3.8,
                "translated_text": "I have had fever and chills since yesterday.",
                "language_code": "hi",
                "language_probability": 0.91,
            }
        ],
    }

    records, notes = adapt_stt(stt_payload, case_id="case_test_stt_001")
    assert not notes
    assert len(records) == 1
    rec = records[0]
    assert rec.audio_start == 0.5
    assert rec.audio_end == 3.8
    assert rec.language == "hi"
    assert rec.confidence == 0.91
    assert "fever and chills" in rec.normalized_text


def test_sources_from_ocr_pipeline_preserves_structured_provenance():
    """Verify the strict organizer output maps directly to source records."""
    ocr_payload = {
        "document_id": "doc_vitals",
        "file_name": "patient_vitals.pdf",
        "is_medical": True,
        "document_type": "vitals_record",
        "ocr_model": "glm-ocr",
        "sources": [
            {
                "block_id": "p1_b1",
                "text": "Blood Pressure: 150/95 mmHg",
                "page": 1,
                "bbox": [50, 50, 250, 80],
                "confidence": 0.96,
                "unreadable": False,
            },
            {
                "block_id": "p1_b2",
                "text": "Pulse: 104 bpm",
                "page": 1,
                "bbox": [50, 90, 200, 120],
                "unreadable": False,
            },
        ],
    }

    records, notes = sources_from_ocr_pipeline(
        ocr_payload, case_id="case_test_ocr_001"
    )
    assert not notes
    assert len(records) == 2
    assert records[0].page == 1
    assert records[0].segment_id == "p1_b1"
    assert "Blood Pressure: 150/95 mmHg" in records[0].normalized_text
    assert records[0].confidence == 0.96
    assert records[1].page == 1
    assert "Pulse: 104 bpm" in records[1].normalized_text
    assert records[1].bbox == [50.0, 90.0, 200.0, 120.0]


def test_organizer_filters_blocks_and_preserves_original_text():
    """The model selects relevance, but text and page come from the OCR output."""
    context = {
        "file_name": "patient_lab_report.pdf",
        "pages": [
            {
                "page_number": 1,
                "blocks": [
                    {
                        "block_id": "p1_b1",
                        "text": "Fasting Blood Sugar: 145 mg/dL",
                    },
                    {
                        "block_id": "p1_b2",
                        "text": "CareFlow architecture diagram",
                    },
                ],
            }
        ],
    }
    organized = {
        "is_medical": True,
        "sources": [
            {
                "block_id": "p1_b1",
                "unreadable": False,
            }
        ],
    }

    result = _merge_organized_sources(organized, context)
    assert result["file_name"] == "patient_lab_report.pdf"
    assert result["is_medical"] is True
    assert result["sources"] == [
        {
            "block_id": "p1_b1",
            "text": "Fasting Blood Sugar: 145 mg/dL",
            "page": 1,
            "unreadable": False,
        }
    ]


def test_organizer_expands_compact_range_ids():
    """Range-compressed IDs still rebuild exact OCR text and page provenance."""
    context = {
        "file_name": "report.png",
        "pages": [
            {
                "page_number": 1,
                "blocks": [
                    {"id": 1, "block_id": "p1_b1", "text": "A"},
                    {"id": 2, "block_id": "p1_b2", "text": "B"},
                    {"id": 3, "block_id": "p1_b3", "text": "C"},
                    {"id": 4, "block_id": "p1_b4", "text": "D"},
                ],
            }
        ],
    }

    result = _merge_organized_sources(
        {
            "is_medical": True,
            "block_ids": "1-2,4",
            "unreadable_ids": "2",
        },
        context,
    )

    assert [source["block_id"] for source in result["sources"]] == [
        "p1_b1",
        "p1_b2",
        "p1_b4",
    ]
    assert result["sources"][1]["unreadable"] is True


def test_organizer_exclusion_gate_keeps_clinical_blocks_by_default():
    """The fast path keeps everything except explicit non-clinical ranges."""
    context = {
        "file_name": "report.png",
        "pages": [
            {
                "page_number": 1,
                "blocks": [
                    {"id": 1, "block_id": "p1_b1", "text": "Hospital logo"},
                    {"id": 2, "block_id": "p1_b2", "text": "BP: 146/92 mmHg"},
                    {"id": 3, "block_id": "p1_b3", "text": "Troponin-I: 0.012 ng/mL"},
                    {"id": 4, "block_id": "p1_b4", "text": "Page 1 of 1"},
                ],
            }
        ],
    }

    result = _merge_organized_sources(
        {
            "is_medical": True,
            "excluded_ids": "1,4",
            "unreadable_ids": "3",
        },
        context,
    )

    assert [source["block_id"] for source in result["sources"]] == [
        "p1_b2",
        "p1_b3",
    ]
    assert result["sources"][1]["unreadable"] is True


def test_organizer_marks_non_medical_documents_and_drops_sources():
    context = {
        "file_name": "architecture.png",
        "pages": [
            {
                "page_number": 1,
                "blocks": [
                    {
                        "block_id": "p1_b1",
                        "text": "CareFlow AI architecture",
                    }
                ],
            }
        ],
    }
    organized = {
        "is_medical": False,
        "sources": [],
    }

    result = _merge_organized_sources(organized, context)
    assert result["is_medical"] is False
    assert result["sources"] == []


def test_organizer_uses_sarvam_schema_and_guardrail_prompt(monkeypatch):
    """GLM-5.3 must receive the JSON schema and injection guardrails."""
    captured = {}

    def fake_chat_completion(**kwargs):
        captured.update(kwargs)
        return json.dumps(
            {
                "is_medical": True,
                "excluded_ids": "2",
                "unreadable_ids": "",
            }
        )

    monkeypatch.setattr(
        "OCR1.organizer.chat_completion", fake_chat_completion
    )
    ocr_result = {
        "file_name": "lab.pdf",
        "ocr_model": "sarvam-vision",
        "pages": [
            {
                "page_number": 1,
                "blocks": [
                    {
                        "label": "text",
                        "text": "Blood Pressure: 160/100 mmHg",
                    },
                    {
                        "label": "text",
                        "text": "CareFlow architecture chart",
                    },
                ],
            }
        ],
    }

    result = organizer.organize_ocr_result(
        ocr_result,
        model="glm5.3",
        base_url="https://api.sarvam.ai",
        api_key="test-key",
    )

    assert captured["model"] == "glm5.3"
    assert captured["base_url"] == "https://api.sarvam.ai"
    assert captured["api_key"] == "test-key"
    assert captured["schema"] == organizer.OUTPUT_SCHEMA
    assert captured["schema"]["properties"]["excluded_ids"] == {"type": "string"}
    assert captured["reasoning_effort"] == "low"
    assert captured["max_tokens"] <= 4096
    system_prompt = captured["messages"][0]["content"].lower()
    assert "untrusted data" in system_prompt
    assert "non_medical" in system_prompt
    assert result["ocr_model"] == "sarvam-vision"
    assert result["sources"] == [
        {
            "block_id": "p1_b1",
            "text": "Blood Pressure: 160/100 mmHg",
            "page": 1,
            "unreadable": False,
        }
    ]


def test_document_processor_reports_organizing_stage_and_timings(monkeypatch):
    """OCR completion is visible before the slower organizer finishes."""
    stages = []

    monkeypatch.setattr(
        "OCR1.model.extract_document", lambda path: {"pages": []}
    )
    monkeypatch.setattr(
        "OCR1.organizer.organize_ocr_result",
        lambda ocr_result, *, model, base_url: {"is_medical": True, "sources": []},
    )

    result = process_document_locally(
        b"image-bytes",
        "report.png",
        "qwen3:8b",
        "http://127.0.0.1:11434/api/chat",
        on_stage=stages.append,
    )

    assert stages == ["organizing"]
    assert set(result["timings"]) == {"ocr_seconds", "organizer_seconds"}


def test_stt_translation_health_and_validation():
    """Verify STT_translation API health endpoint and input validation."""
    client = TestClient(stt_app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

    # Empty audio rejection
    empty_upload = client.post(
        "/speech-to-language",
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )
    assert empty_upload.status_code == 400

def test_adapters_reject_payloads_with_no_usable_text():
    """All-empty OCR/STT payloads must fail rather than enqueue an empty case."""
    with pytest.raises(InputError) as stt_error:
        adapt_stt(
            {"transcript_id": "empty-stt", "segments": [{"text": "   "}]},
            case_id="case_empty_stt",
        )
    assert stt_error.value.code == "empty_transcript"

    with pytest.raises(InputError) as ocr_error:
        sources_from_ocr_pipeline(
            {
                "document_id": "empty-ocr",
                "is_medical": True,
                "document_type": "other_medical",
                "sources": [{"text": "\n\t", "page": 1}],
            },
            case_id="case_empty_ocr",
        )
    assert ocr_error.value.code == "empty_medical_document"

    with pytest.raises(InputError) as non_medical_error:
        sources_from_ocr_pipeline(
            {
                "document_id": "non-medical",
                "is_medical": False,
                "document_type": "non_medical",
                "sources": [],
            },
            case_id="case_non_medical",
        )
    assert non_medical_error.value.code == "non_medical_document"


def test_adapt_stt_preserves_original_hindi_and_hinglish_provenance():
    """Regression test: Hindi/Hinglish transcription survives in original_text, not replaced by translation."""
    hindi_payload = {
        "transcription": "मुझे दो दिन से चक्कर आ रहे हैं और बहुत कमजोरी लग रही है",
        "translated_text": "I have been feeling dizzy and very weak for two days",
        "target_language": "en-IN",
        "document_id": "doc_hindi_audio_001",
        "chunks": [
            {
                "index": 0,
                "start_seconds": 1.0,
                "end_seconds": 4.5,
                "transcription": "मुझे दो दिन से चक्कर आ रहे हैं और बहुत कमजोरी लग रही है",
                "translated_text": "I have been feeling dizzy and very weak for two days",
                "language_code": "hi",
                "language_probability": 0.98,
            }
        ],
    }

    records, notes = adapt_stt(hindi_payload, case_id="case_hindi_001")
    assert not notes
    assert len(records) == 1
    rec = records[0]
    # PROVENANCE CHECK: original_text MUST be the Hindi transcription!
    assert "मुझे दो दिन से चक्कर आ रहे हैं" in rec.original_text
    assert "feeling dizzy" not in rec.original_text
    # English translation preserved in raw_metadata
    assert rec.raw_metadata.get("translated_text") == "I have been feeling dizzy and very weak for two days"
    assert rec.document_ref == "doc_hindi_audio_001"
    assert rec.audio_start == 1.0
    assert rec.audio_end == 4.5
    assert rec.language == "hi"

    # Hinglish provenance test
    hinglish_payload = {
        "transcription": "Mujhe two days se sudden chest pain ho raha hai after walking",
        "translated_text": "I have sudden chest pain for two days after walking",
        "target_language": "en-IN",
        "document_id": "doc_hinglish_001",
        "chunks": [
            {
                "index": 0,
                "start_seconds": 0.0,
                "end_seconds": 3.8,
                "transcription": "Mujhe two days se sudden chest pain ho raha hai after walking",
                "translated_text": "I have sudden chest pain for two days after walking",
                "language_code": "hinglish",
                "language_probability": 0.92,
            }
        ],
    }
    records, notes = adapt_stt(hinglish_payload, case_id="case_hinglish_001")
    assert len(records) == 1
    h_rec = records[0]
    assert "Mujhe two days se sudden chest pain" in h_rec.original_text
    assert h_rec.document_ref == "doc_hinglish_001"
    assert h_rec.language == "hinglish"
