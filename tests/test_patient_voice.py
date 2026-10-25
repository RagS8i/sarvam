from __future__ import annotations

import base64

from .conftest import (
    CLINICIAN,
    INTAKE,
    consent,
    detail,
    ingest_message,
    new_case,
    patient_headers,
    process,
)


def _processed_case(client, patient_id: str) -> str:
    case_id = new_case(client, CLINICIAN, patient_id=patient_id)
    consent(client, case_id, INTAKE)
    response = ingest_message(
        client,
        case_id,
        "I have had a mild headache for two days and I take metformin.",
        headers=CLINICIAN,
    )
    assert response.status_code == 200, response.text
    assert process(client, case_id).status_code == 200
    return case_id


def _sent_followup(client, case_id: str) -> str:
    data = detail(client, case_id)
    question = next(
        item
        for item in data["followups"]
        if item["status"] == "suggested" and item["language"] == "en"
    )
    followup_id = question["followup_id"]
    assert (
        client.post(
            f"/api/cases/{case_id}/followups/{followup_id}/approve",
            headers=CLINICIAN,
            json={"language": "en"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"/api/cases/{case_id}/followups/{followup_id}/send",
            headers=CLINICIAN,
        ).status_code
        == 200
    )
    return followup_id


def test_patient_detail_hides_clinician_only_content(client):
    case_id = _processed_case(client, "patient_privacy")
    staff_view = detail(client, case_id)
    assert staff_view["summary"] is not None
    assert staff_view["facts"]
    assert staff_view["sources"]

    patient = patient_headers("patient_privacy")
    response = client.get(f"/api/cases/{case_id}", headers=patient)
    assert response.status_code == 200, response.text
    data = response.json()
    for key in (
        "sources",
        "facts",
        "warnings",
        "discrepancies",
        "reviews",
        "processing_runs",
        "documents",
        "escalations",
    ):
        assert data[key] == []
    assert data["summary"] is None
    assert data["note"] is None
    assert data["queue_estimate"] is None
    assert data["rule_config"] == {}
    assert data["case"]["case_id"] == case_id

    assert (
        client.get(
            f"/api/cases/{case_id}", headers=patient_headers("someone_else")
        ).status_code
        == 403
    )

    card = client.get(f"/api/cases/{case_id}/patient-card", headers=patient)
    assert card.status_code == 200, card.text
    body = card.json()
    assert "Your CareFlow care summary" in body["text"]
    assert "What happens next" in body["text"]
    assert "draft clinician" not in body["text"].lower()
    assert "metformin metformin" not in body["text"].lower()
    assert "You reported taking metformin." in body["text"]


def test_patient_context_card_translates_and_speaks(
    client_factory, monkeypatch
):
    client = client_factory(sarvam_api_key="test-key")
    case_id = _processed_case(client, "patient_voice_card")
    calls = {}

    def fake_translate(text, target_language, *, source_language="auto"):
        calls["translate"] = {
            "target_language": target_language,
            "source_language": source_language,
            "chars": len(text),
        }
        return {
            "translated_text": "यह आपकी देखभाल का सारांश है।",
            "target_language": target_language,
            "request_ids": ["req_translate"],
        }

    def fake_tts(text, language_code, *, speaker=None, model=None):
        calls["tts"] = {
            "language_code": language_code,
            "speaker": speaker,
            "model": model,
            "chars": len(text),
        }
        return {
            "audio_bytes": b"ID3fake-audio",
            "mime": "audio/mpeg",
            "language_code": language_code,
            "request_ids": ["req_tts"],
            "chunks": 1,
        }

    monkeypatch.setattr("STT_translation.stt.translate_text", fake_translate)
    monkeypatch.setattr("STT_translation.stt.text_to_speech", fake_tts)

    response = client.post(
        f"/api/cases/{case_id}/patient-voice/context-card",
        headers=patient_headers("patient_voice_card"),
        json={"language": "hi-IN", "include_audio": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["language"] == "hi-IN"
    assert body["translated"] is True
    assert body["text"] == "यह आपकी देखभाल का सारांश है।"
    assert base64.b64decode(body["audio_base64"]) == b"ID3fake-audio"
    assert body["audio_mime"] == "audio/mpeg"
    assert body["tts_supported"] is True
    assert calls["translate"]["target_language"] == "hi-IN"
    assert calls["translate"]["source_language"] == "en-IN"
    assert calls["translate"]["chars"] > 0
    assert calls["tts"]["language_code"] == "hi-IN"
    assert calls["tts"]["speaker"] == "shubh"


def test_patient_can_localize_followup(client_factory, monkeypatch):
    client = client_factory(sarvam_api_key="test-key")
    case_id = _processed_case(client, "patient_voice_followup")
    followup_id = _sent_followup(client, case_id)
    patient = patient_headers("patient_voice_followup")

    def fake_translate(text, target_language, *, source_language="auto"):
        return {
            "translated_text": "क्या आपको कोई और लक्षण है?",
            "target_language": target_language,
            "request_ids": [],
        }

    def fake_tts(text, language_code, *, speaker=None, model=None):
        return {
            "audio_bytes": b"ID3question",
            "mime": "audio/mpeg",
            "language_code": language_code,
            "request_ids": [],
            "chunks": 1,
        }

    monkeypatch.setattr("STT_translation.stt.translate_text", fake_translate)
    monkeypatch.setattr("STT_translation.stt.text_to_speech", fake_tts)

    response = client.post(
        f"/api/cases/{case_id}/patient-voice/followups/{followup_id}",
        headers=patient,
        json={"language": "hi-IN", "include_audio": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["followup_id"] == followup_id
    assert body["translated"] is True
    assert body["text"] == "क्या आपको कोई और लक्षण है?"
    assert base64.b64decode(body["audio_base64"]) == b"ID3question"


def test_patient_audio_answer_preserves_native_and_english(
    client_factory, monkeypatch
):
    client = client_factory(sarvam_api_key="test-key")
    case_id = _processed_case(client, "patient_voice_answer")
    followup_id = _sent_followup(client, case_id)
    patient = patient_headers("patient_voice_answer")

    stt_result = {
        "transcription": "मुझे दो दिनों से हल्का सिरदर्द है",
        "translated_text": "I have had a mild headache for two days",
        "target_language": "en-IN",
        "language": "hi-IN",
        "language_code": "hi-IN",
        "language_probability": 0.97,
        "model": "saaras:v4",
        "chunks": [
            {
                "index": 0,
                "start_seconds": 0.0,
                "end_seconds": 2.0,
                "transcription": "मुझे दो दिनों से हल्का सिरदर्द है",
                "translated_text": "I have had a mild headache for two days",
                "language_code": "hi-IN",
                "language_probability": 0.97,
            }
        ],
    }
    monkeypatch.setattr(
        "STT_translation.stt.speech_to_language",
        lambda **kwargs: stt_result,
    )

    response = client.post(
        f"/api/cases/{case_id}/followups/{followup_id}/answers/audio",
        headers=patient,
        files={"file": ("answer.wav", b"not-real-but-not-decoded", "audio/wav")},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["answer"]["text"] == stt_result["transcription"]
    assert body["answer"]["translated_text"] == stt_result["translated_text"]
    assert body["answer"]["document_id"]
    assert body["answer"]["metadata"]["answer_modality"] == "audio"
    assert body["followup"]["status"] == "answered"
    assert body["detected_language"] == "hi-IN"

    patient_view = client.get(f"/api/cases/{case_id}", headers=patient).json()
    answer = next(
        item
        for item in patient_view["followup_answers"]
        if item["answer_id"] == body["answer"]["answer_id"]
    )
    assert answer["text"] == stt_result["transcription"]
    assert answer["translated_text"] == stt_result["translated_text"]
    assert answer["document_id"] == body["answer"]["document_id"]
