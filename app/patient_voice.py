"""Patient-facing context cards, translated voice playback and voice answers.

The clinician pipeline remains English-oriented.  This module builds a
deterministic patient-facing card from evidence-linked facts (never the draft
clinician note), translates it on demand, and speaks it with Sarvam Bulbul.
Follow-up questions can be localized the same way, and audio answers preserve
the native transcript alongside an English translation and the stored recording.
"""
from __future__ import annotations

import base64
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from STT_translation.languages import (
    SARVAM_TRANSLATE_LANGUAGES,
    language_name,
    normalize_bcp47,
    short_code,
    tts_supported,
)

from .config import Settings
from .db import Database
from .followups import FollowUpError, FollowUpService, normalize_followup_language
from .schemas import (
    Fact,
    FactStatus,
    FactType,
    FollowUpStatus,
    SourceRecord,
)
from .storage import Cipher, DocumentStore, random_id, safe_error, sha256_text

_DISCLAIMER = (
    "This is a plain-language summary of information you shared. It is not a "
    "diagnosis and does not confirm that it is safe to wait. If your symptoms "
    "get worse or you feel unsafe, seek urgent medical care."
)

_NEXT_STEPS = [
    "A clinician will review the information you shared.",
    "Your care team may ask you follow-up questions. You can answer by text or voice.",
    "This summary does not replace a clinical assessment or emergency care.",
]


class PatientVoiceError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 502):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _measurement(fact: Fact) -> str:
    value = _clean(fact.value_normalized or fact.value)
    if not value:
        return ""
    unit = _clean(fact.unit)
    return f"{value} {unit}".strip()


def _fact_phrase(fact: Fact) -> Optional[str]:
    label = _clean(fact.label)
    if not label:
        return None
    status = fact.status
    if fact.fact_type == FactType.medication:
        dose = _measurement(fact)
        if dose.lower() == label.lower():
            dose = ""
        elif dose.lower().startswith(label.lower()):
            dose = dose[len(label) :].strip(" ,;-")
        medicine = f"{label} {dose}".strip()
        if status == FactStatus.present:
            return f"You reported taking {medicine}."
        if status == FactStatus.historical:
            return f"You previously took {medicine}."
        if status == FactStatus.absent:
            return f"You reported not taking {medicine}."
        if status == FactStatus.uncertain:
            return f"You were unsure whether you take {medicine}."
        return None
    if fact.fact_type in {FactType.vital_sign, FactType.measurement}:
        measured = _measurement(fact)
        return f"{label}: {measured}." if measured else f"{label} was recorded."
    if status == FactStatus.absent:
        return f"You reported no {label}."
    if status == FactStatus.historical:
        return f"In your history: {label}."
    if status == FactStatus.uncertain:
        return f"You were unsure about {label}."
    if status == FactStatus.present:
        return f"You reported {label}."
    return None


def build_patient_context_card(
    summary: Any = None,
    facts: Optional[List[Fact]] = None,
) -> Dict[str, Any]:
    """Build the deterministic English patient card.

    Only clinical fact types are included.  Clinician notes, warnings,
    discrepancies, urgency, escalation data and draft AI narratives are
    deliberately excluded.
    """
    facts = facts or []
    sections: List[Dict[str, Any]] = []
    used: set[str] = set()

    def add_section(key: str, title: str, items: List[str]) -> None:
        cleaned: List[str] = []
        for item in items:
            item = _clean(item)
            if item and item not in used:
                used.add(item)
                cleaned.append(item)
        if cleaned:
            sections.append({"key": key, "title": title, "items": cleaned})

    main_concern: List[str] = []
    if summary is not None:
        presenting = getattr(summary, "presenting_complaint", None)
        if presenting is not None:
            main_concern = [
                item
                for item in (presenting.items or [])
                if _clean(item)
                and "not clearly stated" not in str(item).lower()
            ]
    add_section("main_concern", "Your main concern", main_concern)

    symptoms = [
        phrase
        for fact in facts
        if fact.fact_type in {FactType.symptom, FactType.explicit_negation}
        for phrase in [_fact_phrase(fact)]
        if phrase
    ]
    add_section("symptoms", "Your symptoms", symptoms)

    medicines = [
        phrase
        for fact in facts
        if fact.fact_type in {FactType.medication, FactType.medication_history}
        for phrase in [_fact_phrase(fact)]
        if phrase
    ]
    allergies = [
        f"Allergy: {_clean(fact.value or fact.label)}."
        for fact in facts
        if str(fact.label or "").lower().startswith("allergy:")
        or str(fact.normalized_text or "").lower().startswith("allergy")
    ]
    add_section("medicines", "Medicines", medicines)
    add_section("allergies", "Allergies", allergies)

    history = [
        phrase
        for fact in facts
        if fact.fact_type == FactType.history
        and not str(fact.label or "").lower().startswith("allergy:")
        for phrase in [_fact_phrase(fact)]
        if phrase
    ]
    add_section("history", "Health history", history)

    timeline = []
    for fact in facts:
        if fact.fact_type not in {
            FactType.onset,
            FactType.duration,
            FactType.progression,
        }:
            continue
        detail = _clean(fact.value_normalized or fact.value or fact.normalized_text)
        label = _clean(fact.label)
        if detail and label and detail.lower() not in label.lower():
            timeline.append(f"{label}: {detail}.")
        elif label:
            timeline.append(f"{label}.")
        elif detail:
            timeline.append(f"{detail}.")
    add_section("timeline", "When it started and how it changed", timeline)

    measurements = [
        phrase
        for fact in facts
        if fact.fact_type in {FactType.vital_sign, FactType.measurement}
        for phrase in [_fact_phrase(fact)]
        if phrase
    ]
    add_section("measurements", "Measurements from your records", measurements)

    add_section("next_steps", "What happens next", _NEXT_STEPS)

    lines: List[str] = ["Your CareFlow care summary", ""]
    for section in sections:
        lines.append(section["title"])
        lines.extend(section["items"])
        lines.append("")
    lines.append(_DISCLAIMER)
    text = "\n".join(lines).strip()

    return {
        "sections": sections,
        "text": text,
        "disclaimer": _DISCLAIMER,
        "has_clinical_content": any(
            section["key"] not in {"next_steps"} for section in sections
        ),
    }


def _trim_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    clipped = text[:limit]
    boundary = max(clipped.rfind("\n"), clipped.rfind(". "), clipped.rfind(" "))
    if boundary > limit // 2:
        clipped = clipped[:boundary]
    return clipped.rstrip() + "\n\n[Summary shortened for voice playback.]"


class PatientVoiceService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        cipher: Cipher,
        documents: DocumentStore,
        followups: FollowUpService,
    ):
        self.settings = settings
        self.db = db
        self.cipher = cipher
        self.documents = documents
        self.followups = followups
        self._translation_cache: "OrderedDict[str, str]" = OrderedDict()
        self._tts_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    # --------------------------------------------------------------- helpers
    def _api_key_present(self) -> bool:
        return bool(self.settings.sarvam_api_key or os.getenv("SARVAM_API_KEY"))

    def _require_api(self) -> None:
        if not self._api_key_present():
            raise PatientVoiceError(
                "sarvam_not_configured",
                "Voice and translation services are not configured.",
                503,
            )

    def _translate(self, text: str, target: str, source: str = "auto") -> str:
        self._require_api()
        key = sha256_text(f"{target}\n{source}\n{text}")
        cached = self._translation_cache.get(key)
        if cached is not None:
            self._translation_cache.move_to_end(key)
            return cached
        try:
            from STT_translation.stt import translate_text
        except Exception as exc:  # pragma: no cover - import failure only
            raise PatientVoiceError(
                "voice_unavailable",
                "Translation service is unavailable.",
                503,
            ) from exc
        try:
            result = translate_text(text, target, source_language=source)
        except Exception as exc:
            raise PatientVoiceError(
                "translation_failed",
                "The summary could not be translated. Please read the English text.",
            ) from exc
        translated = str(result.get("translated_text") or "").strip()
        if not translated:
            raise PatientVoiceError(
                "translation_failed", "Translation returned no text."
            )
        self._translation_cache[key] = translated
        while len(self._translation_cache) > 128:
            self._translation_cache.popitem(last=False)
        return translated

    def _speak(self, text: str, target: str) -> Optional[Dict[str, Any]]:
        if not tts_supported(target):
            return None
        key = sha256_text(f"{target}\n{text}")
        cached = self._tts_cache.get(key)
        if cached is not None:
            self._tts_cache.move_to_end(key)
            return cached
        try:
            from STT_translation.stt import text_to_speech
        except Exception as exc:  # pragma: no cover - import failure only
            raise PatientVoiceError(
                "voice_unavailable", "Voice playback is unavailable.", 503
            ) from exc
        try:
            result = text_to_speech(
                text,
                target,
                speaker=self.settings.sarvam_tts_speaker,
                model=self.settings.sarvam_tts_model,
            )
        except Exception as exc:
            raise PatientVoiceError(
                "tts_failed", "Voice playback could not be generated."
            ) from exc
        audio = result.get("audio_bytes") or b""
        if not audio:
            raise PatientVoiceError("tts_failed", "Voice playback returned no audio.")
        payload = {
            "audio_base64": base64.b64encode(audio).decode("ascii"),
            "audio_mime": result.get("mime") or "audio/mpeg",
            "audio_chunks": result.get("chunks"),
        }
        self._tts_cache[key] = payload
        while len(self._tts_cache) > 16:
            self._tts_cache.popitem(last=False)
        return payload

    def _voice_payload(
        self,
        text: str,
        target: str,
        *,
        include_audio: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "tts_supported": tts_supported(target),
            "audio_base64": None,
            "audio_mime": None,
            "audio_error": None,
        }
        if not include_audio:
            return payload
        if not tts_supported(target):
            payload["audio_error"] = (
                "Voice playback is not available for this language. "
                "The translated text is still available."
            )
            return payload
        try:
            audio = self._speak(text, target)
        except PatientVoiceError as exc:
            payload["audio_error"] = exc.message
            return payload
        if audio:
            payload.update(audio)
        return payload

    # ------------------------------------------------------------- card/text
    def base_card(self, case_id: str) -> Dict[str, Any]:
        summary = self.db.get_summary(case_id)
        facts = self.db.get_facts(case_id, self.cipher)
        card = build_patient_context_card(summary, facts)
        return {"case_id": case_id, **card}

    def context_card(
        self,
        case_id: str,
        *,
        language: Optional[str],
        include_audio: bool,
    ) -> Dict[str, Any]:
        case = self.db.get_case(case_id, include_deleted=True)
        if case is None:
            raise PatientVoiceError("not_found", "Case not found.", 404)
        base = self.base_card(case_id)
        target = (
            normalize_bcp47(language)
            or normalize_bcp47(case.language_hint)
            or "en-IN"
        )
        if target not in SARVAM_TRANSLATE_LANGUAGES:
            raise PatientVoiceError(
                "unsupported_language",
                "This language is not supported for the care summary.",
                422,
            )
        english_text = base["text"]
        translated = False
        text = english_text
        if target != "en-IN":
            text = self._translate(english_text, target, source="en-IN")
            translated = True
        text = _trim_text(text, self.settings.patient_voice_max_chars)
        voice = self._voice_payload(text, target, include_audio=include_audio)
        return {
            "case_id": case_id,
            "language": target,
            "language_name": language_name(target),
            "text": text,
            "english_text": english_text,
            "translated": translated,
            "sections": base["sections"],
            "disclaimer": base["disclaimer"],
            "has_clinical_content": base["has_clinical_content"],
            **voice,
        }

    # -------------------------------------------------------------- followups
    def localize_followup(
        self,
        case_id: str,
        followup_id: str,
        *,
        language: Optional[str],
        include_audio: bool,
        patient_view: bool,
    ) -> Dict[str, Any]:
        record = self.followups.get_followup(case_id, followup_id)
        if patient_view and record.status not in {
            FollowUpStatus.sent,
            FollowUpStatus.answered,
        }:
            raise PatientVoiceError("not_found", "Follow-up question not found.", 404)
        default_target = normalize_bcp47(record.language) or "en-IN"
        target = normalize_bcp47(language) or default_target
        if target not in SARVAM_TRANSLATE_LANGUAGES:
            raise PatientVoiceError(
                "unsupported_language",
                "This language is not supported for follow-up questions.",
                422,
            )
        original = record.text
        source_language = "auto" if record.language == "hinglish" else (
            normalize_bcp47(record.language) or "auto"
        )
        translated = False
        text = original
        same_language = (
            source_language != "auto"
            and source_language == target
            and record.language != "hinglish"
        )
        if target != "en-IN" and not same_language:
            text = self._translate(original, target, source=source_language)
            translated = True
        elif target == "en-IN" and source_language not in {"en-IN", "auto"}:
            text = self._translate(original, target, source=source_language)
            translated = True
        voice = self._voice_payload(text, target, include_audio=include_audio)
        return {
            "case_id": case_id,
            "followup_id": followup_id,
            "status": record.status.value,
            "language": target,
            "language_name": language_name(target),
            "original_text": original,
            "text": text,
            "translated": translated,
            "tts_supported": voice["tts_supported"],
            "audio_base64": voice["audio_base64"],
            "audio_mime": voice["audio_mime"],
            "audio_error": voice["audio_error"],
        }

    # ------------------------------------------------------------------ audio
    def detect_language(self, audio_bytes: bytes, filename: str) -> Dict[str, Any]:
        self._require_api()
        try:
            from STT_translation.stt import detect_language
        except Exception as exc:  # pragma: no cover
            raise PatientVoiceError(
                "voice_unavailable", "Language detection is unavailable.", 503
            ) from exc
        try:
            return detect_language(audio_bytes, filename)
        except Exception as exc:
            raise PatientVoiceError(
                "language_detection_failed",
                "We could not detect the language from that recording. "
                "Please select a language or try again.",
                422,
            ) from exc

    def answer_followup_audio(
        self,
        case_id: str,
        followup_id: str,
        audio_bytes: bytes,
        filename: str,
        mime: str,
        actor: str,
        role: str,
        *,
        reprocess: bool = True,
    ) -> Dict[str, Any]:
        self._require_api()
        # Fail before the external call when the question cannot be answered.
        self.followups.ensure_answerable(case_id, followup_id)
        try:
            from STT_translation.stt import speech_to_language
        except Exception as exc:  # pragma: no cover
            raise PatientVoiceError(
                "voice_unavailable", "Speech transcription is unavailable.", 503
            ) from exc
        try:
            stt_result = speech_to_language(
                audio_bytes=audio_bytes,
                target_language="en-IN",
                filename=filename,
            )
        except Exception as exc:
            raise PatientVoiceError(
                "transcription_failed",
                "The voice answer could not be transcribed. Please retry.",
                422,
            ) from exc

        stored = self.documents.save(case_id, filename, audio_bytes)
        document_id = random_id("doc")
        self.db.add_document(
            document_id,
            case_id,
            stored["stored_name"],
            stored["original_name"],
            mime,
            stored["size"],
            stored["sha256"],
        )
        self.db.audit(
            actor,
            role,
            "followup_answer_audio_uploaded",
            case_id,
            {"document_id": document_id, "mime": mime, "size": stored["size"]},
        )
        try:
            result = self.followups.answer_audio(
                case_id,
                followup_id,
                stt_result,
                document_id,
                stored["original_name"],
                actor,
                role,
                reprocess=reprocess,
            )
        except FollowUpError:
            raise
        except Exception as exc:
            raise PatientVoiceError(
                "answer_failed", f"Voice answer could not be stored: {safe_error(exc)}", 500
            ) from exc
        result["document_id"] = document_id
        result["transcription"] = stt_result.get("transcription")
        result["translated_text"] = stt_result.get("translated_text")
        result["detected_language"] = (
            stt_result.get("language_code") or stt_result.get("language")
        )
        return result
