from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

from backend.services.tingwu_phrase_manager import TingwuPhraseManager
from backend.services.tingwu_openapi_client import TingwuOpenApiClient
from backend.utils.asr_text_normalizer import normalize_asr_text
from backend.utils.logger_handler import get_logger
from backend.utils.path_tool import get_abs_path

try:
    import soundcard as sc  # type: ignore
except Exception:  # pragma: no cover
    sc = None

logger = get_logger("asr")


class SystemAudioRealtimeService:
    """System loopback capture backed by Tingwu realtime ASR."""

    def __init__(
        self,
        on_new_record=None,
        auto_save: bool = True,
        preloaded_model=None,
        dynamic_hotwords_provider: Optional[Callable[[], list[str]]] = None,
        samplerate: int = 16000,
        channels: int = 1,
        blocksize: int = 320,
    ):
        del preloaded_model
        if sc is None:
            raise RuntimeError("soundcard is not available. Install requirements.txt.")

        self.samplerate = int(_env_first("TINGWU_REALTIME_SAMPLE_RATE") or samplerate)
        self.channels = channels
        stream_chunk_bytes = int(_env_first("TINGWU_REALTIME_STREAM_CHUNK_BYTES") or 640)
        self.blocksize = max(stream_chunk_bytes // 2, blocksize)
        self.speaker = self._get_default_speaker()
        self.client = TingwuOpenApiClient()
        self.phrase_manager = TingwuPhraseManager(self.client)
        self.dynamic_hotwords_provider = dynamic_hotwords_provider

        self.is_recording = False
        self.conversation_log: list[dict] = []
        self._log_lock = threading.Lock()
        self._current_speaker = "unknown"
        self._speaker_lock = threading.Lock()
        self.on_new_record = on_new_record
        self.auto_save = auto_save
        self._thread: Optional[threading.Thread] = None
        self._meeting = None
        self._collector = _TingwuRealtimeCollector(self)
        self._task_id: str = ""
        self._meeting_join_url: str = ""
        self._stop_event = threading.Event()
        self.on_partial_result = None

    def _get_default_speaker(self):
        speakers = sc.all_speakers()
        default_speaker = sc.default_speaker()
        if not speakers or default_speaker is None:
            raise RuntimeError("No system speaker is available for loopback recording.")
        return default_speaker

    def _get_loopback_microphone(self):
        self.speaker = self._get_default_speaker()
        microphone = sc.get_microphone(self.speaker.id, include_loopback=True)
        if microphone is None:
            raise RuntimeError(
                f"Unable to create loopback input for speaker {self.speaker.name!r}."
            )
        return microphone

    def start(self) -> None:
        if self.is_recording:
            return

        try:
            import nls
        except ImportError as exc:
            raise RuntimeError(
                "Aliyun NLS/Tingwu realtime SDK is not installed. "
                "Install it from requirements.txt before using online ASR."
            ) from exc

        create_response = self.client.create_realtime_task(self._build_create_task_body())
        task_data = create_response.get("Data") or {}
        self._task_id = str(task_data.get("TaskId") or "")
        self._meeting_join_url = str(task_data.get("MeetingJoinUrl") or "")
        if not self._task_id or not self._meeting_join_url:
            raise RuntimeError(f"Tingwu create task failed: {create_response}")

        meeting_cls = _get_realtime_meeting_class(nls)
        nls.enableTrace(_env_bool("ALIYUN_NLS_ENABLE_TRACE", False))
        self._collector.reset()
        self._meeting = meeting_cls(
            url=self._meeting_join_url,
            on_sentence_begin=self._collector.on_sentence_begin,
            on_sentence_end=self._collector.on_sentence_end,
            on_start=self._collector.on_start,
            on_result_changed=self._collector.on_result_changed,
            on_result_translated=self._collector.on_result_translated,
            on_completed=self._collector.on_completed,
            on_error=self._collector.on_error,
            on_close=self._collector.on_close,
            callback_args=["ggd-online-asr"],
        )
        self._meeting.start()

        self.is_recording = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Started Tingwu realtime ASR task=%s speaker=%s",
            self._task_id,
            self.speaker.name,
        )

    def stop(self, round_num: int = 1) -> None:
        del round_num
        if not self.is_recording:
            return

        self.is_recording = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        if self._meeting is not None:
            try:
                self._meeting.stop()
            except Exception:
                logger.warning("Failed to stop Tingwu meeting cleanly", exc_info=True)

        try:
            self._collector.wait_for_completion(timeout=20.0)
        except Exception:
            logger.warning("Timed out waiting for Tingwu completion", exc_info=True)

        if self._task_id:
            try:
                self.client.stop_realtime_task(self._task_id)
            except Exception:
                logger.warning("Failed to stop Tingwu task %s", self._task_id, exc_info=True)

        self._meeting = None
        self._task_id = ""
        self._meeting_join_url = ""

    def set_speaker(self, speaker, round_num: int = 1) -> None:
        del round_num
        with self._speaker_lock:
            self._current_speaker = str(speaker or "unknown")

    def get_speaker(self):
        with self._speaker_lock:
            return self._current_speaker

    def _record_loop(self) -> None:
        microphone = self._get_loopback_microphone()
        logger.info("Recording system loopback to Tingwu from %s", self.speaker.name)
        with microphone.recorder(
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=self.blocksize,
        ) as recorder:
            while self.is_recording and not self._stop_event.is_set():
                try:
                    frames = recorder.record(numframes=self.blocksize)
                except Exception:
                    time.sleep(0.01)
                    continue
                if frames is None or len(frames) == 0:
                    continue

                pcm_data = self._frames_to_pcm_bytes(frames)
                if not pcm_data or self._meeting is None:
                    continue
                try:
                    self._meeting.send_audio(pcm_data)
                except Exception:
                    logger.warning("Failed to send audio to Tingwu", exc_info=True)
                    time.sleep(0.05)
                    continue

                interval_ms = int(_env_first("TINGWU_REALTIME_STREAM_INTERVAL_MS") or 10)
                time.sleep(max(interval_ms, 1) / 1000.0)

    def _frames_to_pcm_bytes(self, frames: np.ndarray) -> bytes:
        audio = np.asarray(frames, dtype=np.float32)
        if audio.size == 0:
            return b""
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16)
        return pcm.tobytes()

    def _handle_sentence(self, payload: dict[str, Any]) -> None:
        segment = _extract_segment(payload)
        if segment is None:
            return

        start_ms = int(segment.get("start_ms") or 0)
        end_ms = int(segment.get("end_ms") or 0)
        duration = max(0.0, (end_ms - start_ms) / 1000.0) if end_ms >= start_ms else 0.0
        normalized = normalize_asr_text(segment["text"])
        record = {
            "timestamp": time.strftime("%H:%M:%S"),
            "text": normalized["text"],
            "original_text": normalized["original_text"],
            "text_normalized": bool(normalized["changed"]),
            "normalizer_rules": normalized["applied"],
            "emotion": "neutral",
            "speaker": self.get_speaker(),
            "duration": round(duration, 2),
            "round": 1,
        }
        with self._log_lock:
            self.conversation_log.append(record)
        if self.on_new_record:
            self.on_new_record(record)
        if self.auto_save:
            self._save_to_file("game_analysis.json")

    def _save_to_file(self, filename: str) -> None:
        path = filename
        if filename == "game_analysis.json":
            path = get_abs_path(os.path.join("data", "game_analysis.json"))
        with self._log_lock:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.conversation_log, f, ensure_ascii=False, indent=2)

    def _build_create_task_body(self) -> dict[str, Any]:
        phrase_id = None
        if _env_bool("TINGWU_REALTIME_PHRASEBOOK_ENABLED", True):
            try:
                phrase_id = self.phrase_manager.sync_phrase_book(
                    self._get_dynamic_hotwords()
                )
            except Exception:
                logger.warning("Failed to sync Tingwu hotword phrase book", exc_info=True)
        body: dict[str, Any] = {
            "AppKey": _require_env("TINGWU_APP_KEY"),
            "Input": {
                "Format": _env_first("TINGWU_REALTIME_FORMAT") or "pcm",
                "SampleRate": self.samplerate,
                "SourceLanguage": _env_first("TINGWU_REALTIME_SOURCE_LANGUAGE") or "cn",
                "TaskKey": f"ggd{dt.datetime.now().strftime('%Y%m%d%H%M%S')}",
                "ProgressiveCallbacksEnabled": _env_bool(
                    "TINGWU_REALTIME_PROGRESSIVE_CALLBACKS_ENABLED",
                    False,
                ),
            },
            "Parameters": _build_tingwu_parameters(),
        }
        if phrase_id:
            body["Parameters"].setdefault("Transcription", {})
            body["Parameters"]["Transcription"]["PhraseId"] = phrase_id
        language_hints = [
            item.strip()
            for item in (_env_first("TINGWU_REALTIME_LANGUAGE_HINTS") or "").split(",")
            if item.strip()
        ]
        if language_hints:
            body["Input"]["LanguageHints"] = language_hints
        return body

    def _get_dynamic_hotwords(self) -> list[str]:
        provider = self.dynamic_hotwords_provider
        if provider is None:
            return []
        try:
            return [str(x).strip() for x in (provider() or []) if str(x).strip()]
        except Exception:
            logger.warning("Failed to get dynamic hotwords", exc_info=True)
            return []


class _TingwuRealtimeCollector:
    def __init__(self, owner: SystemAudioRealtimeService) -> None:
        self.owner = owner
        self._lock = threading.Lock()
        self._completed = threading.Event()
        self.error_message: str | None = None
        self.intermediate_text = ""
        self.raw_events: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._completed.clear()
        self.error_message = None
        self.intermediate_text = ""
        self.raw_events = []

    def on_sentence_begin(self, message: str, *args) -> None:
        del args
        self._append_raw("sentence_begin", message)

    def on_sentence_end(self, message: str, *args) -> None:
        del args
        payload = self._append_raw("sentence_end", message)
        self.owner._handle_sentence(payload)

    def on_start(self, message: str, *args) -> None:
        del args
        self._append_raw("start", message)

    def on_result_changed(self, message: str, *args) -> None:
        del args
        payload = self._append_raw("result_changed", message)
        text = _extract_text(payload)
        if text:
            self.intermediate_text = text
            normalized = normalize_asr_text(text)
            cb = getattr(self.owner, "on_partial_result", None)
            if callable(cb):
                cb(
                    {
                        "timestamp": time.strftime("%H:%M:%S"),
                        "text": normalized["text"],
                        "original_text": normalized["original_text"],
                        "text_normalized": bool(normalized["changed"]),
                        "normalizer_rules": normalized["applied"],
                        "speaker": self.owner.get_speaker(),
                    }
                )

    def on_result_translated(self, message: str, *args) -> None:
        del args
        self._append_raw("result_translated", message)

    def on_completed(self, message: str, *args) -> None:
        del args
        self._append_raw("completed", message)
        self._completed.set()

    def on_error(self, message: str, *args) -> None:
        del args
        self._append_raw("error", message)
        self.error_message = message
        self._completed.set()

    def on_close(self, *args) -> None:
        del args
        self._completed.set()

    def wait_for_completion(self, timeout: float) -> None:
        if not self._completed.wait(timeout):
            raise TimeoutError("Timed out waiting for Tingwu realtime stream completion")

    def _append_raw(self, event_type: str, message: str) -> dict[str, Any]:
        payload = _safe_json_loads(message)
        with self._lock:
            self.raw_events.append({"type": event_type, "message": payload or message})
        return payload


def _build_tingwu_parameters() -> dict[str, Any]:
    parameters = _loads_json_object(_env_first("TINGWU_REALTIME_CUSTOM_PARAMETERS_JSON") or "{}")
    # System loopback is already a mixed output stream; enabling diarization here
    # often hurts gaming voice-chat accuracy more than it helps.
    if _env_bool("TINGWU_REALTIME_ENABLE_DIARIZATION", False):
        parameters.setdefault("Transcription", {})
        parameters["Transcription"]["DiarizationEnabled"] = True
        parameters["Transcription"]["Diarization"] = {
            "SpeakerCount": int(_env_first("TINGWU_REALTIME_SPEAKER_COUNT") or 2),
        }
    if _env_bool("TINGWU_REALTIME_TRANSLATION_ENABLED", False):
        targets = [
            item.strip()
            for item in (_env_first("TINGWU_REALTIME_TARGET_LANGUAGES") or "").split(",")
            if item.strip()
        ]
        parameters["TranslationEnabled"] = True
        parameters["Translation"] = {"TargetLanguages": targets or ["en"]}
    if _env_bool("TINGWU_REALTIME_AUTO_CHAPTERS_ENABLED", False):
        parameters["AutoChaptersEnabled"] = True
    if _env_bool("TINGWU_REALTIME_MEETING_ASSISTANCE_ENABLED", False):
        parameters["MeetingAssistanceEnabled"] = True
    if _env_bool("TINGWU_REALTIME_SUMMARIZATION_ENABLED", False):
        parameters["SummarizationEnabled"] = True
    if _env_bool("TINGWU_REALTIME_TEXT_POLISH_ENABLED", False):
        parameters["TextPolishEnabled"] = True
    return parameters


def _safe_json_loads(message: str) -> dict[str, Any]:
    try:
        data = json.loads(message)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_text(payload: dict[str, Any]) -> str:
    nested = payload.get("payload") or {}
    return str(
        nested.get("result")
        or nested.get("text")
        or payload.get("result")
        or payload.get("text")
        or ""
    ).strip()


def _extract_segment(payload: dict[str, Any]) -> dict[str, Any] | None:
    nested = payload.get("payload") or {}
    text = _extract_text(payload)
    if not text:
        return None
    start_ms = int(
        nested.get("begin_time")
        or nested.get("start_time")
        or payload.get("begin_time")
        or payload.get("start_time")
        or 0
    )
    end_ms = int(nested.get("end_time") or payload.get("end_time") or start_ms)
    return {"start_ms": start_ms, "end_ms": end_ms, "text": text}


def _get_realtime_meeting_class(nls_module):
    meeting_cls = getattr(nls_module, "NlsRealtimeMeeting", None)
    if meeting_cls is not None:
        return meeting_cls
    raise RuntimeError(
        "The installed 'nls' package does not expose NlsRealtimeMeeting. "
        "Install a Tingwu-capable SDK build."
    )


def _loads_json_object(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _require_env(key: str) -> str:
    value = _env_first(key)
    if not value:
        raise ValueError(f"{key} is not configured")
    return value


def _env_bool(key: str, default: bool) -> bool:
    raw = _env_first(key)
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_first(*keys: str) -> str:
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""
