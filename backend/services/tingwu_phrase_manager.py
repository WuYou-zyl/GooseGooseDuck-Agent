from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from backend.services.tingwu_openapi_client import TingwuOpenApiClient
from backend.utils.asr_hotwords import (
    HOTWORD_OUTPUT,
    ensure_hotword_file,
    merge_hotwords_text,
    normalize_hotword,
)
from backend.utils.logger_handler import get_logger
from backend.utils.path_tool import get_abs_path

logger = get_logger("asr")

_CACHE_PATH = Path(get_abs_path("data/tingwu_phrase_cache.json"))
_DEFAULT_NAME = "ggd-hotwords"
_DEFAULT_DESCRIPTION = "Goose Goose Duck realtime ASR hotwords"
_MAX_PHRASES = 300
_MIN_WEIGHT = -6
_MAX_WEIGHT = 5
_DEFAULT_WEIGHT = 2


class TingwuPhraseManager:
    def __init__(
        self,
        client: TingwuOpenApiClient | None = None,
        *,
        hotword_path: Path | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self.client = client or TingwuOpenApiClient()
        self.hotword_path = hotword_path or HOTWORD_OUTPUT
        self.cache_path = cache_path or _CACHE_PATH
        self._lock = threading.Lock()

    def sync_phrase_book(self, extra_words: list[str] | None = None) -> str | None:
        with self._lock:
            payload = self._build_payload(extra_words or [])
            if not payload["WordWeights"]:
                logger.warning("No valid hotwords collected for Tingwu phrase book")
                return None

            fingerprint = self._fingerprint(payload)
            cache = self._load_cache()
            cached_phrase_id = str(cache.get("phrase_id") or "").strip()

            if cache.get("fingerprint") == fingerprint and cached_phrase_id:
                logger.info("Reusing cached Tingwu PhraseId=%s", cached_phrase_id)
                return cached_phrase_id

            phrase_id = cached_phrase_id or self._find_phrase_id_by_name(payload["Name"])
            if phrase_id:
                logger.info("Updating Tingwu hotword phrase book id=%s", phrase_id)
                self.client.update_phrase(phrase_id, payload)
            else:
                logger.info("Creating Tingwu hotword phrase book name=%s", payload["Name"])
                phrase_id = self.client.create_phrase(payload)

            if not phrase_id:
                logger.warning("Tingwu phrase sync did not return a PhraseId")
                return None

            self._save_cache(
                {
                    "phrase_id": phrase_id,
                    "fingerprint": fingerprint,
                    "name": payload["Name"],
                    "word_count": len(payload["WordWeights"]),
                }
            )
            return phrase_id

    def get_status(self) -> dict[str, Any]:
        cache = self._load_cache()
        ensure_hotword_file(self.hotword_path)
        text = self.hotword_path.read_text(encoding="utf-8", errors="ignore")
        words = [line.strip() for line in text.splitlines() if line.strip()]
        return {
            "phrase_id": str(cache.get("phrase_id") or "").strip() or None,
            "fingerprint": cache.get("fingerprint"),
            "name": cache.get("name") or os.environ.get("TINGWU_HOTWORD_NAME", _DEFAULT_NAME),
            "word_count": len(words),
            "hotwords_text": text,
        }

    def save_hotwords_text(self, text: str) -> dict[str, Any]:
        cleaned_lines: list[str] = []
        seen: set[str] = set()
        for raw in str(text or "").splitlines():
            token = normalize_hotword(raw)
            if not token or token in seen:
                continue
            if len(token) > 10:
                continue
            seen.add(token)
            cleaned_lines.append(token)
        cleaned_lines = cleaned_lines[:_MAX_PHRASES]
        self.hotword_path.parent.mkdir(parents=True, exist_ok=True)
        self.hotword_path.write_text(
            ("\n".join(cleaned_lines) + ("\n" if cleaned_lines else "")),
            encoding="utf-8",
        )
        status = self.get_status()
        status["word_count"] = len(cleaned_lines)
        status["hotwords_text"] = self.hotword_path.read_text(
            encoding="utf-8", errors="ignore"
        )
        return status

    def _build_payload(self, extra_words: list[str]) -> dict[str, Any]:
        ensure_hotword_file(self.hotword_path)
        base_text = self.hotword_path.read_text(encoding="utf-8", errors="ignore")
        merged_text = merge_hotwords_text(base_text, extra_words)

        words: list[str] = []
        seen: set[str] = set()
        for line in merged_text.splitlines():
            token = normalize_hotword(line)
            if not token or token in seen:
                continue
            if len(token) > 10:
                continue
            if any(ch in token for ch in ",，。.!！？?;；:：/\\'\"()[]{}<>@#$%^&*_+=|`~"):
                continue
            seen.add(token)
            words.append(token)

        words = words[:_MAX_PHRASES]
        weight = _env_int("TINGWU_HOTWORD_DEFAULT_WEIGHT", _DEFAULT_WEIGHT)
        weight = max(_MIN_WEIGHT, min(_MAX_WEIGHT, weight))
        return {
            "Name": os.environ.get("TINGWU_HOTWORD_NAME", _DEFAULT_NAME).strip()
            or _DEFAULT_NAME,
            "Description": os.environ.get(
                "TINGWU_HOTWORD_DESCRIPTION", _DEFAULT_DESCRIPTION
            ).strip()
            or _DEFAULT_DESCRIPTION,
            "WordWeights": {word: weight for word in words},
        }

    @staticmethod
    def _fingerprint(payload: dict[str, Any]) -> str:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(body.encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict[str, Any]:
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_cache(self, data: dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _find_phrase_id_by_name(self, name: str) -> str:
        try:
            phrases = self.client.list_phrases()
        except Exception:
            logger.warning("Failed to list Tingwu phrase books", exc_info=True)
            return ""
        for item in phrases:
            if str(item.get("Name") or "").strip() == name:
                return str(item.get("PhraseId") or "").strip()
        return ""


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
