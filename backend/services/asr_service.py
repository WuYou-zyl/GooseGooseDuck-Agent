from __future__ import annotations

import os
import time
import wave
from dataclasses import dataclass
from typing import Callable, Optional

from backend.utils.asr_hotwords import load_hotwords_text, merge_hotwords_text

AutoModel = None
rich_transcription_postprocess = None


def _load_funasr():
    """Import FunASR only when legacy ASR is actually used."""
    global AutoModel, rich_transcription_postprocess
    if AutoModel is not None:
        return AutoModel, rich_transcription_postprocess
    try:
        from funasr import AutoModel as _AutoModel  # type: ignore
        from funasr.utils.postprocess_utils import (  # type: ignore
            rich_transcription_postprocess as _postprocess,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "FunASR is not installed or failed to import. Install requirements.txt "
            "before using legacy FunASR ASR."
        ) from e
    AutoModel = _AutoModel
    rich_transcription_postprocess = _postprocess
    return AutoModel, rich_transcription_postprocess


@dataclass
class ASRConfig:
    model: str = "iic/SenseVoiceSmall"
    vad_model: str = "fsmn-vad"
    sample_rate: int = 44100
    channels: int = 2
    hotword_text: str = ""
    dynamic_hotwords_provider: Optional[Callable[[], list[str]]] = None


class FunASRService:
    def __init__(self, config: Optional[ASRConfig] = None, preloaded_model=None):
        self.config = config or ASRConfig()
        if not self.config.hotword_text:
            self.config.hotword_text = load_hotwords_text()
        if preloaded_model is not None:
            self._model = preloaded_model
        else:
            auto_model, _ = _load_funasr()
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._model = auto_model(
                model=self.config.model,
                vad_model=self.config.vad_model,
                vad_kwargs={"max_single_segment_time": 30000},
                trust_remote_code=True,
                device=device,
            )

    def _get_hotword_text(self) -> str:
        provider = self.config.dynamic_hotwords_provider
        if provider is None:
            return self.config.hotword_text
        try:
            extra_words = provider() or []
        except Exception:
            extra_words = []
        return merge_hotwords_text(self.config.hotword_text, extra_words)

    def transcribe_pcm_frames(self, audio_frames: list[bytes]) -> str:
        if not audio_frames:
            return ""

        audio_data = b"".join(audio_frames)
        temp_dir = os.path.join(os.getcwd(), "tmp_asr")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"audio_{int(time.time() * 1000)}.wav")

        with wave.open(temp_file, "wb") as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.config.sample_rate)
            wf.writeframes(audio_data)

        try:
            result = self._model.generate(
                input=temp_file,
                cache={},
                language="auto",
                hotword=self._get_hotword_text(),
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            if result and len(result) > 0:
                text = result[0].get("text", "").strip()
                _, postprocess = _load_funasr()
                return postprocess(text) if postprocess is not None else text
            return ""
        finally:
            try:
                os.remove(temp_file)
            except OSError:
                pass
