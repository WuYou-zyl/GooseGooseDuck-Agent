from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import numpy as np

from ..schemas.contract import IngestionOutput, iso_now
from ..services.asr_service import FunASRService
from ..services.emotion_service import SimpleEmotionService
from ..services.speaker_detection_service import TemplateSpeakerDetectionService


AsyncConsumer = Callable[[IngestionOutput], Awaitable[None]]


@dataclass
class IngestionAgent:
    """读入侧：ingest_speech_frames / 监控回调经 emit 调用 consumer（如 MemoryGraph）。"""

    session_id: str
    consumer: Optional[AsyncConsumer] = None

    def __post_init__(self):
        self._sequence_id = 0
        self._asr = None  # lazy-loaded to avoid slow startup
        self.emotion = SimpleEmotionService()
        self.speaker_det = TemplateSpeakerDetectionService()

    @property
    def asr(self):
        if self._asr is None:
            self._asr = FunASRService()
        return self._asr

    def _next_seq(self) -> int:
        self._sequence_id += 1
        return self._sequence_id

    async def emit(self, output: IngestionOutput) -> IngestionOutput:
        if self.consumer is not None:
            await self.consumer(output)
        return output

    async def ingest_speech_frames(
        self,
        audio_frames: list[bytes],
        screen_frame_bgr: Optional[np.ndarray] = None,
        is_final: bool = True,
    ) -> IngestionOutput:
        text = self.asr.transcribe_pcm_frames(audio_frames)
        emotion_summary = self.emotion.infer(text) if text else ""

        speaker_id = None
        if screen_frame_bgr is not None:
            speaker_id = self.speaker_det.detect_speaker_id(screen_frame_bgr)

        output = IngestionOutput(
            type="speech",
            content=text,
            metadata={
                **({"emotion_summary": emotion_summary} if emotion_summary else {}),
                **({"speaker_id": speaker_id} if speaker_id else {}),
                "is_final": is_final,
            },
            timestamp=iso_now(),
            session_id=self.session_id,
            meeting_id=None,
            sequence_id=self._next_seq(),
        )
        return await self.emit(output)