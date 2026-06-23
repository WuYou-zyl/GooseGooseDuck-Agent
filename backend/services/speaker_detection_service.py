from __future__ import annotations

from typing import Optional

import numpy as np

from backend.legacy.extract_speaker_num import extract_player_num_from_array


class TemplateSpeakerDetectionService:
    """Detect current speaker id from a meeting screenshot using template matching."""

    def detect_speaker_id(self, frame_bgr: np.ndarray) -> Optional[str]:
        return extract_player_num_from_array(frame_bgr)

