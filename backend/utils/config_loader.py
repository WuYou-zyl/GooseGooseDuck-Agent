"""Configuration loader for the migrated input-processing stack."""

from __future__ import annotations

from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field

from backend.utils.pathtool import get_abs_path


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 9888
    debug: bool = True


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    device_index: Optional[int] = Field(None)
    save_path: str = "backend/tests/temp_audio"
    save_temp: bool = True
    min_silence_time: float = 0.8
    max_silence_time: float = 1.0
    min_segment_time: float = 0.5
    max_segment_time: float = 15.0
    vad_threshold: float = 0.3
    vad_model: str = "silero_vad.onnx"
    spk_model: str = "3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx"


class ASRConfig(BaseModel):
    asr_dir: str = "sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30"
    asr_model: str = ""
    graph: str = ""
    tokens: str = "tokens.txt"
    encoder: str = "encoder.int8.onnx"
    decoder: str = "decoder.onnx"
    joiner: str = "joiner.int8.onnx"
    provider: str = "cpu"


class ModelConfig(BaseModel):
    whisper_model_path: str = "backend/models/whisper-tiny"
    ocr_model_dir: str = "backend/models/paddle_ocr"
    compute_type: str = "int8"


class VisionConfig(BaseModel):
    mode: str = "window"
    target: str = ""
    fps_limit: int = 1


class GameSettingConfig(BaseModel):
    seat_num: int = Field(13, ge=1, le=15)


class AppConfig(BaseModel):
    enabled: bool = False
    server: ServerConfig = Field(default_factory=ServerConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    game_setting: GameSettingConfig = Field(default_factory=GameSettingConfig)


def get_config_path(config_path: str = "config/input_processing.yaml"):
    return get_abs_path(config_path)


def load_config(config_path: str = "config/input_processing.yaml") -> AppConfig:
    path = get_config_path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return AppConfig(**data)


def _model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def get_public_config() -> Dict[str, Any]:
    current = load_config()
    return {
        "server": {
            "host": current.server.host,
            "port": current.server.port,
        },
        "vision": {
            "mode": current.vision.mode,
            "target": current.vision.target,
            "fps_limit": current.vision.fps_limit,
        },
        "game_setting": {
            "seat_num": current.game_setting.seat_num,
        },
    }


def save_public_config(
    payload: Dict[str, Any],
    config_path: str = "config/input_processing.yaml",
) -> Dict[str, Any]:
    path = get_config_path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw.setdefault("server", {})
    raw.setdefault("vision", {})
    raw.setdefault("game_setting", {})

    if "server" in payload:
        server = payload["server"] or {}
        if "host" in server:
            raw["server"]["host"] = str(server["host"])
        if "port" in server:
            raw["server"]["port"] = int(server["port"])

    if "vision" in payload:
        vision = payload["vision"] or {}
        if "mode" in vision:
            raw["vision"]["mode"] = str(vision["mode"])
        if "target" in vision:
            raw["vision"]["target"] = str(vision["target"])
        if "fps_limit" in vision:
            raw["vision"]["fps_limit"] = int(vision["fps_limit"])

    if "game_setting" in payload:
        game_setting = payload["game_setting"] or {}
        if "seat_num" in game_setting:
            raw["game_setting"]["seat_num"] = int(game_setting["seat_num"])

    validated = AppConfig(**raw)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_model_to_dict(validated), f, allow_unicode=True, sort_keys=False)

    global config
    config = validated
    return get_public_config()


config = load_config()
