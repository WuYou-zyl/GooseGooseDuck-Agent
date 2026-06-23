"""Use Gemini multimodal model to extract player roster from game screenshots."""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("ggd-a.gemini-roster")

_PROMPT = """\
这是一张鹅鸭杀（Goose Goose Duck）游戏的截图，显示了玩家名单界面。
请从图片中识别所有玩家的编号（座位号）和游戏内昵称。

请严格按以下 JSON 数组格式返回，不要添加任何其他文字或 markdown 标记：
[{"number": "01", "name": "玩家昵称"}, {"number": "02", "name": "玩家昵称"}]

要求：
- number 用两位数字字符串，如 "01", "02", ..., "16"
- name 是玩家的游戏内显示昵称，原样保留
- 只返回你能确认识别到的玩家
- 如果图片中没有名单，返回空数组 []
"""


def extract_player_roster_gemini(img_bgr: np.ndarray) -> list[dict]:
    """Send game screenshot to Gemini and extract player roster.

    Returns list of ``{"number": "01", "name": "..."}`` dicts,
    same format as the legacy OCR version.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set in environment")
        return []

    try:
        from google import genai
    except ImportError:
        logger.error("google-genai is not installed; run: pip install google-genai")
        return []

    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_bytes = buf.tobytes()
    img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

    client = genai.Client(api_key=api_key)

    print("[Gemini-Roster] Sending screenshot to Gemini ...", flush=True)
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                _PROMPT,
                genai.types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            ],
        )
    except Exception as e:
        logger.error("Gemini API call failed: %s", e, exc_info=True)
        print(f"[Gemini-Roster] API call failed: {e}", flush=True)
        return []

    raw = response.text.strip()
    print(f"[Gemini-Roster] Raw response:\n{raw}", flush=True)

    roster = _parse_roster_json(raw)
    if roster is None:
        logger.warning("Failed to parse Gemini response as roster JSON")
        return []

    print(f"[Gemini-Roster] Final roster ({len(roster)} players):", flush=True)
    for r in roster:
        print(f"  {r['number']} → {r['name']}", flush=True)
    logger.info("Gemini roster: found %d players: %s", len(roster), roster)
    return roster


def _parse_roster_json(text: str) -> Optional[list[dict]]:
    """Parse Gemini response text into a validated roster list."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list):
        return None

    result: list[dict] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        num = str(entry.get("number", "")).strip()
        name = str(entry.get("name", "")).strip()
        if not num or not name:
            continue
        try:
            num_int = int(num)
        except ValueError:
            continue
        if not (1 <= num_int <= 16):
            continue
        result.append({"number": str(num_int).zfill(2), "name": name})

    result.sort(key=lambda x: x["number"])
    seen: set[str] = set()
    deduped = []
    for r in result:
        if r["number"] not in seen:
            seen.add(r["number"])
            deduped.append(r)
    return deduped
