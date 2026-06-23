from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.utils.path_tool import get_abs_path


_RULES_PATH = Path(get_abs_path("data/rule_library/asr_corrections.json"))
_DEFAULT_RULES = [
    {
        "from": "装置",
        "to": "专杀",
        "description": "鹅鸭杀黑话中更常见的是专杀，听悟容易误成装置",
    },
    {
        "from": "专啥",
        "to": "专杀",
        "description": "同音误识别",
    },
    {
        "from": "报价",
        "to": "报警",
        "description": "会议语境中更常见的是报警",
    },
    {
        "from": "报假",
        "to": "报警",
        "description": "会议语境中更常见的是报警",
    },
    {
        "from": "爆警",
        "to": "报警",
        "description": "会议语境中更常见的是报警",
    },
]


def normalize_asr_text(text: str) -> dict[str, Any]:
    original = str(text or "").strip()
    if not original:
        return {
            "text": "",
            "original_text": original,
            "changed": False,
            "applied": [],
        }

    normalized = original
    applied: list[dict[str, str]] = []
    for rule in load_asr_corrections():
        src = str(rule.get("from") or "").strip()
        dst = str(rule.get("to") or "").strip()
        if not src or not dst or src == dst:
            continue
        if src not in normalized:
            continue
        normalized = normalized.replace(src, dst)
        applied.append(
            {
                "from": src,
                "to": dst,
                "description": str(rule.get("description") or "").strip(),
            }
        )

    return {
        "text": normalized,
        "original_text": original,
        "changed": normalized != original,
        "applied": applied,
    }


def load_asr_corrections(path: Path | None = None) -> list[dict[str, str]]:
    target = path or _RULES_PATH
    ensure_asr_corrections_file(target)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        data = []
    if not isinstance(data, list):
        return list(_DEFAULT_RULES)

    rules: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        src = str(item.get("from") or "").strip()
        dst = str(item.get("to") or "").strip()
        if not src or not dst:
            continue
        rules.append(
            {
                "from": src,
                "to": dst,
                "description": str(item.get("description") or "").strip(),
            }
        )
    return rules or list(_DEFAULT_RULES)


def ensure_asr_corrections_file(path: Path | None = None) -> Path:
    target = path or _RULES_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(
            json.dumps(_DEFAULT_RULES, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return target
