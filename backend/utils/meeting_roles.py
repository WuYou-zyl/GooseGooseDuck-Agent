"""从 data/meeting_game_context.txt 解析「角色目录」列表。"""

from __future__ import annotations

import re
from functools import lru_cache

from backend.utils.path_tool import get_abs_path

_SECTION = "## 角色目录"


def _parse_role_line(line: str) -> list[str]:
    line = line.strip()
    if not line:
        return []
    # 支持中文逗号与英文逗号分隔
    parts = re.split(r"[,，]\s*", line)
    return [p.strip() for p in parts if p.strip()]


@lru_cache(maxsize=1)
def load_meeting_role_names() -> tuple[str, ...]:
    """返回 meeting_game_context 中角色目录元组（顺序固定）。"""
    path = get_abs_path("data/meeting_game_context.txt")
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    in_section = False
    for raw in lines:
        line = raw.strip()
        if line.startswith(_SECTION):
            in_section = True
            continue
        if in_section:
            if line.startswith("##"):
                break
            names = _parse_role_line(line)
            if names:
                return tuple(names)
    return ()
