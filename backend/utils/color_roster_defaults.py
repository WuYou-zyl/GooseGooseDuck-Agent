"""座位默认颜色（与对局 UI 一致），供后端合并名单与前端展示。

不包含任何默认玩家昵称/ID；名单中的 player_id 由识别结果或用户填写。
"""

from __future__ import annotations

# 1-based 座位号 1..13 对应颜色；超出则回退最后一项
SEAT_COLORS: list[str] = [
    "白",
    "蓝",
    "绿",
    "粉",
    "红",
    "黄",
    "橙",
    "棕",
    "黑",
    "紫",
    "浅绿",
    "浅蓝",
    "浅紫",
]


def color_for_seat(seat_number: int) -> str:
    if seat_number < 1:
        return SEAT_COLORS[0]
    idx = seat_number - 1
    if idx >= len(SEAT_COLORS):
        return SEAT_COLORS[-1]
    return SEAT_COLORS[idx]
