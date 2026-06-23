# pydantic 用于数据验证；A/B 共用 schema
from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field


def iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class IngestionOutput(BaseModel):
    """读入 Agent 输出，B 的输入。A 负责填充，B 负责消费。"""

    type: Literal["speech", "image"]
    content: str = Field(..., description="转写文本 或 图像描述")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展信息")
    timestamp: str = Field(..., description="ISO 8601 时间戳")
    session_id: str = Field(
        ...,
        description="本局游戏 ID，与 LangGraph checkpoint 的 thread_id 一致，整局共用",
    )
    meeting_id: Optional[str] = Field(
        None,
        description="本轮会议 ID（一次「开始监控→结束监控」周期），用于区分同一局内多场会议",
    )
    sequence_id: Optional[int] = Field(None, description="同 session 内顺序号，便于 B 排序")

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "speech",
                "content": "我怀疑3号是鸭子，他在锅炉房附近鬼鬼祟祟",
                "metadata": {
                    "speaker_id": "4",
                    "speaker_confidence": 0.92,
                    "emotion_summary": "语气坚定、略带怀疑",
                    "sentence_index": 1,
                    "is_final": True,
                },
                "timestamp": "2025-03-18T10:30:00.000Z",
                "session_id": "game_a1b2c3d4e5f67890",
                "meeting_id": "meeting_xyz789",
                "sequence_id": 42,
            }
        }
    }


class DecisionOutput(BaseModel):
    """决策 Agent 输出，推送给前端。"""

    session_id: str
    suggestion_type: Literal["speak", "vote", "respond", "general"]
    content: str = Field(..., description="建议文本")
    structured: Optional[dict[str, Any]] = Field(None, description="结构化建议，如投票对象、发言要点")
    timestamp: str
    trigger: Optional[str] = Field(None, description="触发来源：speech/image/summary")


class PlayerState(BaseModel):
    speaker_id: str
    latest_stance: str = Field(..., description="该玩家最新的立场/观点")
    key_claims: list[str] = Field(default_factory=list, description="该玩家发表过的关键指控")


class MemorySummary(BaseModel):
    session_id: str
    player_summaries: list[PlayerState] = Field(default_factory=list)
    overall_atmosphere: Optional[str] = Field(None, description="当前的整体讨论氛围")
    timestamp: str
    rule_critic_notes: Optional[str] = Field(
        None,
        description="规则校对说明（硬错误/软建议，供审计）",
    )