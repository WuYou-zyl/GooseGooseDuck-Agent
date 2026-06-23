"""LangGraph graph_state：只累积发言与摘要；决策上下文由推理 API 外部传入。"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, Literal

from pydantic import BaseModel, Field

from backend.schemas.contract import IngestionOutput, MemorySummary
from backend.schemas.decision import DecisionResult, RuleCriticReview


PlayerStatus = Literal["存活", "出局"]
EliminationCause = Literal["投票", "刺客会议狙击", "会议外淘汰", "小丑气球爆炸", "其他"]

class PlayerRosterEntry(BaseModel):
    player_id: str = Field(...)
    seat_number: int = Field(...)
    color: str = Field(...)
    status: PlayerStatus = Field(default="存活")
    
class EliminationRecord(BaseModel):
    victim_seat_number: int = Field(..., description="出局玩家座位号")
    cause: EliminationCause | str = Field(...)
    occurred_at: str = Field(..., description="发生时间")
    
class VoteRecord(BaseModel):
    voted_seat_number: int = Field(..., description="被投票玩家座位号，0号表示弃票")
    voted_number: int = Field(..., description="投票数")
    
class MeetingRoundState(BaseModel):
    """单轮会议状态；一轮对应本局内一次「开始监控新一轮会议」（与后端 meeting_id / 读入一致）。"""

    meeting_id: str = Field(
        ...,
        description="与当次监控会话的 meeting_id 一致，不由用户随意编造",
    )
    eliminations: list[EliminationRecord] = Field(default_factory=list)
    votes: list[VoteRecord] = Field(default_factory=list)

    my_view: str = Field(..., description="我视角下发生的事件简述")

class GameSettings(BaseModel):
    goose_count: int = Field(..., description="鹅的数量")
    duck_count: int = Field(..., description="鸭的数量")
    neutral_count: int = Field(..., description="中立的数量")
    known_roles: list[str] = Field(default_factory=list, description="已知角色名称列表")
    my_role: str = Field(..., description="我的角色名称")
    
class SituationSketch(BaseModel):
    game_settings: GameSettings = Field(...)
    player_roster: list[PlayerRosterEntry] = Field(default_factory=list, description="玩家列表")
    meeting_rounds: list[MeetingRoundState] = Field(
        default_factory=list,
        description=(
            "本局内已发生的全部会议（每轮由「开始监控新一轮会议」绑定 meeting_id 追加，"
            "非独立手建列表项；可在各轮内编辑淘汰、投票、我视角等）"
        ),
    )
    guessing_roles: dict[int, list[tuple[str, str]]] = Field(default_factory=dict, description="座位号到可能角色列表的映射，{座位号: [('阵营', '角色名称'), ...]}") 


def default_situation_sketch() -> SituationSketch:
    """State 初始化用空壳结构化局势（数值由客户端/后续接口更新）。"""
    return SituationSketch(
        game_settings=GameSettings(
            goose_count=0,
            duck_count=0,
            neutral_count=0,
            known_roles=[],
            my_role="未知",
        ),
        player_roster=[],
        meeting_rounds=[],
        guessing_roles={},
    )


class MemoryDecisionState(BaseModel):
    ingestions: Annotated[list[IngestionOutput], operator.add] = Field(
        default_factory=list,
        description="本局发言记录（累积）",
    )
    summary: Optional[MemorySummary] = Field(
        None,
        description="MemoryAgent 维护的摘要",
    )
    decision_result: Optional[DecisionResult] = Field(
        None,
        description="最近一次 decision_agent 输出（仅缓存结果，与触发信号无关）",
    )
    situation_sketch: SituationSketch = Field(
        default_factory=default_situation_sketch,
        description="结构化局势（手填/接口维护）",
    )
    situation_sketch_narrative: Optional[str] = Field(
        None,
        description="LLM 生成的局势叙述笔记（与结构化字段分离）",
    )
    sketch_updated_at: Optional[str] = Field(
        None,
        description="局势笔记更新时间 ISO8601",
    )
    sketch_warnings: list[str] = Field(
        default_factory=list,
        description="生成局势笔记时的告警",
    )
    memory_critic_review: Optional[RuleCriticReview] = Field(
        None,
        description="最近一次记忆路径规则判官评审",
    )
    memory_revision_attempts: int = Field(
        0,
        description="本轮发言已完成的记忆修订次数（不含草稿）",
    )
