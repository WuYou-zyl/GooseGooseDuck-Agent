"""
决策 Agent 使用的结构化输入/输出。
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class RuleCriticReview(BaseModel):
    """规则判官输出：挑刺与改正指引，不替代 Agent 全文。"""

    approved: bool = Field(
        ...,
        description="是否认为当前输出可结束修订循环",
    )
    issues: list[str] = Field(default_factory=list, description="挑刺要点")
    correction_instructions: str = Field(
        default="",
        description="给 Memory/Decision Agent 的改正说明",
    )
    rule_hits: list[str] = Field(default_factory=list, description="硬问题摘要")
    raw_notes: Optional[str] = Field(None, description="审计/备注")


class DecisionResult(BaseModel):
    prior_speech_analysis: str = Field(...)
    identity_inference: str = Field(
        default="",
        description="对玩家身份的推测（对应输出中的【玩家身份推测】）",
    )
    speech_suggestion: str = Field(...)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    debug_prompt: Optional[str] = Field(None, description="渲染后发送给 LLM 的完整 prompt（调试用）")
    rule_hits: list[str] = Field(
        default_factory=list,
        description="规则校对时命中的问题点简述",
    )
    rule_critic_notes: Optional[str] = Field(
        None,
        description="规则 Agent 校对说明",
    )
    rule_critic_debug_prompt: Optional[str] = Field(
        None,
        description="规则校对阶段完整 prompt（调试用）",
    )
