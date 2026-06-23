from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from backend.agents.middleware import (
    MemoryAgentContext,
    before_agent_middleware,
    memory_phase_prompt_middleware,
    monitor_tool,
    situation_sketch_model_context,
)
from backend.model.factory import get_memory_chat_model
from backend.schemas.contract import IngestionOutput, MemorySummary, PlayerState
from backend.schemas.decision import RuleCriticReview
from backend.schemas.graph_state import SituationSketch
from backend.services import rag_query
from backend.utils.config_handler import agent_conf
from backend.utils.logger_handler import logger
from backend.utils.prompt_loader import load_memory_revise_prompts, load_summarize_prompt
from backend.agents import last_ai_text, merge_notes




def _format_short_memory(ingestions: list[IngestionOutput], recent_n: int) -> str:
    if not ingestions:
        return "(无)"
    window = ingestions[-recent_n:] if recent_n > 0 else ingestions
    lines: list[str] = []
    for io in window:
        sid = io.metadata.get("speaker_id", "?")
        seq = io.sequence_id
        seq_s = f"seq={seq}" if seq is not None else "seq=?"
        content = (io.content or "").strip()
        if len(content) > 200:
            content = content[:200] + "…"
        lines.append(f"[{seq_s}] 说话者={sid}: {content}")
    return "\n".join(lines)


def _build_user_content(user_template: str, prior: str, short: str, ingestion: str) -> str:
    """避免 str.format 与 JSON 花括号冲突，使用 replace。"""
    return (
        user_template.replace("{prior_memory_summary}", prior)
        .replace("{short_memory}", short)
        .replace("{ingestion}", ingestion)
    )


class MemoryAgent:
    def __init__(
        self,
        model=None,
        system_prompt: str | None = None,
        user_template: str | None = None,
    ):
        loaded_sys, loaded_user = load_summarize_prompt()
        rev_sys, rev_user = load_memory_revise_prompts()
        self._system_prompt = system_prompt if system_prompt is not None else loaded_sys
        self._user_template = user_template if user_template is not None else loaded_user
        self._revise_system = rev_sys
        self._revise_user_template = rev_user
        self.model = model if model is not None else get_memory_chat_model()
        self.agent = create_agent(
            model=self.model,
            tools=[rag_query],
            system_prompt=self._system_prompt,
            context_schema=MemoryAgentContext,
            middleware=[
                situation_sketch_model_context,
                memory_phase_prompt_middleware(self._system_prompt, self._revise_system),
                monitor_tool,
                before_agent_middleware,
            ],
        )

    def _memory_conf(self) -> dict:
        return agent_conf.get("memory_agent") or {}

    def _recursion_limit(self) -> int:
        mc = self._memory_conf()
        if mc.get("recursion_limit") is not None:
            return int(mc["recursion_limit"])
        dc = agent_conf.get("decision_agent") or {}
        return int(dc.get("recursion_limit", 25))

    async def process_draft(
        self,
        ingestions: list[IngestionOutput],
        summary: MemorySummary,
        recent_n: int | None = None,
        *,
        situation_sketch: SituationSketch | None = None,
        situation_sketch_narrative: str | None = None,
    ) -> MemorySummary:
        """首轮草稿：为本条 ingestion 追加一条 PlayerState。"""
        if not ingestions:
            return summary

        if recent_n is None:
            recent_n = int(self._memory_conf().get("recent_n", 13))

        last = ingestions[-1]
        prior_json = json.dumps(
            summary.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
        short_mem = _format_short_memory(ingestions, recent_n)
        ingestion_text = last.model_dump_json(indent=2, ensure_ascii=False)

        user_content = _build_user_content(
            self._user_template, prior_json, short_mem, ingestion_text
        )
        invoke_config = {"recursion_limit": self._recursion_limit()}
        input_state = {"messages": [HumanMessage(content=user_content)]}

        ctx = MemoryAgentContext(
            phase="draft",
            situation_sketch=situation_sketch,
            situation_sketch_narrative=situation_sketch_narrative,
        )
        try:
            result = await self.agent.ainvoke(
                input_state, invoke_config, context=ctx
            )
        except Exception as e:
            logger.warning(f"MemoryAgent: error invoking agent: {e}")
            raise e

        messages: list[BaseMessage] = list(result.get("messages") or [])
        summary_text = last_ai_text(messages)
        if not summary_text:
            logger.warning("MemoryAgent: empty model output, falling back to content snippet")
            summary_text = (last.content or "").strip() or "（无文本）"

        summary.timestamp = datetime.now(timezone.utc).isoformat()
        player_state = PlayerState(
            speaker_id=str(last.metadata.get("speaker_id") or "unknown"),
            latest_stance=summary_text,
        )
        summary.player_summaries.append(player_state)
        return summary

    async def process(
        self,
        ingestions: list[IngestionOutput],
        summary: MemorySummary,
        recent_n: int | None = None,
    ) -> MemorySummary:
        """兼容入口：等价于 process_draft（完整判官循环由 LangGraph 编排）。"""
        return await self.process_draft(ingestions, summary, recent_n)

    async def revise_from_critic(
        self,
        summary: MemorySummary,
        review: RuleCriticReview,
        ingestions: list[IngestionOutput],
        recent_n: int | None = None,
        *,
        situation_sketch: SituationSketch | None = None,
        situation_sketch_narrative: str | None = None,
    ) -> MemorySummary:
        """根据判官意见修订最后一条玩家的 latest_stance，不追加新 PlayerState。"""
        if not ingestions or not summary.player_summaries:
            return summary

        if recent_n is None:
            recent_n = int(self._memory_conf().get("recent_n", 13))

        last = ingestions[-1]
        critic_json = review.model_dump_json(indent=2, ensure_ascii=False)
        mem_json = json.dumps(
            summary.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
        short_mem = _format_short_memory(ingestions, recent_n)
        ingestion_text = last.model_dump_json(indent=2, ensure_ascii=False)

        user_content = (
            self._revise_user_template.replace("{critic_review_json}", critic_json)
            .replace("{memory_summary_json}", mem_json)
            .replace("{short_memory}", short_mem)
            .replace("{ingestion}", ingestion_text)
        )
        invoke_config = {"recursion_limit": self._recursion_limit()}
        input_state = {"messages": [HumanMessage(content=user_content)]}

        ctx = MemoryAgentContext(
            phase="revise",
            situation_sketch=situation_sketch,
            situation_sketch_narrative=situation_sketch_narrative,
        )
        try:
            result = await self.agent.ainvoke(
                input_state, invoke_config, context=ctx
            )
        except (AttributeError, NotImplementedError, TypeError):
            result = await asyncio.to_thread(
                self.agent.invoke, input_state, invoke_config, context=ctx
            )

        messages: list[BaseMessage] = list(result.get("messages") or [])
        text = last_ai_text(messages)
        if not text:
            logger.warning("MemoryAgent.revise_from_critic: empty output, keep previous stance")
            text = summary.player_summaries[-1].latest_stance

        ps = list(summary.player_summaries)
        ps[-1] = ps[-1].model_copy(update={"latest_stance": text})
        note = merge_notes(summary.rule_critic_notes, review.raw_notes or "")
        if review.issues:
            note = merge_notes(note, "判官：" + "；".join(review.issues[:5]))
        summary.timestamp = datetime.now(timezone.utc).isoformat()
        return summary.model_copy(update={"player_summaries": ps, "rule_critic_notes": note})
