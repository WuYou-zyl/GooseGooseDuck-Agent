from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from typing import Any, Literal

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from backend.agents.middleware import (
    RuleCriticContext,
    before_agent_middleware,
    rule_critic_phase_prompt_middleware,
    monitor_tool,
    situation_sketch_model_context,
)
from backend.model.factory import (
    get_rule_critic_chat_model,
    get_rule_critic_fallback_chat_model,
)
from backend.schemas.contract import IngestionOutput, MemorySummary
from backend.schemas.graph_state import SituationSketch
from backend.schemas.decision import DecisionResult, RuleCriticReview
from backend.services import rag_query
from backend.utils.config_handler import agent_conf
from backend.utils.logger_handler import logger
from backend.utils.prompt_loader import (
    load_rule_critic_decision_prompts,
    load_rule_critic_memory_prompts,
)
from backend.agents import last_ai_text, stream_deltas_from_chunk


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not (text or "").strip():
        return None
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence:
        raw = fence.group(1).strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    i = raw.find("{")
    j = raw.rfind("}")
    if i >= 0 and j > i:
        try:
            obj = json.loads(raw[i : j + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def is_rule_critic_enabled() -> bool:
    c = agent_conf.get("rule_critic") or {}
    return bool(c.get("enabled", True))


def _recursion_limit() -> int:
    return int(agent_conf.get("rule_critic").get("recursion_limit", 15))


def _timeout_seconds(which: str) -> float:
    c = agent_conf.get("rule_critic") or {}
    key = f"{which}_timeout_seconds"
    if c.get(key) is not None:
        return float(c[key])
    return float(c.get("timeout_seconds", 120.0))


def _build_review_debug_prompt(
    system_prompt: str, user_content: str, messages: list[BaseMessage]
) -> str:
    lines = [
        "### System\n",
        system_prompt,
        "\n\n### User\n",
        user_content,
        "\n\n### Tool trace\n",
    ]
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                lines.append(f"{tc.get('name')}: {tc.get('args')}\n")
    return "".join(lines)


def _review_from_dict(data: dict[str, Any]) -> RuleCriticReview | None:
    if "approved" not in data:
        return None
    try:
        return RuleCriticReview(
            approved=bool(data["approved"]),
            issues=[str(x) for x in data.get("issues", [])]
            if isinstance(data.get("issues"), list)
            else [],
            correction_instructions=str(data.get("correction_instructions") or ""),
            rule_hits=[str(x) for x in data.get("rule_hits", [])]
            if isinstance(data.get("rule_hits"), list)
            else [],
            raw_notes=data.get("raw_notes")
            if data.get("raw_notes") is None or isinstance(data.get("raw_notes"), str)
            else str(data.get("raw_notes")),
        )
    except Exception:
        return None


def _require_stream_model(model: BaseChatModel) -> None:
    if not callable(getattr(model, "astream", None)):
        raise RuntimeError(
            "RuleCriticAgent 需要支持流式 (astream) 的 Chat 模型；当前模型不可用。"
        )


class RuleCriticAgent:
    """规则判官：仅流式 create_agent；记忆/决策 phase 由 runtime context + middleware 切换。"""

    def __init__(self, model=None):
        ms, mu = load_rule_critic_memory_prompts()
        ds, du = load_rule_critic_decision_prompts()
        self._review_memory_system_prompt = ms
        self._review_memory_user_template = mu
        self._review_decision_system_prompt = ds
        self._review_decision_user_template = du
        stream_model = model if model is not None else get_rule_critic_chat_model()
        if not isinstance(stream_model, BaseChatModel):
            raise TypeError("RuleCriticAgent model must be a BaseChatModel instance")
        _require_stream_model(stream_model)
        self.model = stream_model
        self.agent = create_agent(
            model=self.model,
            tools=[rag_query],
            system_prompt=self._review_memory_system_prompt,
            context_schema=RuleCriticContext,
            middleware=[
                situation_sketch_model_context,
                rule_critic_phase_prompt_middleware(ms, ds),
                monitor_tool,
                before_agent_middleware,
            ],
        )
        fallback_model = model if model is not None else get_rule_critic_fallback_chat_model()
        if not isinstance(fallback_model, BaseChatModel):
            raise TypeError("RuleCriticAgent fallback model must be a BaseChatModel instance")
        self.fallback_model = fallback_model
        self.fallback_agent = create_agent(
            model=self.fallback_model,
            tools=[rag_query],
            system_prompt=self._review_memory_system_prompt,
            context_schema=RuleCriticContext,
            middleware=[
                situation_sketch_model_context,
                rule_critic_phase_prompt_middleware(ms, ds),
                monitor_tool,
                before_agent_middleware,
            ],
        )

    async def _stream_collect_messages(
        self,
        user_content: str,
        phase: Literal["memory", "decision"],
        *,
        situation_sketch: SituationSketch | None = None,
        situation_sketch_narrative: str | None = None,
    ) -> list[BaseMessage]:
        invoke_config = {"recursion_limit": _recursion_limit()}
        input_state = {"messages": [HumanMessage(content=user_content)]}
        phase_ctx = RuleCriticContext(
            phase=phase,
            situation_sketch=situation_sketch,
            situation_sketch_narrative=situation_sketch_narrative,
        )
        messages: list[BaseMessage] | None = None
        try:
            # 异步持续产出大量事件（输出chunk/调用工具...）
            async for event in self.agent.astream_events(
                input_state,
                invoke_config,
                version="v2",
                context=phase_ctx,
            ):
                et = event.get("event")
                if et == "on_chain_end":
                    out = (event.get("data") or {}).get("output")
                    if isinstance(out, dict) and out.get("messages") is not None:
                        messages = list(out["messages"])
        except Exception as e:
            logger.error("RuleCriticAgent: astream_events failed: %s", e)
            logger.warning("RuleCriticAgent: falling back to ainvoke")
            try:
                result = await self.fallback_agent.ainvoke(
                    input_state, invoke_config, context=phase_ctx
                )
                messages = list(result.get("messages") or [])
            except Exception as fallback_error:
                raise RuntimeError(
                    f"规则判官失败：流式错误={e}；非流式回退错误={fallback_error}"
                ) from fallback_error
        if messages is None:
            raise RuntimeError(
                "规则判官流未返回最终 messages，请检查 LangGraph 与模型流式配置。"
            )
        return messages

    async def review_memory(
        self,
        summary: MemorySummary,
        last_ingestion: IngestionOutput | None,
        *,
        situation_sketch: SituationSketch | None = None,
        situation_sketch_narrative: str | None = None,
    ) -> RuleCriticReview:
        if not is_rule_critic_enabled():
            return RuleCriticReview(
                approved=True,
                issues=[],
                correction_instructions="",
                rule_hits=[],
                raw_notes=None,
            )

        logger.info(
            "[rule critic agent] 开始记忆判官: %s",
            situation_sketch.model_dump() if situation_sketch is not None else None,
        )
        
        timeout = _timeout_seconds("memory")
        mem_json = json.dumps(
            summary.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
        ing_json = (
            last_ingestion.model_dump_json(indent=2, ensure_ascii=False)
            if last_ingestion
            else "{}"
        )
        user_content = (
            self._review_memory_user_template.replace("{memory_summary_json}", mem_json)
            .replace("{last_ingestion_json}", ing_json)
        )

        try:
            messages = await asyncio.wait_for(
                self._stream_collect_messages(
                    user_content,
                    "memory",
                    situation_sketch=situation_sketch,
                    situation_sketch_narrative=situation_sketch_narrative,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("RuleCriticAgent.review_memory: stream timeout after %ss", timeout)
            return RuleCriticReview(
                approved=True,
                issues=[],
                correction_instructions="",
                rule_hits=[],
                raw_notes="规则判官不可用，跳过修订：timeout",
            )
        except RuntimeError as e:
            logger.warning("RuleCriticAgent.review_memory: %s", e)
            return RuleCriticReview(
                approved=True,
                issues=[],
                correction_instructions="",
                rule_hits=[],
                raw_notes=f"规则判官不可用，跳过修订：{e}",
            )

        raw = last_ai_text(messages)
        data = _parse_json_object(raw)
        if not data:
            return RuleCriticReview(
                approved=True,
                issues=["无法解析判官 JSON"],
                correction_instructions="",
                rule_hits=[],
                raw_notes="判官输出无法解析，跳过修订",
            )

        rev = _review_from_dict(data)
        if rev is None:
            return RuleCriticReview(
                approved=True,
                issues=[],
                correction_instructions="",
                rule_hits=[],
                raw_notes="判官 JSON 字段不完整，跳过修订",
            )
        return rev

    async def review_decision_stream(
        self,
        result: DecisionResult,
        iteration: int,
        situation_sketch: SituationSketch | None = None,
        situation_sketch_narrative: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """决策判官流式：yield 增量事件，最后 yield type=critic_complete。"""
        if not is_rule_critic_enabled():
            rev = RuleCriticReview(
                approved=True,
                issues=[],
                correction_instructions="",
                rule_hits=[],
                raw_notes=None,
            )
            yield {
                "type": "critic_complete",
                "review": rev.model_dump(),
                "iteration": iteration,
                "rule_critic_debug_prompt": None,
            }
            return

        logger.info(
            "[rule critic agent] 开始决策判官: %s",
            situation_sketch.model_dump() if situation_sketch is not None else None,
        )

        user_content = (
            self._review_decision_user_template
            .replace("{original_context}", result.debug_prompt or "")
            .replace("{prior_speech_analysis}", result.prior_speech_analysis or "")
            .replace("{identity_inference}", result.identity_inference or "")
            .replace("{speech_suggestion}", result.speech_suggestion or "")
        )
        invoke_config = {"recursion_limit": _recursion_limit()}
        input_state = {"messages": [HumanMessage(content=user_content)]}
        phase_ctx = RuleCriticContext(
            phase="decision",
            situation_sketch=situation_sketch,
            situation_sketch_narrative=situation_sketch_narrative,
        )

        messages: list[BaseMessage] | None = None
        try:
            async for event in self.agent.astream_events(
                input_state,
                invoke_config,
                version="v2",
                context=phase_ctx,
            ):
                et = event.get("event")
                if et == "on_chat_model_stream":
                    chunk = (event.get("data") or {}).get("chunk")
                    for kind, delta in stream_deltas_from_chunk(chunk):
                        yield {
                            "type": kind,
                            "delta": delta,
                            "iteration": iteration,
                            "phase": "critic",
                        }
                elif et == "on_tool_start":
                    name = (event.get("data") or {}).get("name") or event.get("name")
                    yield {
                        "type": "tool_start",
                        "name": str(name or ""),
                        "iteration": iteration,
                        "phase": "critic",
                    }
                elif et == "on_tool_end":
                    name = (event.get("data") or {}).get("name") or event.get("name")
                    yield {
                        "type": "tool_end",
                        "name": str(name or ""),
                        "iteration": iteration,
                        "phase": "critic",
                    }
                elif et == "on_chain_end":
                    out = (event.get("data") or {}).get("output")
                    if isinstance(out, dict) and out.get("messages") is not None:
                        messages = list(out["messages"])
        except Exception as e:
            logger.error("RuleCriticAgent.review_decision_stream: astream_events failed: %s", e)
            logger.warning(
                "RuleCriticAgent.review_decision_stream: falling back to ainvoke"
            )
            try:
                fallback_result = await self.fallback_agent.ainvoke(
                    input_state, invoke_config, context=phase_ctx
                )
                messages = list(fallback_result.get("messages") or [])
                yield {
                    "type": "content",
                    "delta": "\n[判官流式输出失败，已切换为非流式推理]\n",
                    "iteration": iteration,
                    "phase": "critic",
                }
            except Exception as fallback_error:
                raise RuntimeError(
                    f"规则判官决策失败：流式错误={e}；非流式回退错误={fallback_error}"
                ) from fallback_error

        if messages is None:
            raise RuntimeError(
                "规则判官决策流未返回最终 messages，请检查 LangGraph 与模型流式配置。"
            )

        debug_prompt = _build_review_debug_prompt(
            self._review_decision_system_prompt, user_content, messages
        )
        raw = last_ai_text(messages)
        data = _parse_json_object(raw)
        if not data:
            rev = RuleCriticReview(
                approved=True,
                issues=[],
                correction_instructions="",
                rule_hits=[],
                raw_notes="判官输出无法解析，跳过修订",
            )
            yield {
                "type": "critic_complete",
                "review": rev.model_dump(),
                "iteration": iteration,
                "rule_critic_debug_prompt": debug_prompt,
            }
            return

        rev = _review_from_dict(data)
        if rev is None:
            rev = RuleCriticReview(
                approved=True,
                issues=[],
                correction_instructions="",
                rule_hits=[],
                raw_notes="判官 JSON 字段不完整，跳过修订",
            )
        yield {
            "type": "critic_complete",
            "review": rev.model_dump(),
            "iteration": iteration,
            "rule_critic_debug_prompt": debug_prompt,
        }
