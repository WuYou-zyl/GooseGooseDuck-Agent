"""LangChain Agent 中间件：按运行时 context 切换 system prompt（草案/修订、记忆判官/决策判官）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from langchain.agents import AgentState
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
    before_agent,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from backend.schemas.graph_state import SituationSketch
from backend.utils.logger_handler import logger


def format_situation_sketch_for_model(
    sketch: SituationSketch | None,
    narrative: str | None,
) -> str:
    parts: list[str] = []
    if sketch is not None:
        parts.append("【结构化局势】\n" + sketch.model_dump_json(indent=2))
    if narrative and str(narrative).strip():
        parts.append("【局势笔记】\n" + str(narrative).strip())
    return "\n\n".join(parts)


def _merge_extra_into_system_message(
    request: ModelRequest, extra_text: str
) -> ModelRequest:
    sm = request.system_message
    prev = sm.content
    if isinstance(prev, str):
        content = prev
    else:
        parts: list[str] = []
        for block in prev if isinstance(prev, list) else [prev]:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(str(block))
        content = "\n".join(part for part in parts if part)
    new_sm = SystemMessage(content=content + "\n\n" + extra_text)
    return request.override(system_message=new_sm)


@wrap_model_call
async def situation_sketch_model_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    ctx = request.runtime.context
    sk = getattr(ctx, "situation_sketch", None)
    nar = getattr(ctx, "situation_sketch_narrative", None)
    sketch_obj = sk if isinstance(sk, SituationSketch) else None
    narrative_str = nar if isinstance(nar, str) else None
    blob = format_situation_sketch_for_model(sketch_obj, narrative_str)
    if not blob.strip():
        return await handler(request)
    return await handler(_merge_extra_into_system_message(request, blob))


def _tool_call_name_args(req: ToolCallRequest) -> tuple[str, object]:
    """兼容 dict / 对象，避免 tool_call 结构差异导致 KeyError。"""
    tc = getattr(req, "tool_call", None)
    if tc is None:
        return "?", {}
    if isinstance(tc, dict):
        name = tc.get("name")
        if name is None and isinstance(tc.get("function"), dict):
            name = tc["function"].get("name")
        args = tc.get("args")
        if args is None and "arguments" in tc:
            args = tc.get("arguments")
        return str(name or "?"), args if args is not None else {}
    name = getattr(tc, "name", None) or getattr(tc, "id", None)
    args = getattr(tc, "args", None)
    return str(name or "?"), args if args is not None else {}


@wrap_tool_call
async def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """监控工具调用，并记录工具调用日志。"""

    tname, targs = _tool_call_name_args(request)
    logger.info(f"[tool monitor] 执行工具: {tname}")
    logger.info(f"[tool monitor] 工具参数: {targs}")

    try:
        result = await handler(request)
        logger.info(f"[tool monitor] 工具 {tname} 执行成功")
        return result
    except Exception as e:
        logger.error(f"工具{tname}调用失败，原因：{str(e)}")
        raise e


@dataclass
class MemoryAgentContext:
    """MemoryAgent：draft=摘要草案，revise=按判官意见修订。"""

    phase: Literal["draft", "revise"] = "draft"
    situation_sketch: SituationSketch | None = None
    situation_sketch_narrative: str | None = None


@dataclass
class DecisionAgentContext:
    """DecisionAgent：draft=首轮决策，revise=按判官修订三段输出。"""

    phase: Literal["draft", "revise"] = "draft"
    situation_sketch: SituationSketch | None = None
    situation_sketch_narrative: str | None = None


@dataclass
class RuleCriticContext:
    """RuleCriticAgent：memory=记忆判官，decision=决策判官。"""

    phase: Literal["memory", "decision"] = "memory"
    situation_sketch: SituationSketch | None = None
    situation_sketch_narrative: str | None = None


def _agent_log_label(runtime: Runtime) -> str:
    """LangChain AgentState 通常无 name；用语境类型 + phase 区分。"""
    ctx = getattr(runtime, "context", None)
    if isinstance(ctx, MemoryAgentContext):
        return f"memory_agent:{ctx.phase}"
    if isinstance(ctx, DecisionAgentContext):
        return f"decision_agent:{ctx.phase}"
    if isinstance(ctx, RuleCriticContext):
        return f"rule_critic:{ctx.phase}"
    if ctx is not None:
        ph = getattr(ctx, "phase", None)
        if ph is not None:
            return f"context:{ph}"
    name = getattr(runtime, "name", None)
    if name:
        return str(name)
    return "agent"


@before_agent
async def before_agent_middleware(state: AgentState, runtime: Runtime) -> None:
    """在 Agent 执行前记录日志。"""
    fallback = getattr(state, "agent_name", None) or getattr(state, "name", None)
    label = fallback if fallback else _agent_log_label(runtime)
    logger.info(f"[agent before] 执行 Agent: {label}")
    return None


def _phase_str(ctx: object, attr: str = "phase") -> str | None:
    if ctx is None:
        return None
    if isinstance(ctx, (MemoryAgentContext, DecisionAgentContext, RuleCriticContext)):
        return ctx.phase
    p = getattr(ctx, attr, None)
    return str(p) if p is not None else None


def memory_phase_prompt_middleware(draft_system: str, revise_system: str):
    logger.info(f"[memory phase prompt middleware] 选择阶段: draft_system 或 revise_system")
    @dynamic_prompt
    def _select(request: ModelRequest) -> str:
        if _phase_str(request.runtime.context) == "revise":
            logger.info("memory agent 修订阶段")
            return revise_system
        logger.info("memory agent 起草阶段")
        return draft_system

    return _select


def decision_phase_prompt_middleware(draft_system: str, revise_system: str):
    logger.info(f"[decision phase prompt middleware] 选择阶段: {draft_system} 或 {revise_system}")
    @dynamic_prompt
    def _select(request: ModelRequest) -> str:
        if _phase_str(request.runtime.context) == "revise":
            logger.info("decision agent 修订阶段")
            return revise_system
        logger.info("decision agent 起草阶段")
        return draft_system

    return _select


def rule_critic_phase_prompt_middleware(memory_system: str, decision_system: str):
    logger.info(f"[rule critic phase prompt middleware] 选择阶段: {memory_system} 或 {decision_system}")
    @dynamic_prompt
    def _select(request: ModelRequest) -> str:
        if _phase_str(request.runtime.context) == "decision":
            logger.info("决策判官阶段")
            return decision_system
        logger.info("记忆判官阶段")
        return memory_system

    return _select
