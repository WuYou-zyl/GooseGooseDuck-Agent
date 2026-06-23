"""为给定 thread_id 确保存在 LangGraph state（便于首条发言前写局势/合并名单）。"""

from __future__ import annotations

from typing import Any

from backend import app_state
from backend.schemas.graph_state import MemoryDecisionState


async def ensure_session_state(session_id: str) -> None:
    """若该会话尚无 state，则写入默认 MemoryDecisionState。"""
    g = app_state.graph
    if g is None:
        return
    config: dict[str, Any] = {"configurable": {"thread_id": session_id}}
    snap = await g.graph.aget_state(config)
    if snap.values:
        return
    payload = MemoryDecisionState().model_dump()
    update_fn = getattr(g.graph, "aupdate_state", None)
    if update_fn is not None:
        await update_fn(config, payload, as_node="memory_draft")
    else:
        g.graph.update_state(config, payload, as_node="memory_draft")


async def ensure_session_checkpoint(session_id: str) -> None:
    """兼容旧调用名：等价于 ensure_session_state。"""
    await ensure_session_state(session_id)
