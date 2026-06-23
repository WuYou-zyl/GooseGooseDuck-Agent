"""Memory + decision API (/api/v1/*)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend import app_state
from backend.schemas.contract import IngestionOutput
from backend.schemas.decision import DecisionResult
from backend.schemas.graph_state import MemoryDecisionState, SituationSketch
from backend.utils.color_roster_defaults import SEAT_COLORS
from backend.utils.meeting_roles import load_meeting_role_names
from backend.utils.session_checkpoint import ensure_session_state

logger = logging.getLogger("ggd-b")

_DECISION_NO_CHECKPOINT = (
    "本会话尚无记忆 state。请先「开始新一局」，再「开始监控新一轮会议」，或确认 session_id 与本局一致。"
)
_DECISION_NO_SUMMARY = (
    "尚无记忆摘要：请在本局监控中至少产生一条已入库的发言（ASR 读入成功后后台会更新摘要），"
    "并确认推理使用的 session 与本局 game session 相同；「开始推理」不会单独触发记忆 Agent。"
)
_DECISION_NO_INGESTIONS_FILTER = "所选发言人为空：没有匹配的读入记录，请调整筛选或勾选玩家。"

router = APIRouter(tags=["memory"])


async def execute_decision_stream(
    session_id: str,
    *,
    extra: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """流式：草稿 + 规则判官 + 修订全阶段；最后 yield type=done（含最终 DecisionResult）。

    与 DecisionAgent.run_draft_stream、MemoryGraph.run_decision_critic_loop_stream 签名对齐。
    """
    g = app_state.graph
    if g is None:
        raise HTTPException(status_code=503, detail="MemoryGraph not initialized")

    speaker_filter = extra.get("speaker_filter") if extra else None

    try:
        config = {"configurable": {"thread_id": session_id}}
        snap = await g.graph.aget_state(config)
        raw = snap.values
        if not raw:
            raise HTTPException(status_code=400, detail=_DECISION_NO_CHECKPOINT)

        state = MemoryDecisionState.model_validate(raw)
        if not state.summary:
            raise HTTPException(status_code=400, detail=_DECISION_NO_SUMMARY)

        ingestions = state.ingestions
        latest_meeting_id = next(
            (i.meeting_id for i in reversed(ingestions) if i.meeting_id),
            None,
        )
        if latest_meeting_id:
            ingestions = [i for i in ingestions if i.meeting_id == latest_meeting_id]
        if speaker_filter and isinstance(speaker_filter, list):
            ingestions = [
                i
                for i in ingestions
                if i.metadata.get("speaker_id") in speaker_filter
            ]
            if not ingestions:
                raise HTTPException(
                    status_code=400,
                    detail="No ingestions match the selected speakers.",
                )

        draft_final: DecisionResult | None = None
        async for ev in g.decision_agent.run_draft_stream(
            state.summary,
            recent_ingestions=ingestions,
            situation_sketch=state.situation_sketch,
            situation_sketch_narrative=state.situation_sketch_narrative,
        ):
            if ev.get("type") == "draft_complete":
                raw_dr = ev.get("result")
                draft_final = (
                    DecisionResult.model_validate(raw_dr)
                    if isinstance(raw_dr, dict)
                    else raw_dr
                )
            out = dict(ev)
            if "phase" not in out:
                out["phase"] = "draft"
            yield out

        if draft_final is None:
            raise HTTPException(
                status_code=500, detail="Draft stream produced no result."
            )

        # 草稿阶段可能较久；此处重新读取 checkpoint，确保判官与修订使用最新局势（含刚保存的 roster）
        snap_critic = await g.graph.aget_state(config)
        if snap_critic.values:
            state = MemoryDecisionState.model_validate(snap_critic.values)

        final = draft_final
        async for ev in g.run_decision_critic_loop_stream(
            draft_final,
            situation_sketch=state.situation_sketch,
            situation_sketch_narrative=state.situation_sketch_narrative,
        ):
            yield ev
            if ev.get("type") == "decision_critic_done" and isinstance(
                ev.get("result"), dict
            ):
                final = DecisionResult.model_validate(ev["result"])

        await g.persist_decision_result(session_id, final)
        yield {"type": "done", "result": final.model_dump()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Decision stream failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/v1/status")
async def get_status_v1(
    session_id: str | None = Query(
        None,
        description="若提供，则附带该会话的轻量局势笔记等 graph_state 字段",
    ),
):
    base: dict = {"status": "running" if app_state.graph else "not_initialized"}
    if session_id and app_state.graph:
        config = {"configurable": {"thread_id": session_id}}
        try:
            snap = await app_state.graph.graph.aget_state(config)
            raw = snap.values
            if raw:
                st = MemoryDecisionState.model_validate(raw)
                base["situation_sketch"] = st.situation_sketch
                base["situation_sketch_narrative"] = st.situation_sketch_narrative
                base["sketch_updated_at"] = st.sketch_updated_at
                base["sketch_warnings"] = st.sketch_warnings
                base["ingestion_count"] = len(st.ingestions)
                base["has_summary"] = st.summary is not None
                base["summary_timestamp"] = (
                    st.summary.timestamp if st.summary is not None else None
                )
        except Exception:
            logger.debug("status: no state for session_id=%s", session_id, exc_info=True)
    return base


class SituationSketchPutBody(BaseModel):
    session_id: str = Field(..., description="LangGraph thread_id / 会话 ID")
    situation_sketch: SituationSketch = Field(..., description="完整结构化局势（覆盖写入）")


@router.get("/api/v1/game-catalog")
async def get_game_catalog():
    """座位默认颜色与 meeting_game_context 角色目录（供前端局势编辑）。"""
    return {
        "seat_colors": list(SEAT_COLORS),
        "role_names": list(load_meeting_role_names()),
    }


@router.put("/api/v1/situation-sketch")
async def put_situation_sketch(body: SituationSketchPutBody):
    """将客户端编辑后的局势写入 graph_state（不改 agents/schema 定义）。"""
    g = app_state.graph
    if g is None:
        raise HTTPException(status_code=503, detail="MemoryGraph not initialized")
    await ensure_session_state(body.session_id)
    config = {"configurable": {"thread_id": body.session_id}}
    snap = await g.graph.aget_state(config)
    raw = snap.values
    if not raw:
        raise HTTPException(
            status_code=400,
            detail=_DECISION_NO_CHECKPOINT,
        )
    MemoryDecisionState.model_validate(raw)
    ts = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "situation_sketch": body.situation_sketch,
        "sketch_updated_at": ts,
    }
    update_fn = getattr(g.graph, "aupdate_state", None)
    if update_fn is not None:
        await update_fn(config, payload, as_node="memory_draft")
    else:
        g.graph.update_state(config, payload, as_node="memory_draft")
    return {"status": "ok", "session_id": body.session_id, "sketch_updated_at": ts}


# @router.post("/api/v1/ingestion")
# async def receive_ingestion(ingestion: IngestionOutput):
#     """Receive an IngestionOutput and feed it into MemoryGraph."""
#     g = app_state.graph
#     if g is None:
#         raise HTTPException(status_code=503, detail="MemoryGraph not initialized")
#     try:
#         await g.ainvoke(ingestion, ingestion.session_id)
#     except Exception as e:
#         logger.error("Failed to process ingestion: %s", e, exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e)) from e
#     logger.info("Ingestion processed: session=%s seq=%s", ingestion.session_id, ingestion.sequence_id)
#     return {"status": "ok", "session_id": ingestion.session_id}

