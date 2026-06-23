"""
LangGraph：memory_draft →（条件）→ rule_review_memory ↔ memory_revise（判官循环直至通过或达上限）。

判官总开关在 `memory_draft` 之后的条件边上短路；`rule_review_memory` 之后条件边负责通过 / 达上限 / 修订。

决策：`run_draft_stream` → `run_decision_critic_loop_stream`（判官与修订均支持 SSE 增量）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from backend.agents.decision_agent import DecisionAgent
from backend.agents.memory_agent import MemoryAgent
from backend.agents.rule_critic_agent import RuleCriticAgent, is_rule_critic_enabled
from backend.schemas.contract import IngestionOutput, MemorySummary
from backend.schemas.decision import DecisionResult, RuleCriticReview
from backend.schemas.graph_state import MemoryDecisionState, SituationSketch
from backend.utils.config_handler import agent_conf
from backend.utils.logger_handler import logger
from backend.utils.situation_context import (
    get_situation_sketch,
    get_situation_sketch_narrative,
)
from backend.agents import merge_notes


class MemoryGraph:
    """
    START → memory_draft →（条件）→ rule_review_memory →（条件）→ END | memory_revise → rule_review_memory …
    """

    def __init__(self, checkpointer):
        self.checkpointer = checkpointer
        self.memory_agent = MemoryAgent()
        self.decision_agent = DecisionAgent()
        self.rule_critic_agent = RuleCriticAgent()
        self.graph = self._build()

    def _route_after_memory_draft(self, state: MemoryDecisionState) -> str:
        if not state.summary or not state.ingestions:
            logger.warning("MemoryGraph: no summary or ingestions")
            return END
        if not is_rule_critic_enabled():
            return END
        return "rule_review_memory"

    def _route_after_memory_review(self, state: MemoryDecisionState) -> str:
        rc = agent_conf.get("rule_critic") or {}
        review = state.memory_critic_review
        if review is None:
            return END
        if review.approved:
            return END
        max_iter = int(rc.get("memory_max_iterations", rc.get("max_iterations", 3)))
        if state.memory_revision_attempts >= max_iter:
            return END
        return "memory_revise"

    def _rule_review_memory_node(self):
        agent = self.rule_critic_agent

        async def node(state: MemoryDecisionState) -> dict:
            if not state.summary or not state.ingestions:
                return {}
            if not is_rule_critic_enabled():
                return {}
            last = state.ingestions[-1]
            review = await agent.review_memory(
                state.summary,
                last,
                situation_sketch=get_situation_sketch(state),
                situation_sketch_narrative=get_situation_sketch_narrative(state),
            )
            out: dict[str, Any] = {"memory_critic_review": review}
            rc = agent_conf.get("rule_critic") or {}
            max_iter = int(rc.get("memory_max_iterations", rc.get("max_iterations", 3)))
            if (
                state.summary
                and not review.approved
                and state.memory_revision_attempts >= max_iter
            ):
                note = "规则判官：已达最大修订轮数仍未通过"
                out["summary"] = state.summary.model_copy(
                    update={
                        "rule_critic_notes": merge_notes(
                            state.summary.rule_critic_notes, note
                        )
                    }
                )
            return out

        return node

    def _memory_revise_node(self):
        agent = self.memory_agent

        async def node(state: MemoryDecisionState) -> dict:
            if (
                not state.summary
                or not state.memory_critic_review
                or not state.ingestions
            ):
                return {}
            summary = await agent.revise_from_critic(
                state.summary,
                state.memory_critic_review,
                state.ingestions,
                situation_sketch=get_situation_sketch(state),
                situation_sketch_narrative=get_situation_sketch_narrative(state),
            )
            return {
                "summary": summary,
                "memory_revision_attempts": state.memory_revision_attempts + 1,
            }

        return node

    def _memory_draft_node(self):
        agent = self.memory_agent

        async def node(state: MemoryDecisionState) -> dict:
            """首轮草稿：为本条 ingestion 追加 PlayerState。"""
            if not state.ingestions:
                return {}
            prior = state.summary or MemorySummary(
                session_id=state.ingestions[0].session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            summary = await agent.process_draft(
                state.ingestions,
                prior,
                situation_sketch=get_situation_sketch(state),
                situation_sketch_narrative=get_situation_sketch_narrative(state),
            )
            return {"summary": summary}

        return node

    def _build(self):
        builder = StateGraph(MemoryDecisionState)
        builder.add_node("memory_draft", self._memory_draft_node())
        builder.add_node("rule_review_memory", self._rule_review_memory_node())
        builder.add_node("memory_revise", self._memory_revise_node())
        builder.add_edge(START, "memory_draft")
        builder.add_conditional_edges(
            "memory_draft",
            self._route_after_memory_draft,
            {
                END: END,
                "rule_review_memory": "rule_review_memory",
            },
        )
        builder.add_conditional_edges(
            "rule_review_memory",
            self._route_after_memory_review,
            {
                END: END,
                "memory_revise": "memory_revise",
            },
        )
        builder.add_edge("memory_revise", "rule_review_memory")
        return builder.compile(checkpointer=self.checkpointer)

    async def ainvoke(self, ingestion: IngestionOutput, thread_id: str) -> Any:
        """每条发言：重置本轮修订计数，跑草稿与判官循环。"""
        config = {"configurable": {"thread_id": thread_id}}
        return await self.graph.ainvoke(
            {
                "ingestions": [ingestion],
                "memory_revision_attempts": 0,
                "memory_critic_review": None,
            },
            config,
        )

    async def run_decision_critic_loop_stream(
        self,
        draft_result: DecisionResult,
        situation_sketch: SituationSketch | None = None,
        situation_sketch_narrative: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """规则判官 + 修订循环的流式事件，最后 yield decision_critic_done（含最终 DecisionResult）。"""
        rc = agent_conf.get("rule_critic") or {}
        max_iter = int(rc.get("decision_max_iterations", rc.get("max_iterations", 3)))
        result = draft_result

        if not is_rule_critic_enabled():
            yield {
                "type": "decision_critic_done",
                "result": result.model_dump(),
                "phase": "critic",
            }
            return

        revision_attempts = 0
        while True:
            yield {
                "type": "critic_round_start",
                "phase": "critic",
                "iteration": revision_attempts,
            }
            review: RuleCriticReview | None = None
            dbg: str | None = None
            async for ev in self.rule_critic_agent.review_decision_stream(
                result,
                revision_attempts,
                situation_sketch=situation_sketch,
                situation_sketch_narrative=situation_sketch_narrative,
            ):
                yield ev
                if ev.get("type") == "critic_complete":
                    review = RuleCriticReview.model_validate(ev["review"])
                    dbg = ev.get("rule_critic_debug_prompt")

            if review is None:
                yield {
                    "type": "decision_critic_done",
                    "result": result.model_dump(),
                    "phase": "critic",
                }
                return

            result = result.model_copy(
                update={
                    "rule_hits": review.rule_hits,
                    "rule_critic_notes": review.raw_notes or result.rule_critic_notes,
                    "rule_critic_debug_prompt": dbg,
                }
            )
            if review.approved:
                break
            if revision_attempts >= max_iter:
                w = list(result.warnings)
                w.append("规则判官：已达最大修订轮数仍未通过")
                result = result.model_copy(update={"warnings": w})
                break

            yield {
                "type": "revise_round_start",
                "phase": "revise",
                "iteration": revision_attempts,
            }
            async for ev in self.decision_agent.revise_from_critic_stream(
                result,
                review,
                iteration=revision_attempts,
                situation_sketch=situation_sketch,
                situation_sketch_narrative=situation_sketch_narrative,
            ):
                yield ev
                if ev.get("type") == "revise_complete" and isinstance(
                    ev.get("result"), dict
                ):
                    result = DecisionResult.model_validate(ev["result"])

            revision_attempts += 1

        yield {
            "type": "decision_critic_done",
            "result": result.model_dump(),
            "phase": "critic",
        }

    async def persist_decision_result(
        self, thread_id: str, result: DecisionResult
    ) -> None:
        config = {"configurable": {"thread_id": thread_id}}
        update_fn = getattr(self.graph, "aupdate_state", None)
        if update_fn is not None:
            await update_fn(
                config,
                {"decision_result": result},
                as_node="memory_draft",
            )
        else:
            self.graph.update_state(
                config,
                {"decision_result": result},
                as_node="memory_draft",
            )


MemoryDecisionGraph = MemoryGraph
