"""从 MemoryDecisionState 取局势，供 Agent Runtime context 使用。"""

from __future__ import annotations

from backend.schemas.graph_state import MemoryDecisionState, SituationSketch


def get_situation_sketch(state: MemoryDecisionState) -> SituationSketch:
    return state.situation_sketch


def get_situation_sketch_narrative(state: MemoryDecisionState) -> str | None:
    return state.situation_sketch_narrative
