"""Agents."""
from langchain_core.messages import AIMessage, BaseMessage
from typing import Any

def last_ai_text(messages: list[BaseMessage]) -> str:
    """与 decision_agent 一致：取最后一条无 tool_calls 的 AIMessage 文本。"""
    for m in reversed(messages):
        if not isinstance(m, AIMessage):
            continue
        if m.tool_calls:
            continue
        if isinstance(m.content, str) and m.content.strip():
            return m.content.strip()
    for m in reversed(messages):
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            return m.content.strip()
    return ""

def stream_deltas_from_chunk(chunk: Any) -> list[tuple[str, str]]:
    """从流式 chunk 拆出 (事件类型, 文本增量)"""
    out: list[tuple[str, str]] = []
    if chunk is None:
        return out
    ak = getattr(chunk, "additional_kwargs", None) or {}
    if isinstance(ak, dict):
        for k in ("reasoning_content", "thinking"):
            v = ak.get(k)
            if isinstance(v, str) and v:
                out.append(("thinking", v))
                break
    content = getattr(chunk, "content", None)
    if isinstance(content, str) and content:
        out.append(("content", content))
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            t = str(block.get("type") or "")
            txt = block.get("text") or block.get("content")
            if not isinstance(txt, str) or not txt:
                continue
            if "think" in t.lower() or "reason" in t.lower():
                out.append(("thinking", txt))
            else:
                out.append(("content", txt))
    return out

def merge_notes(a: str | None, b: str) -> str:
    a = (a or "").strip()
    b = (b or "").strip()
    if a and b:
        return f"{a} | {b}"
    return a or b