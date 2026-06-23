from __future__ import annotations

import os
import threading
from typing import Any

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from backend.utils.config_handler import model_conf

_lock = threading.Lock()
_current_backend = "api"
_chat_cache: dict[str, BaseChatModel] = {}
_embedding_cache: Embeddings | None = None


def _api_key() -> str:
    k = model_conf.get("api_key")
    if isinstance(k, str) and k.strip():
        return k.strip()
    return os.environ.get("DASHSCOPE_API_KEY", "") or ""


def _base_url() -> str | None:
    configured = model_conf.get("base_url")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    env_url = os.environ.get("DASHSCOPE_BASE_URL", "") or ""
    return env_url.strip() or None


def _model_name(role: str) -> str:
    if role == "rag":
        return str(model_conf["rag_model_name"])
    if role == "decision":
        decision_conf = model_conf.get("decision")
        if isinstance(decision_conf, dict):
            override = decision_conf.get("model_name")
            if override is not None and str(override).strip():
                return str(override).strip()
    return str(model_conf["agent_model_name"])


def _embedding_model_name() -> str:
    embedding_conf = model_conf.get("embedding")
    if isinstance(embedding_conf, dict):
        override = embedding_conf.get("model_name")
        if override is not None and str(override).strip():
            return str(override).strip()
    return str(model_conf["embedding_model_name"])


def _model_kwargs(section: str) -> dict[str, Any]:
    sec = model_conf.get(section)
    if not isinstance(sec, dict):
        out = {"temperature": 0.7}
        if section == "rag":
            # RAG 汇总链路走非流式调用，DashScope 非流式必须关闭 thinking。
            out["enable_thinking"] = False
        return out
    out: dict[str, Any] = {}
    for key in ("enable_thinking", "temperature"):
        if key in sec:
            out[key] = sec[key]
    if "temperature" not in out:
        out["temperature"] = 0.7
    if section == "rag" and "enable_thinking" not in out:
        # 配置缺失时兜底，避免 non-streaming 调用被参数校验拦截。
        out["enable_thinking"] = False
    return out


def _build_local_model() -> BaseChatModel:
    cached = _chat_cache.get("local")
    if cached is not None:
        return cached
    from langchain_ollama import ChatOllama

    model = ChatOllama(
        model="qwen2.5:7b-instruct",
        base_url="http://localhost:11434",
        num_ctx=4096,
    )
    _chat_cache["local"] = model
    return model


def _get_chat_model(
    role: str, section: str, *, force_streaming: bool = False
) -> BaseChatModel:
    cache_key = f"{_current_backend}:{role}:{'stream' if force_streaming else 'plain'}"
    with _lock:
        if _current_backend == "local":
            return _build_local_model()
        cached = _chat_cache.get(cache_key)
        if cached is not None:
            return cached
        model = ChatTongyi(
            model=_model_name(role),
            api_key=_api_key(),
            base_url=_base_url(),
            model_kwargs=_model_kwargs(section),
            streaming=force_streaming,
        )
        _chat_cache[cache_key] = model
        return model


def get_rag_chat_model() -> BaseChatModel:
    return _get_chat_model("rag", "rag")


def get_memory_chat_model() -> BaseChatModel:
    return _get_chat_model("memory", "memory")


def get_decision_chat_model() -> BaseChatModel:
    # 决策链路为流式优先，显式打开 streaming，避免被 SDK 判定为非流式调用。
    return _get_chat_model("decision", "decision", force_streaming=True)


def get_decision_fallback_chat_model() -> BaseChatModel:
    return _get_chat_model("decision", "decision", force_streaming=False)


def get_rule_critic_chat_model() -> BaseChatModel:
    # 规则判官也依赖流式事件，保持与决策链路一致。
    return _get_chat_model("rule_critic", "rule_critic", force_streaming=True)


def get_rule_critic_fallback_chat_model() -> BaseChatModel:
    return _get_chat_model("rule_critic", "rule_critic", force_streaming=False)


def get_embedding_model() -> Embeddings:
    global _embedding_cache
    with _lock:
        if _embedding_cache is None:
            _embedding_cache = DashScopeEmbeddings(
                model=_embedding_model_name(),
                dashscope_api_key=_api_key(),
            )
        return _embedding_cache


def set_model_backend(backend: str) -> str:
    global _current_backend
    if backend not in ("api", "local"):
        raise ValueError("backend must be 'api' or 'local'")
    with _lock:
        _current_backend = backend
        _chat_cache.clear()
    return _current_backend


def get_current_backend() -> str:
    return _current_backend
