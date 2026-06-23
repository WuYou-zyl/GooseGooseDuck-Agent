import os

from backend.utils.config_handler import prompts_conf
from backend.utils.path_tool import get_abs_path
from backend.utils.logger_handler import logger


def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_conf['rag_summarize_prompt_path'])
    except KeyError as e:
        logger.error(f"[load_rag_prompts] yaml config file doesn't have key {e}")
        raise e
    
    try:
        return open(rag_prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[load_rag_prompts] Failed to load rag prompts from {rag_prompt_path}")
        raise e
    
def load_summarize_prompt() -> tuple[str, str]:
    """加载 memory agent 的 system 与 user 模板。

    user 模板须含占位符：{prior_memory_summary}、{short_memory}、{ingestion}（由 MemoryAgent 做 replace 注入）。
    """
    try:
        system_path = get_abs_path(prompts_conf['memory_agent_system_prompt_path'])
        user_path = get_abs_path(prompts_conf['memory_agent_user_prompt_path'])
    except Exception as e:
        logger.error(f"[load_summarize_prompt] Failed to load memory agent prompts from {system_path} and {user_path}")
        raise e
    try:
        system_text = open(system_path, "r", encoding="utf-8").read()
        user_text = open(user_path, "r", encoding="utf-8").read()
    except OSError as e:
        logger.error(f"[load_summarize_prompt] yaml config file doesn't have key {e}")
        raise e
    return system_text, user_text

def load_decision_prompts() -> tuple[str, str]:
    """加载决策 Agent 的 system 与 user 模板（user 侧含 {decision_context} 等占位符）。"""
    try:
        system_path = get_abs_path(prompts_conf["decision_agent_system_prompt_path"])
        user_path = get_abs_path(prompts_conf["decision_agent_user_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_decision_prompts] yaml config file doesn't have key {e}")
        raise e
    try:
        system_text = open(system_path, "r", encoding="utf-8").read()
        user_text = open(user_path, "r", encoding="utf-8").read()
    except OSError as e:
        logger.error(f"[load_decision_prompts] Failed to load: {e}")
        raise e
    return system_text, user_text


def load_rule_critic_memory_prompts() -> tuple[str, str]:
    try:
        system_path = get_abs_path(
            prompts_conf["rule_critic_memory_system_prompt_path"]
        )
        user_path = get_abs_path(prompts_conf["rule_critic_memory_user_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rule_critic_memory_prompts] missing key {e}")
        raise e
    system_text = open(system_path, "r", encoding="utf-8").read()
    user_text = open(user_path, "r", encoding="utf-8").read()
    return system_text, user_text


def load_rule_critic_decision_prompts() -> tuple[str, str]:
    try:
        system_path = get_abs_path(
            prompts_conf["rule_critic_decision_system_prompt_path"]
        )
        user_path = get_abs_path(prompts_conf["rule_critic_decision_user_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rule_critic_decision_prompts] missing key {e}")
        raise e
    system_text = open(system_path, "r", encoding="utf-8").read()
    user_text = open(user_path, "r", encoding="utf-8").read()
    return system_text, user_text


def load_memory_revise_prompts() -> tuple[str, str]:
    try:
        system_path = get_abs_path(
            prompts_conf["memory_agent_revise_system_prompt_path"]
        )
        user_path = get_abs_path(prompts_conf["memory_agent_revise_user_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_memory_revise_prompts] missing key {e}")
        raise e
    system_text = open(system_path, "r", encoding="utf-8").read()
    user_text = open(user_path, "r", encoding="utf-8").read()
    return system_text, user_text


def load_decision_revise_prompts() -> tuple[str, str]:
    try:
        system_path = get_abs_path(
            prompts_conf["decision_agent_revise_system_prompt_path"]
        )
        user_path = get_abs_path(prompts_conf["decision_agent_revise_user_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_decision_revise_prompts] missing key {e}")
        raise e
    system_text = open(system_path, "r", encoding="utf-8").read()
    user_text = open(user_path, "r", encoding="utf-8").read()
    return system_text, user_text


def load_situation_sketch_prompts() -> tuple[str, str]:
    try:
        system_path = get_abs_path(prompts_conf["situation_sketch_system_prompt_path"])
        user_path = get_abs_path(prompts_conf["situation_sketch_user_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_situation_sketch_prompts] missing key {e}")
        raise e
    system_text = open(system_path, "r", encoding="utf-8").read()
    user_text = open(user_path, "r", encoding="utf-8").read()
    return system_text, user_text