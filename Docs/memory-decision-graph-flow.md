# Memory Agent、决策 Agent 与 Graph 整体流程

本文描述仓库中 **记忆（Memory）**、**决策（Decision）** 与 **LangGraph（MemoryGraph）** 的职责划分与数据流，便于从入口到落盘快速建立心智模型。

---

## 1. 全局架构一览

| 组件 | 文件 | 作用 |
|------|------|------|
| **MemoryGraph** | `backend/agents/my_graph.py` | LangGraph：每条发言跑「记忆草稿 →（可选）规则判官 ↔ 记忆修订」；并封装「决策判官 + 决策修订」的**流式**循环（决策本身不在 Graph 节点里跑草稿）。 |
| **MemoryAgent** | `backend/agents/memory_agent.py` | `create_agent` + `rag_query`：为**当前条**发言更新 `MemorySummary`（草稿追加 `PlayerState`，修订只改最后一条立场）。 |
| **DecisionAgent** | `backend/agents/decision_agent.py` | 仅**流式**：`run_draft_stream` 产出三段式决策；`revise_from_critic_stream` 按判官意见修订。 |
| **RuleCriticAgent** | `backend/agents/rule_critic_agent.py` | 记忆路径：`review_memory`（可超时降级）；决策路径：`review_decision_stream`（SSE 增量）。 |
| **Checkpoint 状态** | `backend/schemas/graph_state.py` | `MemoryDecisionState`：累积 `ingestions`、当前 `summary`、`situation_sketch*`、记忆判官相关字段等。 |

应用启动时在 `main.py` 中用 **AsyncSqliteSaver** 初始化 `MemoryGraph`，`thread_id` 通常等于 `session_id`，实现按会话持久化。

---

## 2. LangGraph：记忆路径（每条发言）

**入口**：`MemoryGraph.ainvoke(ingestion, thread_id)`  
**状态**：`ingestions` 使用 `operator.add` 与 checkpoint **合并累积**；每次调用会重置 `memory_revision_attempts`、`memory_critic_review`（见 `my_graph.py`）。

```text
START
  → memory_draft          （MemoryAgent.process_draft：为本条 ingestion 追加一条 PlayerState）
  → [条件边]
        · 无 summary/ingestions 或判官关闭 → END
        · 否则 → rule_review_memory
  → [条件边]
        · review.approved → END
        · 达 memory_max_iterations → END（可能写入 rule_critic_notes）
        · 否则 → memory_revise
  → memory_revise           （MemoryAgent.revise_from_critic：改最后一条 stance，不追加 PlayerState）
  → rule_review_memory    （循环）
```

**要点**：

- **记忆判官**在 Graph **节点内**同步完成（`RuleCriticAgent.review_memory`）；关闭判官时整条路径在 `memory_draft` 后直接 `END`。
- **局势**：`MemoryDecisionState.situation_sketch` / `situation_sketch_narrative` 通过中间件注入各 Agent；叙事笔记在 ingestion 后由 `schedule_situation_sketch_after_ingestion` **防抖**异步更新 checkpoint（见 `backend/services/situation_sketch.py`）。

---

## 3. 决策路径（推理 API，非 Graph 节点草稿）

决策的**草稿与判官循环**由 `backend/routers/decision.py` 中的 `execute_decision_stream` 编排，**不**作为 LangGraph 节点执行（与文件头注释一致：`run_draft_stream` → `run_decision_critic_loop_stream`）。

```text
1. 从 checkpoint 读取 MemoryDecisionState（session_id = thread_id）
2. 校验 summary、ingestions 存在
3. DecisionAgent.run_draft_stream
      → SSE：thinking/content/tool_* 等，最后 draft_complete → DecisionResult
4. MemoryGraph.run_decision_critic_loop_stream
      → 若判官关闭：直接 decision_critic_done
      → 否则循环：
            RuleCriticAgent.review_decision_stream（流式事件）
            → 若通过或达 decision_max_iterations：结束或打 warnings
            → 否则 DecisionAgent.revise_from_critic_stream
5. persist_decision_result：把最终 DecisionResult 写入 checkpoint（as_node=memory_draft）
6. yield type=done
```

**与记忆 Graph 的关系**：决策**读取**同一会话的 `summary` + `ingestions`；写回的是 **`decision_result` 缓存**，不驱动记忆节点自动重跑。

---

## 4. 读入（Ingestion）如何进入 Graph

- **A 侧监控线程**（`backend/routers/ingestion.py`）：`_forward_ingestion_to_memory` 在主事件循环上调度 `graph.ainvoke` + `schedule_situation_sketch_after_ingestion`。
- **B 侧 HTTP**（`POST /api/v1/ingestion`）：同样 `ainvoke` + 局势笔记调度。
- **推理流式**（`POST /api/inference/stream`）：传入 `session_id` 与可选 `extra`（如 `speaker_filter`），调用 `execute_decision_stream(session_id, extra=...)`，不经过「单条 ingestion」Graph，但依赖此前已累积的 checkpoint。
- **局势写入**（`PUT /api/v1/situation-sketch`）：请求体为 `session_id` + 完整 `situation_sketch`，写入 checkpoint（与 `persist_decision_result` 相同 `aupdate_state` 方式）。
- **识别名单**（`POST /api/scan-roster?session_id=`）：可选 query；若提供 `session_id` 且已有 checkpoint，将 Gemini 名单按座位合并进 `situation_sketch.player_roster`。

---

## 5. 中间件与 Prompt 阶段

三类 Agent 均使用 `backend/agents/middleware.py`：

- **`situation_sketch_model_context`**：把结构化局势 + 局势笔记拼进 system。
- **`memory_phase_prompt_middleware` / `decision_phase_prompt_middleware` / `rule_critic_phase_prompt_middleware`**：按 runtime context 的 `phase` 切换草案/修订或记忆判官/决策判官的 system prompt。
- **`before_agent_middleware`、`monitor_tool`**：日志与工具监控。

Memory/Decision 通过各自的 `*Context` dataclass 传入 `phase` 与 `situation_sketch*`。

---

## 6. 数据小结

| 数据 | 产生位置 | 持久化 |
|------|----------|--------|
| `ingestions` | 读入 | Checkpoint 累积 |
| `MemorySummary` | MemoryAgent + 记忆判官循环 | Checkpoint |
| `situation_sketch_narrative` 等 | situation_sketch 服务 | Checkpoint |
| `DecisionResult` | DecisionAgent + 决策判官循环 | `persist_decision_result` 写入 checkpoint |

---

*文档仅描述当前代码结构，不含实现建议；分析与改进见 `code-review-redundancy-and-suggestions.md`。*
