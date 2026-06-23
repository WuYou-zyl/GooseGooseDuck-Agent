# langgraph 图架构

```python
START → memory_draft →（条件）→ rule_review_memory →（条件）→ END | memory_revise → rule_review_memory …
```

- `memory_draft` 用于起草玩家发言总结草案。
- `memory_draft` **之后** 的条件边：无摘要/发言、或规则判官关闭（`rule_critic.enabled` 为 false）时直接 `END`，不进入 `rule_review_memory`（不在此调用 `review_memory`）。
- `rule_review_memory` 根据规则判官对草案提出修改建议（单次 `review_memory` 调用）。
- `rule_review_memory` **之后** 的条件边：通过、达最大修订次数、或评审缺失时 `END`；否则进入 `memory_revise`。
- `memory_revise` 修订总结草案后回到 `rule_review_memory`，直至通过或达上限。

**决策推理**：仅提供 `POST /api/inference/stream`（SSE）；流式包含草稿、规则判官、修订各阶段事件。

## middleware 中间件

- 运行时 Context 机制：根据运行时传入的context的不同,为 decision agent 选择(起草/修订),memory agent 选择(起草/修订),critic agent 选择(记忆/决策判官)

## rule-critic-agent

## Graph State

```
ingestions->发言累积
summary->memory agent维护的摘要
decision_result->decision agent 输出
situation_sketch->局势笔记,局势笔记记录了整局游戏所有的确定性信息.(鹅鸭中立个数,已知必有角色等)
...
```

## decision agent

- 开启 thinking 模式,选择流式输出防止用户等待时间过长
- 加载 `draft` 和 `revise` 两套提示词
- 

> `MemorySummary` 中带有 `recent_events`，记录最近发生的大事，是否不需要？需要观察 services 中的 `situation_sketch`

