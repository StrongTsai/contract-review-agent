# 合同审核数字员工 · 设计文档(架构)

> 版本 v1.1 · 2026-09-12

## 1. 总体架构

四节点流水线,含一个条件分支(人工审批):

```
extract_contract → review_risks → human_review ──确认──→ generate_report → END
    (信息抽取)      (风险审核)       (人工确认)  └──驳回──→ END
```

- 前两个节点调 LLM,`human_review` 可能 `interrupt()` 暂停等人工,`generate_report` 纯 Python 拼接。
- `human_review` 之后是**条件边**:读 `human_decision` 决定「确认 → 出报告」还是「驳回 → 结束」。
- 图编译时挂 `MemorySaver` checkpointer,`interrupt` 依赖它保存/恢复状态。

## 2. 技术选型

| 关注点 | 选型 | 理由 |
|---|---|---|
| 编排框架 | LangGraph | 状态管理 + 节点/边/条件边建模,面试目标技术栈 |
| 模型 | DeepSeek(deepseek-v4-pro) | 现有 key;thinking 关闭以省 token/提速 |
| 结构化输出 | Pydantic + `with_structured_output` | 用 JSON Schema 约束模型输出 |
| 人工审批 | `interrupt` + checkpointer | 高危动作前暂停,由真人拍板 |
| 可观测性 | LangSmith | 环境变量开启,callback 机制零侵入 |
| 配置 | python-dotenv | 从 `.env` 读环境变量 |

## 3. 状态设计(ContractState)

```python
class ContractState(MessagesState, total=False):
    contract_text: str          # 输入:合同原文
    contract_info: ContractInfo # 第 1 节点产出
    review: ReviewResult        # 第 2 节点产出
    human_decision: str         # 第 3 节点产出:人工确认/驳回
    report: str                 # 第 4 节点产出
```

- 继承 `MessagesState` 自带 `messages` 字段。
- `total=False`:字段非必填,节点可只更新自己负责的字段。
- `human_decision` 承载人工决定,既是「留痕」,又是条件边的路由依据。

## 4. 数据结构设计(Pydantic schemas)

4 个类:`Party`(签约方)、`ContractInfo`(抽取结果)、`RiskFinding`(单条风险)、`ReviewResult`(审核结果)。

关键设计决策:
- `amount`/`term` 用 `str | None` 而非 `float`:合同金额常含中文大写/千分位,str 更通用;缺失用 None。
- `severity`/`overall_risk` 用裸 `str` 而非 `Literal`:当前教学阶段保持简单,后续可收紧。
- 每个字段 `Field(description=...)` 用中文写清语义——description 是喂给模型的唯一「字段填什么」提示。

## 5. 提示词设计(prompts)

- `EXTRACT_SYSTEM`:信息抽取专家,规则含「缺失填 None 不编造」「key_clauses 简洁概括」。
- `REVIEW_SYSTEM`:合同审核律师,规则含「五类风险清单」「每风险给等级/理由/建议」「实事求是勿凑数」。

原则:**prompt 词汇与 schema 字段对齐**(如 prompt 里的「金额/期限」通过 schema 的 description「合同金额/合同期限」精确映射到 `amount`/`term`)。

## 6. 节点设计

| 节点 | 是否调 LLM | 输入 | 输出 |
|---|---|---|---|
| extract_contract | ✅ `with_structured_output(ContractInfo)` | contract_text | contract_info |
| review_risks | ✅ `with_structured_output(ReviewResult)` | contract_info + contract_text | review |
| human_review | ❌(可能 `interrupt`) | review | human_decision |
| generate_report | ❌ 纯字符串拼接 | contract_info + review | report |

- `human_review` 只筛 `severity == "高"` 的风险展示给人工;无高危则直接写「自动通过」,不打断。
- 条件边函数 `route_after_human_review`(非节点,只读状态选路径):`human_decision` 含「驳回」→ END,否则 → generate_report。

## 7. 数据流

```
contract_text (str)
   → extract_contract: 拼成 [SystemMessage, HumanMessage] → LLM → ContractInfo
   → review_risks: 将 ContractInfo 逐字段格式化 + 原文 → LLM → ReviewResult
   → human_review: 筛高危 → interrupt("请确认...") → 人工输入 → human_decision
        └─ 条件边:驳回 → END / 确认 → generate_report
   → generate_report: 拼接 ContractInfo + ReviewResult → report (str)
```

## 8. 可观测性设计

- 通过 `.env` 设置 `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY`。
- 机制:LangChain 每个 runnable 内置 callback 钩子,追踪器是挂上去的一个回调(横切关注点解耦)。
- 业务代码零埋点,追踪靠配置注入。

## 9. 关键设计决策与理由(留坑)

| 决策 | 理由 |
|---|---|
| 用 `str` 不用 `Literal` | 教学阶段保持简单,避免类型体操 |
| `amount` 用 `str` 不用 `float` | 金额格式多样,str 通用、无精度问题 |
| `review_risks` 逐字段拼接上下文 | 避免 Pydantic repr 噪声,给模型干净带标签的输入 |
| `generate_report` 不调 LLM | 报告是确定性格式化,不花 token 且输出稳定 |
| thinking 关闭 | DeepSeek 思考模式费 token/慢,抽取审核不需要 |
| `@cache` 缓存模型实例 | 避免重复初始化 ChatOpenAI |
| 人工审批用条件边而非新节点 | 路由「只读状态选路径、不做事」,与做事的节点分离;同款机制后续复用于 agent 环 |
| 驳回不生成报告 | 报告是「批准后」的产物;驳回时 `report` 留空,main 侧返回驳回提示 |
| checkpointer 用 `JsonPlusSerializer` 注册 Pydantic 类型 | 自定义类存进 state,msgpack 反序列化需白名单登记,否则未来版本直接报错 |

## 10. 架构演进方向(未来)

1. RAG 知识库 + 工具调用:审核节点自主查法条/判例,检索即工具 → 从带分支 workflow 升级成 agent。
2. OCR 层(眼睛)+ LLM(大脑)的异构多模型。
3. supervisor 多智能体(抽取/审核/数据专家并行)。
