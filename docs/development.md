# 合同审核数字员工 · 开发文档

> 版本 v1.4 · 2026-09-19

## 1. 项目结构

```
contract-review-agent/
├── src/
│   └── contract_review/    # 包:所有源码(绝对导入)
│       ├── settings.py     # 环境变量加载 + 项目根路径
│       ├── llm.py          # DeepSeek 模型单例
│       ├── schemas.py      # 结构化输出数据类
│       ├── prompts.py      # 两个 system prompt
│       ├── knowledge.py    # 全量民法典 + 判例语料 + 向量检索(VectorStore)
│       ├── tools.py        # search_law / query_case 检索工具
│       ├── ocr.py          # RapidOCR 图片识别(眼睛)
│       ├── graph.py        # LangGraph 四节点图(含条件边 + checkpointer + 嵌套 agent)
│       ├── supervisor.py   # supervisor 多智能体图(4 专家 + 动态派工 + 去重 + 转人工)
│       └── main.py         # CLI 入口(两阶段 invoke,处理中断/恢复)
├── data/          # 语料数据(civil_code.jsonl 民法典全文)
├── .env           # 真实密钥(不入库)
├── .env.example   # 密钥模板(占位符)
├── samples/       # 测试合同
└── docs/          # 需求/设计/开发/踩坑/语法概念文档
```

## 2. 环境准备

```bash
# 依赖(uv 或 pip)
uv add langchain-openai langgraph pydantic python-dotenv
uv add rapidocr onnxruntime   # OCR 眼睛

# 配置密钥
cp .env.example .env   # 然后填入真实 DEEPSEEK_API_KEY 与 LANGCHAIN_API_KEY
```

## 3. 文件说明

| 文件 | 职责 | 关键实现 |
|---|---|---|
| settings.py | 读环境变量 | `load_dotenv()` + `os.getenv` |
| llm.py | 模型单例 | `@cache` + `ChatOpenAI` + thinking 关闭 |
| schemas.py | 4 个 Pydantic 类 | `str \| None`、`list[T]`、嵌套 |
| prompts.py | 2 个 system prompt | 防幻觉规则、风险清单 |
| knowledge.py | 法条/判例语料 + 向量库 | `SentenceTransformer` + `VectorStore`(建库预计算向量)+ `LAW_STORE`/`CASE_STORE` |
| tools.py | 检索工具 | `@tool` 的 `search_law`/`query_case`,调 `VectorStore.search` |
| ocr.py | OCR 眼睛 | `@cache` + `RapidOCR`,图片 → 文本 |
| graph.py | 四节点图 + 条件边 + 嵌套 agent | `create_agent`(review_risks 内)+ `interrupt` + `add_conditional_edges` + `MemorySaver` |
| supervisor.py | supervisor 多智能体图 | 4 专家工厂 + supervisor 串行派工 + 去重 + interrupt 转人工 |
| main.py | CLI,两阶段 invoke | `argparse` + `get_state` 检测中断 + `Command(resume)` |

## 4. 运行

```bash
uv run python -m contract_review.main samples/contract.txt          # 文本文件输入
uv run python -m contract_review.main samples/contract.png          # 图片输入(OCR 转文本)
uv run python -m contract_review.main --text "甲方乙方...合同正文"   # 直接传文本
uv run python -m contract_review.supervisor samples/contract.txt    # supervisor 多智能体版(4 专家分工)
```

有高危风险时会暂停、打印风险点、等待输入 `y/n`;`y` 出报告,`n` 打印「已驳回,未生成报告」。

## 5. 验证

```bash
# 图编译验证
uv run python -c "from contract_review.graph import app; print(list(app.get_graph().nodes))"

# 可观测性验证(需 LANGCHAIN_TRACING_V2=true)
# 运行后到 smith.langchain.com 查看 contract-review-agent 项目的 trace

# 人工审批验证:分别输入 y / n,确认 y 出报告、n 不出报告
```

## 6. 开发记录

- [x] settings.py — 环境变量
- [x] llm.py — DeepSeek 模型(thinking 关闭)
- [x] schemas.py — 4 个数据类
- [x] prompts.py — 2 个 system prompt
- [x] graph.py — 三节点流水线
- [x] main.py — CLI 入口
- [x] .env + LangSmith 追踪
- [x] 端到端跑通 samples/contract.txt
- [x] interrupt 转人工:human_review 节点 + checkpointer
- [x] main.py 两阶段 invoke(get_state 检测中断 + Command(resume) 恢复)
- [x] 条件边路由:确认 → 报告 / 驳回 → END
- [x] checkpointer 注册 Pydantic 类型(msgpack 白名单)
- [x] RAG 向量检索:knowledge.py 语料 + VectorStore(bge-small-zh-v1.5 本地 embedding)
- [x] 审核节点 agent 化:create_agent + search_law/query_case 工具,自主查证法条/判例
- [x] 语料升级为全量《民法典》(1260 条原文,data/civil_code.jsonl 加载,语义检索验证通过)
- [x] OCR 接入:ocr.py(RapidOCR)+ main.py 按扩展名分图片/文本输入,端到端跑通 samples/contract.png
- [x] src layout 重构:代码归入 src/contract_review/ 包,绝对导入,`python -m contract_review.main` 入口
- [x] supervisor 多智能体:supervisor.py(4 专家工厂 + 串行派工 + 动态路由),端到端跑通
- [x] 专家 findings 去重:clause 子串 + risk_type embedding 两段式,宁漏勿杀
- [x] supervisor CLI 入口:复用 load_contract_text,支持 file / --text

## 7. 踩坑记录

详见 [`docs/pitfalls.md`](pitfalls.md)(独立维护,写代码遇坑主动追加)。已覆盖 15 条,新增:checkpointer 忘传 thread_id(#13)、`__name__ == "main"` 写错(#14)、clause 文本相似度去重误杀同条款不同风险点(#15)。

## 8. 后续开发计划

1. ~~RAG 知识库 + 工具调用(agent 化)~~ ✅ 已完成:review 节点内嵌 create_agent,自主查证。
2. ~~OCR 接入~~ ✅ 已完成:RapidOCR(ONNX 跑 PP-OCR 模型)本地识别图片。
3. ~~PDF 支持~~ ✅ 已完成:fitz/PyMuPDF 栅格化 → OCR,端到端跑通 contract.pdf。
4. ~~supervisor 多智能体~~ ✅ 已完成:supervisor.py(4 专家 + 串行派工 + embedding 去重 + 转人工)。
5. LangSmith 追踪的 token/成本分析。
6. supervisor 4 专家从串行派工改 `Send` 并行(当前串行,4 专家依次跑,慢)。
