# 合同审核数字员工

基于 LangGraph + DeepSeek 的合同审核 agent:输入合同文本 → 结构化抽取 → 自主查证审核 → 人工审批 → 生成报告。

## 能力

- **工作流编排**:四节点 `StateGraph`(抽取 → 审核 → 人工审批 → 报告)+ 条件边 + 断点续跑(`MemorySaver` checkpointer)
- **自主 agent**:审核节点内嵌 `create_agent`,绑定 `search_law` / `query_case` 工具,自主决定查什么、查几次
- **RAG**:全量《民法典》1260 条 + 判例,本地 `bge-small-zh-v1.5` embedding,向量语义检索
- **OCR 眼睛**:本地 `RapidOCR`(ONNX 跑 PP-OCR 模型)识别扫描件/图片 → 「OCR 当眼睛 + DeepSeek 当大脑」
- **转人工**:高危风险用 `interrupt` 挂起,人工确认 / 驳回后走条件边路由
- **结构化输出**:Pydantic 模型 + `response_format`(函数调用模拟结构化输出)
- **supervisor 多智能体**:审核拆成 4 个专家(违约金 / 责任 / 付款 / 争议),supervisor 派工,embedding 去重
- **Send 并行派工**:supervisor 一次列出专家,`Send` 扇出 4 专家并发审核(串行 / 并行两版共存)

## 目录结构

```
contract-review-agent/
├── src/
│   └── contract_review/    # 包:所有源码
│       ├── settings.py     # 读 .env 环境变量 + 项目根路径
│       ├── llm.py          # DeepSeek 模型单例(@cache)
│       ├── schemas.py      # Pydantic 结构化输出模型
│       ├── prompts.py      # 抽取 / 审核 system prompt
│       ├── knowledge.py    # 民法典 + 判例语料 + VectorStore 向量检索
│       ├── tools.py        # search_law / query_case 检索工具
│       ├── ocr.py          # RapidOCR 图片识别
│       ├── graph.py                 # LangGraph 四节点图(嵌套 agent + 条件边 + checkpointer)
│       ├── experts.py               # 多智能体共享(4 专家 + 去重 + summarize)
│       ├── supervisor.py            # supervisor 多智能体(串行派工)
│       ├── supervisor_parallel.py   # supervisor 多智能体(Send 并行派工)
│       └── main.py                  # CLI 入口(两阶段 invoke,处理中断/恢复)
├── data/           # civil_code.jsonl 民法典全文
├── samples/        # 示例合同
└── docs/           # 需求/设计/开发/踩坑/概念/面试话术
```

## 快速开始

```bash
uv sync                         # 安装依赖 + 构建包(uv 管理)
cp .env.example .env            # 填入 DEEPSEEK_API_KEY
uv run python -m contract_review.main samples/contract.txt

# 或直接传图片(OCR 转文本)
uv run python -m contract_review.main samples/contract.png

# 或直接传文本
uv run python -m contract_review.main --text "甲方委托乙方开发系统,乙方每逾期一日按合同总金额千分之五支付违约金。"

# supervisor 多智能体版(4 专家分工审核)
uv run python -m contract_review.supervisor samples/contract.txt

# supervisor 并行版(Send 扇出,4 专家并发)
uv run python -m contract_review.supervisor_parallel samples/contract.txt
```

## 图结构

单 agent 版(默认):
```
START → extract_contract → review_risks → human_review ─(确认)─→ generate_report → END
                                            └────────(驳回)─→ END
```

supervisor 多智能体版 —— 串行派工(supervisor.py):
```
START → extract_contract → supervisor ⇄ {penalty / liability / payment / dispute}
                                      └──(FINISH)──→ summarize → human_review ─(确认)─→ generate_report → END
                                                                              └──(驳回)──→ END
```

supervisor 多智能体版 —— Send 并行(supervisor_parallel.py):
```
START → extract_contract → supervisor ─(Send 扇出)→ {penalty / liability / payment / dispute 并行}
                                                 └──(全部跑完)→ summarize → human_review ─(确认)─→ generate_report → END
                                                                                        └──(驳回)──→ END
```

- `extract_contract`:结构化抽取标的 / 签约方 / 金额 / 期限 / 关键条款
- `review_risks` / `supervisor` + 4 专家:三种审核实现——单 agent 自主查证、调度员串行派工、`Send` 并行派工
- `summarize`:专家 findings 去重后,LLM 补整体等级 + 结论
- `human_review`:高危风险 `interrupt` 挂起,人工确认 / 驳回
- `generate_report`:纯 Python 拼装可读报告

## 路线图

- [x] 工作流 + interrupt 转人工 + 断点续跑
- [x] 审核节点 agent 化(create_agent)
- [x] 向量 RAG(全量民法典 + 判例)
- [x] LangSmith 追踪
- [x] OCR 接入(RapidOCR,本地图片识别)
- [x] PDF 支持(fitz/PyMuPDF 栅格化后再 OCR)
- [x] supervisor 多智能体(4 专家 + 动态派工 + embedding 去重)
- [x] Send 并行派工(supervisor_parallel.py,4 专家扇出并发)
