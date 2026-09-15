# 合同审核数字员工

基于 LangGraph + DeepSeek 的合同审核 agent:输入合同文本 → 结构化抽取 → 自主查证审核 → 人工审批 → 生成报告。

## 能力

- **工作流编排**:四节点 `StateGraph`(抽取 → 审核 → 人工审批 → 报告)+ 条件边 + 断点续跑(`MemorySaver` checkpointer)
- **自主 agent**:审核节点内嵌 `create_agent`,绑定 `search_law` / `query_case` 工具,自主决定查什么、查几次
- **RAG**:全量《民法典》1260 条 + 判例,本地 `bge-small-zh-v1.5` embedding,向量语义检索
- **OCR 眼睛**:本地 `RapidOCR`(ONNX 跑 PP-OCR 模型)识别扫描件/图片 → 「OCR 当眼睛 + DeepSeek 当大脑」
- **转人工**:高危风险用 `interrupt` 挂起,人工确认 / 驳回后走条件边路由
- **结构化输出**:Pydantic 模型 + `response_format`(函数调用模拟结构化输出)

## 目录结构

```
contract-review-agent/
├── settings.py    # 读 .env 环境变量
├── llm.py         # DeepSeek 模型单例(@cache)
├── schemas.py     # Pydantic 结构化输出模型
├── prompts.py     # 抽取 / 审核 system prompt
├── knowledge.py   # 民法典 + 判例语料 + VectorStore 向量检索
├── tools.py       # search_law / query_case 检索工具
├── ocr.py         # RapidOCR 图片识别
├── graph.py       # LangGraph 四节点图(嵌套 agent + 条件边 + checkpointer)
├── main.py        # CLI 入口(两阶段 invoke,处理中断/恢复)
├── data/          # civil_code.jsonl 民法典全文
├── samples/       # 示例合同
└── docs/          # 需求/设计/开发/踩坑/概念/面试话术
```

## 快速开始

```bash
uv sync                         # 安装依赖(uv 管理)
cp .env.example .env            # 填入 DEEPSEEK_API_KEY
uv run python main.py samples/contract.txt

# 或直接传图片(OCR 转文本)
uv run python main.py samples/contract.png

# 或直接传文本
uv run python main.py --text "甲方委托乙方开发系统,乙方每逾期一日按合同总金额千分之五支付违约金。"
```

## 图结构

```
START → extract_contract → review_risks → human_review ─(确认)─→ generate_report → END
                                            └────────(驳回)─→ END
```

- `extract_contract`:结构化抽取标的 / 签约方 / 金额 / 期限 / 关键条款
- `review_risks`:内嵌 `create_agent`,自主调工具查证法条 / 判例,输出风险清单
- `human_review`:高危风险 `interrupt` 挂起,人工确认 / 驳回
- `generate_report`:纯 Python 拼装可读报告

## 路线图

- [x] 工作流 + interrupt 转人工 + 断点续跑
- [x] 审核节点 agent 化(create_agent)
- [x] 向量 RAG(全量民法典 + 判例)
- [x] LangSmith 追踪
- [x] OCR 接入(RapidOCR,本地图片识别)
- [x] PDF 支持(fitz/PyMuPDF 栅格化后再 OCR)
- [ ] supervisor 多智能体
