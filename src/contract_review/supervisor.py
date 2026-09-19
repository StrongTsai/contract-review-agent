from typing import Annotated, TypedDict, Literal
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver 
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

import operator
import numpy as np
import re
import argparse

from contract_review.schemas import RiskFinding, ReviewResult, ContractInfo
from contract_review.llm import get_model
from contract_review.graph import extract_contract, human_review, generate_report, route_after_human
from contract_review.knowledge import get_embedder
from contract_review.main import load_contract_text


class SupervisorState(TypedDict, total=False):
    contract_text: str
    contract_info: ContractInfo
    findings: Annotated[list[RiskFinding], operator.add]
    next: str
    review: ReviewResult
    report: str
    human_decision: str
    completed: Annotated[list[str], operator.add]


# 调度模型:supervisor 只吐一个字符串
class Route(BaseModel):
    next: Literal["penalty_expert", "liability_expert", "payment_expert", "dispute_expert", "FINISH"]


# 专家：输出一批 RiskFinding，追加进findings
class SpecialistResult(BaseModel):
    findings: list[RiskFinding]


# 收口：findings忠实保留，LLM只补整体等级 + 结论
class Verdict(BaseModel):
    overall_risk: str
    summary: str


SUPERVISOR_SYSTEM = """你是合同审核的调度员。基于合同原文和已发现的风险,决定下一个派哪个专家或收工。
  可选专家:
  - penalty_expert:违约金/罚则条款
  - liability_expert:责任承担/免责/赔偿条款
  - payment_expert:付款/验收/交付条款
  - dispute_expert:管辖/争议解决/法律适用条款
  规则:
  - 每个专家只派一次,已派过的(看"已派过的专家")不要再派。
  - 合同里还有哪类条款没审,就派对应的专家。
  - 所有相关专家都派完,输出 FINISH。
  """


PENALTY_SYSTEM = """你是违约金条款审核专家。只审违约金/罚则相关条款。
找出风险，给出等级（高/中/低）、理由、修改建议。没有明显问题就返回空列表。
clause 字段必须逐字引用合同原文(含条款编号),不得截断、不得概括、不得改写。"""


LIABILITY_SYSTEM = """你是责任条款审核专家。只审责任承担、免责、赔偿相关的条款。
  重点检查:
  - 一方免责过多,另一方承担全部/无限责任
  - 责任不对等:双方义务与违约责任明显失衡
  - 免责范围过宽,或兜底条款把风险全甩给一方
  - 赔偿上限缺失或畸低,一方损失无法弥补
  找出风险,给出等级(高/中/低)、理由、修改建议。没有明显问题就返回空列表。
  clause 字段必须逐字引用合同原文(含条款编号),不得截断、不得概括、不得改写。"""


PAYMENT_SYSTEM = """你是付款与验收条款审核专家。只审付款、验收、交付相关的条款。
  重点检查:
  - 付款条件对一方明显不利(如先付全款后交付、无进度款保护)
  - 验收标准模糊,或验收权单方面掌握在一方手里
  - 交付节点/时间不明确,逾期交付无约束
  - 付款与交付、验收节点不挂钩,一方有"钱货两空"风险
  找出风险,给出等级(高/中/低)、理由、修改建议。没有明显问题就返回空列表。
  clause 字段必须逐字引用合同原文(含条款编号),不得截断、不得概括、不得改写。"""


DISPUTE_SYSTEM = """你是争议解决条款审核专家。只审管辖、争议解决、法律适用相关的条款。
  重点检查:
  - 管辖法院/仲裁机构约定对一方明显不便(如约定在对方所在地)
  - 仲裁与诉讼选择不当,费用/程序对一方不利
  - 法律适用不透明或对一方不利
  - 争议解决条款缺失或自相矛盾
  找出风险,给出等级(高/中/低)、理由、修改建议。没有明显问题就返回空列表。
  clause 字段必须逐字引用合同原文(含条款编号),不得截断、不得概括、不得改写。"""


# 工厂函数
def make_expert(name: str, system_prompt: str):
    def expert(state: SupervisorState) -> dict:
        model = get_model().with_structured_output(schema=SpecialistResult, method="function_calling")
        result = model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state["contract_text"])
            ]
        )
        # return是把状态加到函数入参参数的SupervisorState里吗？
        return {
            "findings": result.findings, "completed": [name]
        }
    return expert


penalty_expert = make_expert("penalty_expert", PENALTY_SYSTEM)
liability_expert = make_expert("liability_expert", LIABILITY_SYSTEM)
payment_expert = make_expert("payment_expert", PAYMENT_SYSTEM)
dispute_expert = make_expert("dispute_expert", DISPUTE_SYSTEM)


def supervisor(state: SupervisorState) -> dict:
    # 为什么填的是Route
    model = get_model().with_structured_output(Route, method="function_calling")
    found = state.get("findings", [])
    completed = state.get("completed", [])
    found_text = "\n".join(f"- {f.risk_type}" for f in found) or "（暂无）"
    completed_text = "、".join(completed) or "（尚未派工）"
    route = model.invoke(
        [
            SystemMessage(content=SUPERVISOR_SYSTEM),
            HumanMessage(content=f"合同原文: \n{state["contract_text"]}\n\n已派过的专家: {completed_text}\n已发现的风险类型:\n{found_text}")
        ]
    )
    return {
        "next": route.next
    }


SEVERITY_ORDER = {"高": 3, "中": 2, "低": 1}

def norm_clause(s: str) -> str:
    return re.sub(r"\s+", "", s)


def same_clause(a: str, b: str, min_len: int = 10) -> bool:
    if len(a) < min_len or len(b) < min_len:
        return False
    return a in b or b in a


def dedup_findings(findings: list[RiskFinding], type_threshold: float = 0.75) -> list[RiskFinding]:
    if len(findings) <= 1:
        return findings
    ordered = sorted(findings, key=lambda f: SEVERITY_ORDER.get(f.severity, 0), reverse=True)
    types = [f.risk_type for f in ordered]
    tvec = get_embedder().encode(types, normalize_embeddings=True)
    kept = []
    for i in range(len(ordered)):
        dup = False
        for j in kept:
            same = same_clause(norm_clause(ordered[i].clause), norm_clause(ordered[j].clause))
            if same and float(tvec[i] @ tvec[j]) >= type_threshold:
                dup = True
                break
        if not dup:
            kept.append(i)
    return [ordered[i] for i in kept]


def summarize(state: SupervisorState) -> dict:
    findings = state["findings"]
    new_findings = dedup_findings(findings)
    model = get_model().with_structured_output(Verdict, method="function_calling")
    verdict = model.invoke(
        [
            SystemMessage(content="基于下列风险,给整体风险等级(高/中/低)和一句话结论。"),
            HumanMessage("\n".join(
                f" -[{f.severity}] {f.risk_type}: {f.reason}" for f in new_findings
            )),
        ]
    )
    return {
        "review": ReviewResult(
            findings=new_findings,
            overall_risk=verdict.overall_risk,
            summary=verdict.summary,
        )
    }


# 组装
graph = StateGraph(state_schema=SupervisorState)

graph.add_node("extract_contract", extract_contract)
graph.add_node("supervisor", supervisor)
graph.add_node("penalty_expert", penalty_expert)
graph.add_node("liability_expert", liability_expert)
graph.add_node("payment_expert", payment_expert)
graph.add_node("dispute_expert", dispute_expert)
graph.add_node("summarize", summarize)
graph.add_node("human_review", human_review)
graph.add_node("generate_report", generate_report)

graph.add_edge(START, "extract_contract")
graph.add_edge("extract_contract", "supervisor")
graph.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "penalty_expert": "penalty_expert",
        "liability_expert": "liability_expert",
        "payment_expert": "payment_expert",
        "dispute_expert": "dispute_expert",
        "FINISH": "summarize",
    },
)
graph.add_edge("penalty_expert", "supervisor")     # 专家干完回 supervisor 再决策
graph.add_edge("liability_expert", "supervisor")
graph.add_edge("payment_expert", "supervisor")
graph.add_edge("dispute_expert", "supervisor")
graph.add_edge("summarize", "human_review")
graph.add_conditional_edges(
    "human_review",
    route_after_human,
    {"report": "generate_report", "reject": END},
)
graph.add_edge("generate_report", END)

checkpoint = MemorySaver(serde=JsonPlusSerializer(
      allowed_msgpack_modules=[ContractInfo, ReviewResult, RiskFinding]  # 多了 RiskFinding
  ))
app = graph.compile(checkpointer=checkpoint)

def review_contract(contract_text: str) -> str:
    config = {"configurable": {"thread_id": "cyz_test1"}}
    state = {
        "contract_text": contract_text,
    }
    result = app.invoke(state, config=config)

    snapshot = app.get_state(config=config)
    if snapshot.next:
        # 卡住了，区分两种暂停：真 interrupt，假 interrupt_before/after（空暂停，直接续跑）
        tasks = snapshot.tasks or ()
        interrupts = tasks[0].interrupts if tasks else ()

        if interrupts:
            payload = interrupts[0].value
            print("\n" + payload)
            decision = input("\n是否继续生成报告？[y/n]: ")
            result = app.invoke(
                Command(resume="已人工确认，继续" if decision.lower() == "y" else "已驳回，合同需修改！"),
                config=config,
            )
        else:
            result = app.invoke(None, config=config)

    report = result.get("report")
    if report:
        return result["report"]
    else:
        return "已驳回,未生成报告"


def main() -> None:
    parser = argparse.ArgumentParser(description="合同审核数字员工(多智能体版):输入合同文本,输出审核报告")
    parser.add_argument("file", nargs="?", help="合同文件路径(.txt/.png/.pdf)")
    parser.add_argument("--text", help="直接传入合同文本(与 file 二选一)")
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif args.file:
        text = load_contract_text(args.file)
    else:
        parser.error("请提供文件路径,或使用 --text 直接传入合同文本")

    print(review_contract(text))

if __name__ == "__main__":
    main()