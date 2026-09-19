from typing import Annotated, TypedDict, Literal

from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command, Send

import operator
import argparse

from contract_review.schemas import RiskFinding, ReviewResult, ContractInfo
from contract_review.llm import get_model
from contract_review.graph import extract_contract, human_review, generate_report, route_after_human
from contract_review.main import load_contract_text
from contract_review import experts


class SupervisorParallelState(TypedDict, total=False):
    contract_text: str
    contract_info: ContractInfo
    findings: Annotated[list[RiskFinding], operator.add]
    review: ReviewResult
    report: str
    human_decision: str
    experts: list[str]


class Route(BaseModel):
    experts: list[Literal["penalty_expert", "liability_expert", "payment_expert", "dispute_expert"]]


SUPERVISOR_PARALLEL_SYSTEM = """你是合同审核的调度员。基于合同原文和已发现的风险,决定下一个派哪个专家或收工。
  可选专家:
  - penalty_expert:违约金/罚则条款
  - liability_expert:责任承担/免责/赔偿条款
  - payment_expert:付款/验收/交付条款
  - dispute_expert:管辖/争议解决/法律适用条款
  规则:
  - 理解合同信息，一次列出所有你认为需要派的专家
  """

def supervisor_parallel(state: SupervisorParallelState) -> dict:
    model = get_model().with_structured_output(Route, method="function_calling")
    route = model.invoke(
        [
            SystemMessage(content=SUPERVISOR_PARALLEL_SYSTEM),
            HumanMessage(content=f"合同原文: \n{state['contract_text']}\n\n")
        ]
    )
    print(f"派出的专家: {route.experts}")
    return {"experts": route.experts}


def route_after_supervisor(state: SupervisorParallelState):
    return [Send(name, {"contract_text": state["contract_text"]}) for name in state["experts"]]


penalty_expert = experts.penalty_expert
liability_expert = experts.liability_expert
payment_expert = experts.payment_expert
dispute_expert = experts.dispute_expert


graph = StateGraph(state_schema=SupervisorParallelState)

graph.add_node("extract_contract", extract_contract)
graph.add_node("supervisor_parallel", supervisor_parallel)
graph.add_node("penalty_expert", penalty_expert)
graph.add_node("liability_expert", liability_expert)
graph.add_node("payment_expert", payment_expert)
graph.add_node("dispute_expert", dispute_expert)
graph.add_node("summarize", experts.summarize)
graph.add_node("human_review", human_review)
graph.add_node("generate_report", generate_report)

graph.add_edge(START, "extract_contract")
graph.add_edge("extract_contract", "supervisor_parallel")
graph.add_conditional_edges(
    "supervisor_parallel",
    route_after_supervisor
)
graph.add_edge("penalty_expert", "summarize")     # 专家干完回 supervisor 再决策
graph.add_edge("liability_expert", "summarize")
graph.add_edge("payment_expert", "summarize")
graph.add_edge("dispute_expert", "summarize")
graph.add_edge("summarize", "human_review")
graph.add_conditional_edges(
    "human_review",
    route_after_human,
    {"report": "generate_report", "reject": END},
)
graph.add_edge("generate_report", END)

checkpoint = MemorySaver(serde=JsonPlusSerializer(
    allowed_msgpack_modules=[ContractInfo, ReviewResult, RiskFinding]
))
app = graph.compile(checkpointer=checkpoint)


def review_contract(contract_text: str) -> str:
    config = {"configurable": {"thread_id": "cyz_test1"}}
    state = {"contract_text": contract_text}
    result = app.invoke(state, config=config)

    snapshot = app.get_state(config=config)
    if snapshot.next:
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
    parser = argparse.ArgumentParser(description="合同审核数字员工(多智能体并行版):输入合同文本,输出审核报告")
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
