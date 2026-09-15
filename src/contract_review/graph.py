from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from functools import cache

from contract_review.llm import get_model
from contract_review.prompts import EXTRACT_SYSTEM, REVIEW_SYSTEM
from contract_review.schemas import ContractInfo, ReviewResult
from contract_review.tools import search_law, query_case


class ContractState(MessagesState, total=False):
    contract_text: str
    contract_info: ContractInfo
    review: ReviewResult
    report: str
    human_decision: str


def extract_contract(state: ContractState) -> dict:
    origin_contract = state["contract_text"]
    model = get_model().with_structured_output(ContractInfo, method="function_calling")
    info = model.invoke(
        [SystemMessage(content=EXTRACT_SYSTEM), HumanMessage(content=origin_contract)]
    )
    return {
        "contract_info": info
    }


@cache
def get_review_agent():
    return create_agent(
        get_model(),
        tools=[search_law, query_case],
        system_prompt=REVIEW_SYSTEM,
        response_format=ReviewResult
    )


def review_risks(state: ContractState) -> dict:
    # model = get_model().with_structured_output(ReviewResult, method="function_calling")
    info = state["contract_info"]
    origin_contract = state["contract_text"]
    agent = get_review_agent()
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=f"合同原文: \n{origin_contract}\n已抽取的合同信息: \n{info}")]
        }
    )
    # 为什么要把info每个字段重新拼接？
    # review = model.invoke(
    #     [
    #         SystemMessage(content=REVIEW_SYSTEM),
    #         HumanMessage(content=f"合同原文: \n{origin_contract}\n已抽取的合同信息: \n{info}")
    #     ]
    # )
    return {
        "review": result["structured_response"]
    }


def human_review(state: ContractState) -> dict:
    r = state["review"]
    lines = []
    for i, f in enumerate(r.findings, 1):
        if f.severity == "高":
            lines.append(f"{i}. [{f.severity}] {f.risk_type}")
            lines.append(f"   条款: {f.clause}")
            lines.append(f"   理由: {f.reason}")
            lines.append(f"   建议: {f.suggestion}")
            lines.append("")

    if len(lines) == 0:
        return {
            "human_decision": "无高危风险,自动通过"
        }
    
    severity_list = "\n".join(lines)
    human_decision = interrupt(f"请确认风险点\n{severity_list}")
    print(f"\n{human_decision}\n")

    return {
        "human_decision": human_decision
    }


def generate_report(state:ContractState) -> dict:
    info = state["contract_info"]
    review = state["review"]
    parties = ", ".join(f"{p.name}({p.role})" for p in info.parties)

    lines = [
        "=" * 56,
        "合 同 审 核 报 告",
        "=" * 56,
        f"标的: {info.subject}",
        f"签约方: {parties}",
        f"金额: {info.amount or '未明确'}",
        f"期限: {info.term or '未明确'}",
        f"整体风险等级: {review.overall_risk}",
        "",
        f"审核结论: {review.summary}",
        "",
        "-" * 56,
        f"共发现 {len(review.findings)} 处风险:",
    ]
    for i, f in enumerate(review.findings, 1):
        lines.append(f"{i}. [{f.severity}] {f.risk_type}")
        lines.append(f"   条款: {f.clause}")
        lines.append(f"   理由: {f.reason}")
        lines.append(f"   建议: {f.suggestion}")
        lines.append("")

    report = "\n".join(lines)
    return {"report": report, "messages": [AIMessage(content=report)]}


def route_after_human(state: ContractState) -> str:
    return "reject" if "驳回" in state["human_decision"] else "report"


graph = StateGraph(state_schema=ContractState)

graph.add_node("extract_contract", extract_contract)
graph.add_node("review_risks", review_risks)
graph.add_node("human_review", human_review)
graph.add_node("generate_report", generate_report)

graph.add_edge(START, "extract_contract")
graph.add_edge("extract_contract", "review_risks")
graph.add_edge("review_risks", "human_review")
graph.add_conditional_edges(
    "human_review",
    route_after_human,
    {
        "report": "generate_report",
        "reject": END
    }
)
graph.add_edge("generate_report", END)

checkpoint = MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=[ContractInfo, ReviewResult]))
app = graph.compile(checkpointer=checkpoint)