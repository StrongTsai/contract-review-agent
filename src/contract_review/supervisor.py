from typing import Annotated, TypedDict, Literal
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver 
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
import operator

from contract_review.schemas import RiskFinding, ReviewResult, ContractInfo
from contract_review.llm import get_model
from contract_review.graph import extract_contract, human_review, generate_report, route_after_human


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
    next: Literal["penalty_expert", "FINISH"]


# 专家：输出一批 RiskFinding，追加进findings
class SpecialistResult(BaseModel):
    findings: list[RiskFinding]


# 收口：findings忠实保留，LLM只补整体等级 + 结论
class Verdict(BaseModel):
    overall_risk: str
    summary: str


SUPERVISOR_SYSTEM = """你是合同审核的调度员。基于合同原文和已发现的风险,决定下一步。
  规则:
  - 每个专家只派一次,已覆盖的风险类型不要重复派。
  - 违约金条款还没审 → 派 penalty_expert;都审完了 → FINISH。
  """


PENALTY_SYSTEM = """你是违约金条款审核专家。只审违约金/罚则相关条款。
找出风险，给出等级（高/中/低）、理由、修改建议。没有明显问题就返回空列表。"""


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


def penalty_expert(state: SupervisorState) -> dict:
    model = get_model().with_structured_output(schema=SpecialistResult, method="function_calling")
    result = model.invoke(
        [
            SystemMessage(content=PENALTY_SYSTEM),
            HumanMessage(content=state["contract_text"])
        ]
    )
    return {
        "findings": result.findings, "completed": ["penalty_expert"]
    }


def summarize(state: SupervisorState) -> dict:
    model = get_model().with_structured_output(Verdict, method="function_calling")
    findings = state["findings"]
    verdict = model.invoke(
        [
            SystemMessage(content="基于下列风险,给整体风险等级(高/中/低)和一句话结论。"),
            HumanMessage("\n".join(
                f" -[{f.severity}] {f.risk_type}: {f.reason}" for f in findings
            )),
        ]
    )
    return {
        "review": ReviewResult(
            findings=findings,
            overall_risk=verdict.overall_risk,
            summary=verdict.summary,
        )
    }


# 组装
graph = StateGraph(state_schema=SupervisorState)

graph.add_node("extract_contract", extract_contract)
graph.add_node("supervisor", supervisor)
graph.add_node("penalty_expert", penalty_expert)
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
        "FINISH": "summarize",
    },
)
graph.add_edge("penalty_expert", "supervisor")     # 专家干完回 supervisor 再决策
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