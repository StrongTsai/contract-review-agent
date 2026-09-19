import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from contract_review.knowledge import get_embedder
from contract_review.llm import get_model
from contract_review.schemas import ReviewResult, RiskFinding


class SpecialistResult(BaseModel):
    findings: list[RiskFinding]


class Verdict(BaseModel):
    overall_risk: str
    summary: str


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


def make_expert(name: str, system_prompt: str):
    def expert(state: Any) -> dict:
        model = get_model().with_structured_output(schema=SpecialistResult, method="function_calling")
        result = model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state["contract_text"]),
            ]
        )
        return {"findings": result.findings}
    return expert


penalty_expert = make_expert("penalty_expert", PENALTY_SYSTEM)
liability_expert = make_expert("liability_expert", LIABILITY_SYSTEM)
payment_expert = make_expert("payment_expert", PAYMENT_SYSTEM)
dispute_expert = make_expert("dispute_expert", DISPUTE_SYSTEM)


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


def summarize(state: Any) -> dict:
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
