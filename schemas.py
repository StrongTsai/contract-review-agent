from pydantic import BaseModel, Field


class Party(BaseModel):
    name: str = Field(description="签约方名称")
    role: str = Field(description="角色，如 甲方/乙方")


class ContractInfo(BaseModel):
    subject: str = Field(description="合同标的，一句话概括")
    parties: list[Party] = Field(description="合同签约方列表")
    amount: str | None = Field(description="合同金额,无法确定则为 None")
    term: str | None = Field(description="合同期限/有效期,无法确定则为 None")
    key_clauses: list[str] = Field(description="关键条款摘要列表")


class RiskFinding(BaseModel):
    clause: str = Field(description="涉及风险的具体条款原文或位置")
    risk_type: str = Field(description="风险类型,如 高额违约金/单方解约权/责任不对等/条款模糊")
    severity: str = Field(description="风险等级:高/中/低")
    reason: str = Field(description="为什么判定为风险")
    suggestion: str = Field(description="修改建议")


class ReviewResult(BaseModel):
    findings: list[RiskFinding] = Field(description="风险发现列表")
    overall_risk: str = Field(description="整体风险等级:高/中/低")
    summary: str = Field(description="一句话审核结论")