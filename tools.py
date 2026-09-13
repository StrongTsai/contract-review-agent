from langchain_core.tools import tool
from knowledge import CASE_STORE, LAW_STORE, format_hits

from typing import Annotated
from pydantic import Field

@tool
def search_law(keyword: Annotated[str, Field(description="待检索的法律概念关键词，如“违约金”、“违约责任”")]) -> str:
    """检索合同相关的法律法规条文。当需要判断某条款是否合法、引用法条依据时调用。
    返回: 命中的发条正文；未命中返回【未找到相关法条】。"""
    hits = LAW_STORE.search(keyword)
    return format_hits(hits, "未找到相关法条")

@tool
def query_case(keyword: Annotated[str, Field(description="待检索的法律关键词，如“违约金过高”、“格式条款”")]) -> str:
    """检索类似合同纠纷的既往判例。当需要判断某类条款在司法实践中的处理倾向时调用。
    返回: 命中的判例摘要；未命中返回【未找到相关判例】"""
    hits = CASE_STORE.search(keyword)
    return format_hits(hits, "未找到相关判例")