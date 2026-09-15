import json
from functools import cache

import numpy as np
from sentence_transformers import SentenceTransformer

from contract_review.settings import PROJECT_ROOT


CASE_SUMMARIES = [
    {
        "title": "某软件开发合同违约金调减案",
        "content": "约定每日按合同总金额千分之五支付违约金,折合年化约 182.5%,法院认定明显过高,调减为以未付款项为基数、按 LPR 四倍以内计算。",
    },
    {
        "title": "某格式条款未提示说明案",
        "content": "提供方未对免除自身责任的格式条款尽提示说明义务,该条款不成为合同内容,对相对方不发生效力。",
    },
    {
        "title": "某责任不对等条款显失公平案",
        "content": "约定「因系统运行造成的一切损失均由一方承担」,法院认定双方权利义务明显失衡、显失公平,可请求撤销或予以调整。",
    },
]


def _load_civil_code():
    path = PROJECT_ROOT / "data" / "civil_code.jsonl"
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [{"title": f"《民法典》{r['article_number']}", "content": r["content"]} for r in rows]


@cache
def get_embedder():
    return SentenceTransformer("BAAI/bge-small-zh-v1.5")


def _embed(texts):
    return get_embedder().encode(list(texts), normalize_embeddings=True)


class VectorStore:
    def __init__(self, entries):
        self.entries = entries
        self.docs = [f"{e['title']} {e['content']}" for e in entries]
        self.vectors = _embed(self.docs)  # 建库时 embed 一次

    def search(self, query, top_k=3):
        q = _embed([query])[0]  # 每次只 embed 查询
        scores = self.vectors @ q
        order = np.argsort(-scores)[:top_k]
        return [self.entries[i] for i in order]


LAW_STORE = VectorStore(_load_civil_code())
CASE_STORE = VectorStore(CASE_SUMMARIES)


def format_hits(hits, empty_msg):
    if not hits:
        return empty_msg
    return "\n\n".join(f"{h['title']}\n{h['content']}" for h in hits)
