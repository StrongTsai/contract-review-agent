from functools import cache
from langchain_openai import ChatOpenAI
from settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

@cache
def get_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        temperature=0,
        streaming=False,
        openai_api_base=DEEPSEEK_BASE_URL,
        openai_api_key=DEEPSEEK_API_KEY,
        extra_body={
            "thinking":{
                "type":"disabled"
            }
        },
    )