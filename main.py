import argparse

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import app


def review_contract(contract_text: str) -> str:
    config = {"configurable": {"thread_id": "cyz_test1"}}
    state = {
        "messages": [HumanMessage(content=contract_text)],
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
    parser = argparse.ArgumentParser(description="合同审核数字员工:输入合同文本,输出审核报告")
    parser.add_argument("file", nargs="?", help="合同文本文件路径")
    parser.add_argument("--text", help="直接传入合同文本(与 file 二选一)")
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        parser.error("请提供文件路径,或使用 --text 直接传入合同文本")

    print(review_contract(text))


if __name__ == "__main__":
    main()
