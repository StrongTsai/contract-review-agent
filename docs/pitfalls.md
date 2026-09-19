# 合同审核数字员工 · 踩坑记录

> 维护说明:写代码过程中遇到的真坑,按「现象 → 根因 → 解法 → 教训」记在这里。后续有新坑,Claude 主动追加,不用等用户提醒。

## 环境与依赖

### 1. 文件名单复数:`schema.py` vs `schemas.py`
- **现象**:`from schemas import ...` 报 `ModuleNotFoundError`。
- **根因**:文件存成了 `schema.py`(单数),和 import 的 `schemas`(复数)不一致。
- **解法**:统一用 `schemas.py`(复数),与 import 语句对齐。
- **教训**:模块名是约定,存盘前先对一遍 import 语句。

### 2. 真实密钥写进 `.env.example`
- **现象**:git 仓库里 `.env.example` 带了真实 `DEEPSEEK_API_KEY`,push 即泄露。
- **根因**:复制 `.env` 当模板,忘了擦真 key。
- **解法**:key 只存 `.env`(不入库);`.env.example` 用 `sk-你的DeepSeek-key` 占位符。
- **教训**:秘密 = 信任边界;模板文件默认「可能被公开」,永不写真值。

### 3. 验证脚本忘 `load_dotenv()`
- **现象**:LangSmith 客户端报 `401 Invalid token`。
- **根因**:独立进程里直接 `Client()`,没加载 `.env`,`LANGCHAIN_API_KEY` 是空。
- **解法**:每个独立进程入口先 `load_dotenv()`。
- **教训**:`.env` 不会自动注入,任何「直接跑」的脚本都得自己加载。

## LangGraph 状态与中断

### 4. `interrupt()` 不挂 checkpointer 报错
- **现象**:`python main.py` 裸跑时 interrupt 不工作/报错。
- **根因**:interrupt 依赖 checkpointer 保存/恢复状态;只有 `langgraph dev` 会替你自动配置,裸跑必须显式 `MemorySaver`。
- **解法**:`app = graph.compile(checkpointer=MemorySaver())`。
- **教训**:interrupt 的「暂停」本质是「存 checkpoint + 等 resume」,没有 checkpointer 就没有暂停。

### 5. 只看 `next` 非空,就以为有 interrupt 值
- **现象**:`snapshot.tasks[0].interrupts[0]` 可能 `IndexError`。
- **根因**:`next` 非空只表示「图暂停了」;暂停可能来自 `interrupt()`(有值),也可能来自 `interrupt_before/after`(interrupts 为空)。
- **解法**:先判断 `tasks[].interrupts` 是否为空,再取 `interrupts[0].value`。
- **教训**:「暂停」≠「有值可取的暂停」,两者要分开判断。

### 6. 空暂停续跑传了 `state` 而非 `None`
- **现象**:`interrupt_before/after` 续跑时,状态被初始输入覆盖。
- **根因**:`invoke` 的第一个参数是「新输入」,会被当成对当前状态的一次更新合并;续跑应传 `None` 表示「无新输入」。
- **解法**:空暂停用 `invoke(None, config)`;只有带值恢复才用 `Command(resume=...)`。
- **教训**:`invoke(None)` 是「接着跑」的规范表达,不是传原来的 state。

### 7. state 字段没声明,resume 后拿不到
- **现象**:toy 图 resume 后 `result` 里没有预期的字段(如 `answer`)。
- **根因**:用了 `MessagesState`,它只声明 `messages`,没声明 `answer`;LangGraph 不会保存没声明的字段。
- **解法**:用 `TypedDict` + `total=False` 显式声明所有要跨节点传递的字段。
- **教训**:state 的字段必须显式声明,checkpoint 只序列化声明的字段。

## 序列化 / checkpoint

### 8. Pydantic 对象存进 state,msgpack 反序列化 warning
- **现象**:运行报 `Deserializing unregistered type schemas.ContractInfo`,提示未来版本会直接拒绝。
- **根因**:自定义 Pydantic 类不在 msgpack 默认白名单;反序列化是信任边界,默认拒绝。
- **解法**:`MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=[ContractInfo, ReviewResult]))`。
- **教训**:把自定义类型塞进 state,就要显式登记白名单。

## 图结构 / 路由

### 9. 驳回后仍生成报告
- **现象**:人工输入 `n`,图还是出报告。
- **根因**:`human_review → generate_report` 是固定边,人工决定被收集进 `human_decision` 却从没被「用」去路由。
- **解法**:换成 `add_conditional_edges` + 路由函数,读 `human_decision` 决定「确认 → 报告 / 驳回 → END」;main 侧用 `get("report")` 兜底。
- **教训**:「收集决定」和「根据决定分支」是两个动作;前者是 interrupt,后者是条件边。

### 10. 用了弃用 API(`create_react_agent`)没发现
- **现象**:编辑器提示 `create_react_agent` 已弃用(deprecated)。
- **根因**:LangChain v1 把预置 agent 统一到 `langchain.agents.create_agent`,旧的 `langgraph.prebuilt.create_react_agent` 被降级保留。
- **解法**:迁到 `create_agent`,`prompt=` 改名 `system_prompt=`,其余(`tools`、`response_format`、`structured_response`)不变。
- **教训**:写生产代码前先看官方弃用提示;抄网上旧教程时尤其要核对 API 是否已换代。

## OCR / 多模态

### 11. OCR 按「视觉行」吐文本,naive join 把换行处硬切开
- **现象**:OCR 识别合同图,`10%` 被切成两行 `剩余1` / `0%作为质保金`。
- **根因**:OCR 返回的是「视觉行」,不是「段落」。图里 `10%` 恰好落在换行处,OCR 忠实地切成两个文本框;`"\n".join(result.txts)` 把这个视觉换行硬保留了。同理顿号 `、`、句号 `。` 偶尔丢,是标点识别本身不稳。
- **解法**:v1 用 naive join 够——下游 LLM 抗噪,能把 `剩余1\n0%` 重建回「剩余10%」。真要「版面还原」(按标点重建段落、处理表格/多栏)再单独做,不在 OCR 后堆复杂规则。
- **教训**:OCR 的「忠实转录」和「版面理解」是两件事;前者交给 OCR,后者交给 LLM 归一化,各干各擅长的。

### 12. PyMuPDF 弃用 `fitz` 模块名,改 `import pymupdf`
- **现象**:运行打印 `warning: The fitz API is deprecated... Use import pymupdf instead`。
- **根因**:新版 PyMuPDF(1.28+)把模块名从 `fitz` 统一成 `pymupdf`,`fitz` 作为旧别名保留但已弃用。
- **解法**:`import fitz` → `import pymupdf`;`fitz.open(...)` → `pymupdf.open(...)`。pip 包名和 import 名现在一致了。
- **教训**:旧教程里 `import fitz` 是 PyMuPDF 的经典写法,现已换代;装完第三方库先看版本和弃用警告。

### 13. checkpointer 下裸 invoke 忘传 `thread_id`
- **现象**:`app.invoke(...)` 报 `ValueError: Checkpointer requires one or more of the following 'configurable' keys: thread_id, checkpoint_ns, checkpoint_id`。
- **根因**:图挂了 `MemorySaver` 后,每次 `invoke` 都要指定「存到哪个会话」(`configurable.thread_id`),不传就没法持久化。
- **解法**:`app.invoke(state, config={"configurable": {"thread_id": "1"}})`。
- **教训**:checkpointer 是「状态外置」,`thread_id` 就是会话隔离的键;多用户时每个用户一个 `thread_id`。这也是 `main.py` 里要传 `config` 的原因。

### 14. `if __name__ == "main"` 写错,CLI 静默退出
- **现象**:`python -m contract_review.supervisor samples/contract.txt` 打印 `Loading weights` 后直接退出,无任何输出、无报错。
- **根因**:入口守卫写成 `if __name__ == "main":`(少两个下划线)。直接运行脚本时 `__name__` 的值是 `"__main__"`,和 `"main"` 永不相等,条件恒为 False,`main()` 从未被调用 → import 完(触发 embedder 加载)就静默退出。
- **解法**:`if __name__ == "__main__":`。
- **教训**:`__main__` 是双下划线魔法常量,`"main"` 只是普通字符串,两者不相等。这是「入口守卫」的第二类变体(第一类是「忘了包守卫」,见前文 CLI 入口相关踩坑),两个都是「能 import 通过、但运行没反应」的静默故障。

### 15. 用 clause 文本相似度去重,误杀「同条款不同风险点」
- **现象**:去重后,同一条款的两个不同风险(如「赔偿全部损失」条款的「责任不对等」和「条款模糊」)只剩一个,真风险被吞。
- **根因**:去重只看 clause(条款文本),把「同一条款」误当「同一风险点」;而「同一风险点」= 同一条款 + 同一个风险角度,后者要靠 risk_type 判。
- **解法**:判据改两段式——`clause` 子串(同一条款)+ `risk_type` embedding 相似度(同一风险点),两个都命中才去重。
- **教训**:去重语义是「风险点级」不是「条款级」;embedding 对长文本复述(如 reason)无区分度(共同词主导),短标签(如 risk_type)才有区分度。合同审核「宁漏勿杀」——多留一条是烦,少留一条是事故。

### 16. `Send` 空 dict 扇出,分支节点拿不到父 state
- **现象**:`python -m contract_review.supervisor_parallel` 报 `KeyError: 'contract_text'`,发生在 expert 节点。
- **根因**:`Send(节点, arg)` 的第二个参数是**该分支节点收到的完整输入 state**,不是「额外增量」;传 `{}` 时,分支节点拿到的就是空 dict,读 `contract_text` 自然 KeyError。fan-out 出去的分支不会自动继承父节点的主 state。
- **解法**:`Send(name, {"contract_text": state["contract_text"]})`,把该专家要用的字段显式塞进 arg。
- **教训**:`Send` 的 arg 语义 = 分支的完整输入,不是 merge 增量;误当增量传空 dict 是第一次用 `Send` 的经典坑。
