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
