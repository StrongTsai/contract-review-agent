# 合同审核数字员工 · 语法与概念速查

> 维护说明:Python / LangGraph / LangChain 的语法或 API 概念问题,按「是什么 → 为什么/怎么用 → 例子」记在这里。后续遇到类似问题,Claude 主动追加,不用等用户提醒。

## Python 语法

### `str | None`(联合类型 / Union)
- **是什么**:一个变量可以是 `str`,也可以是 `None`(空)。
- **为什么**:合同里的「金额」「期限」可能没写,用 `None` 表示「缺失」,比硬塞空字符串更诚实。
- **例子**:`amount: str | None = Field(description="合同金额")`。

### `Field` + `Annotated`(Pydantic 字段元数据)
- **是什么**:给字段/类型「贴标签」。`Field(description=...)` 里的 `description` 会写进 JSON Schema,成为模型读到的「这个字段是什么意思」。
- **为什么**:模型只能看到 schema(字段名 + 类型 + description),看不到你的代码;description 是喂给模型的唯一字段说明。
- **用法**:两种贴法,本质相同——
  - Pydantic 模型字段:`amount: str = Field(description="合同金额")`
  - 孤立类型(如工具参数):`keyword: Annotated[str, Field(description="法律概念关键词")]`
- **其它**:`Field` 还能贴约束 `ge`/`le`/`min_length`(Pydantic 校验)、`default`(默认值),不只是 description。

### `TypedDict` + `total=False`
- **是什么**:定义「状态」的结构。`total=False` 表示字段不必全填,节点可只更新自己负责的字段。
- **为什么**:LangGraph 的 state 就是这种结构;`ContractState(MessagesState, total=False)` 里各字段由不同节点产出。
- **注意**:state 字段必须**显式声明**,checkpoint 只序列化声明的字段(没声明就拿不到)。

### `@cache` 装饰器(记忆化 / 单例)
- **是什么**:记忆化(memoization)——函数算过的结果按「入参」存起来,同样的入参下次直接返回缓存,不重算函数体。
- **`@` 是语法糖**:`@cache` + `def get_model()` 等价于 `get_model = cache(get_model)`,即「用 cache 包一层再赋回原名」。
- **为什么 llm.py 用它**:`get_model()` 无参,每次都是同一个 key。第一次真正创建 `ChatOpenAI`(贵),之后所有节点每次调用都直接返回**同一个**实例 → 单例模式,全程只建一个模型。
- **直观理解**:函数体里的 `print` 只会在「每种新入参」第一次触发,第二次同参调用不再打印。
- **两个边角**:
  - 入参必须可哈希(hashable),dict/list 这类不可哈希参数会报错;
  - `cache`(Python 3.9+) = `lru_cache(maxsize=None)`,**无上限**、结果永不淘汰。入参可能性很多时内存会涨,应改用 `@lru_cache(maxsize=128)` 限制并自动淘汰。

### `enumerate(seq, start=0)`
- **是什么**:遍历时同时拿「下标」和「元素」。
- **为什么**:报告里要「第几条风险」,`for i, f in enumerate(review.findings, 1)` 让 i 从 1 开始,不用手写计数器。
- **例子**:`for i, f in enumerate(findings, 1): print(f"{i}. {f}")` → `1. 第一条`。

### `argparse`(命令行参数)
- **是什么**:解析 `python main.py xxx --yyy` 这种命令行输入。
- **为什么**:CLI 入口要区分「文件路径」(位置参数)和「直接传文本」(`--text` 可选参数)。
- **要点**:
  - 位置参数 `parser.add_argument("file", nargs="?")` —— `?` 表示可缺省;
  - 可选参数 `parser.add_argument("--text")` —— 带 `--` 前缀,传值时写 `--text "..."`。

### `f()(x)` 双括号调用(可调用对象 / `__call__`)
- **是什么**:两次调用叠在一起写。`get_ocr_engine()(image_path)` = 先 `get_ocr_engine()` 拿到实例,再 `(image_path)` 调用这个实例。所以 `(image_path)` **不是** `get_ocr_engine` 的参数,而是调用它的**返回值**。
- **为什么实例能被「调用」**:Python 里一个对象的类定义了 `__call__` 方法,这个对象就「可调用」——`obj(x)` 等价 `obj.__call__(x)`。RapidOCR 就是这种设计,把实例做得像函数:
  ```python
  engine = RapidOCR()   # __init__:建实例、加载模型
  result = engine(img)  # __call__:真正的识别逻辑
  ```
- **对照已有的**:`get_embedder().encode(x)` 是「方法调用」(实例 `.` 方法);`get_ocr_engine()(x)` 是「实例调用」(实例有 `__call__`)。本质都是「先拿实例、再用它」,一个走方法、一个走 `__call__`。
- **可读性**:双括号难读,拆两行更清楚:`engine = get_ocr_engine()` → `result = engine(image_path)`。

## LangChain / LangGraph

### `MessagesState`(预定义消息状态)
- **从哪来**:`from langgraph.graph import MessagesState`(你在 graph.py 已 import)。
- **是什么**:LangGraph 预定义的 `TypedDict`,只有 `messages` 一个字段,定义大致是:
  ```python
  class MessagesState(TypedDict):
      messages: Annotated[list[AnyMessage], add_messages]
  ```
- **关键魔法 `add_messages` reducer**:`messages` 字段不是普通 `list`,而是挂了 `add_messages` 归并器。节点返回 `{"messages": [新消息]}` 时是**追加**到现有列表,而不是覆盖(顺带:同 id 消息会替换、RemoveMessage 会删除)。
- **为什么 agent 需要**:循环里 agent/tools 每次返回的新消息都追加,`state["messages"]` 就成了不断变长的完整对话历史,模型每次都能看到「之前的工具调用 + 结果」。
- **你已经在用**:`ContractState(MessagesState, total=False)` 就是继承它、免费拿到「会追加的 messages 字段」,再叠加自己的字段。

### `create_agent`(预置 agent,替代已弃用的 `create_react_agent`)
- **是什么**:LangChain v1 的 agent 工厂,`from langchain.agents import create_agent`,一行搭好「agent 节点 + 工具循环」。
- **弃用背景**:`langgraph.prebuilt.create_react_agent` 是 LangGraph v0 写法,已弃用;官方替代是 `create_agent`(底层仍跑 LangGraph,引入 middleware 中间件体系)。抄旧教程时先核对。
- **用法**:
  ```python
  from langchain.agents import create_agent
  app = create_agent(get_model(), tools=[search_law, query_case],
                     system_prompt="...", response_format=ReviewResult)
  ```
  注意:旧 `prompt=` 已改名 `system_prompt=`;`tools`/`response_format` 不变。
- **内部结构**:和旧 `create_react_agent` 一样,背后就是 agent 节点(bind_tools 模型)+ ToolNode + 条件边成环。手动搭一遍就懂它。
- **`response_format` 出结构化结果**:传 Pydantic 模型,循环里自由调工具、最后强制吐结构化对象;结果在 `result["structured_response"]`。底层实现是注入一个名为模型名的合成工具(如 `ReviewResult`),让模型「调用」它来输出结构化数据,再解析成对象——所以对 DeepSeek 这类非原生结构化输出模型,自动选了「函数调用模拟结构化输出」(等价于 `method="function_calling"`),照样能工作。

### `with_structured_output` vs `bind_tools`
- **相同点**:都是 function calling,给模型一个「输出格式」约束。
- **不同点**:
  - `with_structured_output(PydanticModel)` → 强制输出固定 JSON 结构(信息抽取);
  - `bind_tools([...])` → 给模型一箱工具,让它**决定**「调哪个工具 or 直接回答」(agent 决策)。
- **记法**:前者「把答案装进框里」,后者「发一箱工具让它自己挑」。

### `@tool` 装饰器
- **是什么**:把普通函数变成模型可调用的工具。
- **关键**:docstring 是喂给模型的「说明书」——写清「何时用 + 参数语义」,别写类型(类型模型从签名已经拿到了)。
- **例子**:
```python
@tool
def search_law(keyword: str) -> str:
    """检索合同相关法条。判断条款合法性时调用。
    keyword:法律概念关键词,如「违约金」。"""
    return ...
```

### `interrupt` + `Command(resume=...)`
- **`interrupt(msg)`**:在节点里暂停图、把控制权吐回调用方,msg 是给人工看的内容。
- **`Command(resume=值)`**:续跑,把「人工决定」传回图。
- **依赖**:裸跑必须挂 checkpointer(`MemorySaver`),`langgraph dev` 自动配。
- **判断是否中断**:看 `snapshot.tasks[].interrupts` 是否为空,别只看 `snapshot.next`(next 非空只表示「暂停」)。

### `get_state` vs `get_state_history`
- **`get_state(config)`**:当前状态,1 个快照——查「现在卡没卡、当前值是什么」。
- **`get_state_history(config)`**:完整历史,N 个快照、最新在前——调试回溯、审计、时间旅行。

## 机器学习 / 向量基础

### embedding 为什么是 512 维(向量维度)
- **维度是模型的出厂规格,不是你的选项**:`bge-small-zh-v1.5` = 512,`bge-base-zh` = 768,`bge-large-zh` = 1024,OpenAI `text-embedding-3-large` = 3072。选模型 = 选维度;维度是模型最后一层输出神经元的个数,训练时就焊死了,不能调。
- **为什么需要这么多维**:embedding 是「语义空间坐标」,维度 = 轴的数量。维度越多表达力越强,越能区分细微差别(「违约金过高」vs「违约金条款」能各占一块区域不挤)。2D 只能放平面上,512D 有 512 个轴。
- **为什么是 512 这种数**:2 的幂(128/256/512/768/1024)对硬件友好;「small」= 小而够用。
- **权衡**:维度越高表达越强、但越慢越占内存;选小模型 = 拿一点精度换速度。

### 为什么「每次执行都 Loading weights」(进程生命周期 + 两层缓存)
- **现象**:每跑一次 `python agent_demo.py` 都打印 `Loading weights: 71/71`。
- **根因**:每次 `python xxx.py` 都是全新进程,`@cache` 只在「同一个进程内」生效,进程一退缓存连同已加载模型全清空。而 `LAW_STORE = VectorStore(...)` 在模块顶层,`import` 那一刻就 `_embed` → `get_embedder()` → 从磁盘把模型读进内存,于是每次起进程都 load 一次。
- **两层缓存要分清**:
  - 磁盘缓存(HuggingFace):权重下载到 `~/.cache/huggingface/hub`,只下载一次,所以显示的是「加载」(快),不是「下载」。
  - 内存缓存(`@cache`):进程内复用同一个模型实例;进程一退就清空。
- **`@cache` 到底省什么**:没有它,`get_embedder()` 每次调用都 new 一个 `SentenceTransformer` → 每查一次就 reload 一次;有了它,进程内只 load 一次。
- **生产**:长驻服务(FastAPI)启动 load 一次、常驻复用;短命脚本每次起进程必 load 一次,属正常。

### RAG 的「切分(chunking)」——为什么你的代码里没写切分
- **你的代码没切分,是对的**:语料是「一条法条/一个判例 = 一个向量」。`self.docs = [f"{e['title']} {e['content']}" for e in entries]` 后 `_embed(self.docs)`,就是把 9 个字符串一起传给 `encode`,得到 `(9, 512)` 矩阵,每行一个 entry 的向量。切分粒度 = list 的每个元素。
- **没有「默认切分」**:`SentenceTransformer.encode` 只负责把你给的每个字符串算成一个向量,不会替你拆。给 9 个短字符串出 9 个向量;给 1 个超长字符串只出 1 个向量。
- **为什么这里不需要切**:每条法条/判例本身 1~3 句,是天然原子块,再切丢上下文、检索更噪。
- **什么时候必须切**:文档很长(整本合同、整部法典)时——① 一个向量表达不下长文语义,检索粒度太粗;② 硬约束:`bge-small-zh-v1.5` 的 `max_seq_length=512` token,超长文本被 tokenizer **静默截断**、后面内容白白丢掉。所以长文要先切块、再逐块 embed。
- **切分在管道里的位置**:文档 → 切分(chunking)→ 逐块 embed → 存向量 → query embed → 检索。你的项目跳过了切分,因为「文档」已是原子单位(按法条条目切,本身就是一种语义边界切分)。
- **面试要点**:chunk 大小和重叠(overlap)是经典调参点;「一条法条 = 一个 chunk」是合理的语义切分策略。
