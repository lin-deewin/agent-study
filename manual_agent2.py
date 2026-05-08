"""
LangGraph 版 Agent 循环 —— 把 while 循环建模成有向图

manual_agent.py 里的 Agent 循环：
    while True:
        llm 决策 → 有工具调用? → 执行工具 → 回到开头
                  → 没有? → 结束

在 LangGraph 中，这被抽象为一张图：
    ┌───────────┐  有tool_calls     ┌─────────────┐
    │ llm 节点  ├──────────────→    │ tools 节点   │
    └────┬─────┘                   └──────┬──────┘
        │ 无tool_calls                    │
        ↓                                 │
      (END)  ←───────────────────────────┘  (自动回到 llm)

核心概念：
  State   — 在节点之间流动的共享数据（消息列表、迭代计数等）
  Node    — 一个执行单元（调用 LLM、执行工具）
  Edge    — 节点之间的连线（普通边：固定走向；条件边：根据状态决定走向）
  Reducer — 定义 State 中每个字段如何合并新值（add_messages 是内置的消息合并器）
"""

import json
import os
from typing import Annotated, TypedDict, Literal

from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field, SecretStr

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ═══════════════════════════════════════════════════════════════
# 第 0 步：准备 LLM 和工具（和 manual_agent.py 完全一样）
# ═══════════════════════════════════════════════════════════════

class BriefingOutput(BaseModel):
    """硬件配置简报的结构化输出"""
    标题: str = Field(description="简报标题")
    关键词: list[str] = Field(description="3-5个核心关键词")
    核心配置推荐: str = Field(description="推荐的核心硬件配置及理由")
    性能对比分析: str = Field(description="各平台性能对比（含权威评测数据）")
    价格分析: str = Field(description="各配置当前市场价格分析")
    结论与建议: str = Field(description="最终结论和购买建议")
    参考来源: list[str] = Field(description="参考信息来源URL列表")


model = ChatAnthropic(
    model_name="deepseek-v4-pro",
    temperature=0,
    base_url="https://api.deepseek.com/anthropic",
    api_key=SecretStr(os.environ["DS_API_KEY"]),
    timeout=120,
)

search_tool = TavilySearch(max_results=5)
tools_by_name = {search_tool.name: search_tool}

model_with_tools = model.bind_tools([search_tool])
# model_structured 在这里不需要了，DeepSeek API 不支持 with_structured_output
# model_structured = model.with_structured_output(BriefingOutput)

system_prompt = (
    "你是一名资深的硬件配置大师，知道最合理的电脑配置组装。"
    "你的任务是针对用户提出的主题，通过搜索引擎获取最新、最准确的信息。"
    "要求：\n"
    "1. 必须根据搜索到的实际数据撰写简报。\n"
    "2. 简报应包含：目前的硬件价格、硬件性能（要用最权威的软件评测结果），硬件参数。\n"
    "3. 必须列出参考的信息来源。\n"
    "4. 最终输出必须是一个标准 JSON。"
)


# ═══════════════════════════════════════════════════════════════
# 第 1 步：定义 State —— 图中流动的数据结构
# ═══════════════════════════════════════════════════════════════
# State 是 LangGraph 的核心概念。每个节点读取 State、返回部分更新，
# LangGraph 自动用 Reducer 合并新旧 State。
#
# add_messages：内置的消息列表 Reducer，会自动处理消息的追加（而非覆盖）
# iteration_count：用自定义 reducer，追踪搜索迭代轮数

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 消息历史，自动追加
    iteration_count: int                      # 当前迭代次数


# ═══════════════════════════════════════════════════════════════
# 第 2 步：定义节点 —— 图中的执行单元
# ═══════════════════════════════════════════════════════════════

def call_llm(state: AgentState) -> dict:
    """
    LLM 节点：调用模型，返回 AI 消息。
    对应 manual_agent.py 中 for 循环里的 stream() 调用。

    State 输入 → 调用 LLM → 返回 {"messages": [AIMessage], "iteration_count": n+1}
    """
    iteration = state.get("iteration_count", 0) + 1
    messages = state["messages"]

    print(f"\n>>> 第 {iteration} 轮 LLM 调用 <<<")
    print("  [思考] ", end="", flush=True)

    full_message: AIMessage | None = None
    for chunk in model_with_tools.stream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
        full_message = chunk if full_message is None else full_message + chunk

    print()

    has_tools = bool(full_message.tool_calls)
    if not has_tools:
        print("  [决策] 不再需要工具，搜索阶段结束")
    else:
        tool_names = [tc["name"] for tc in full_message.tool_calls]
        print(f"  [决策] 需要调用工具: {', '.join(tool_names)}")

    # 返回的 dict 会被 LangGraph 自动合并到 State
    # messages 用 add_messages 合并，iteration_count 直接覆盖
    return {
        "messages": [full_message],
        "iteration_count": iteration,
    }


def call_tools(state: AgentState) -> dict:
    """
    工具节点：执行最近一条 AI 消息中的所有 tool_calls。
    对应 manual_agent.py 中 for tool_call in full_message.tool_calls 的循环。

    遍历 tool_calls → 执行工具 → 截断结果 → 返回 ToolMessage 列表
    """
    messages = state["messages"]
    last_message = messages[-1]

    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        print(f"  → 调用工具: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

        tool = tools_by_name[tool_name]
        result = tool.invoke(tool_args)

        # 搜索结果可能是 dict 或 list，统一转成 JSON 字符串
        result_str = json.dumps(result, ensure_ascii=False, indent=2)
        preview = result_str[:200]
        print(f"  ← 工具返回: {preview}...")

        truncated = result_str[:3000]
        tool_messages.append(ToolMessage(content=truncated, tool_call_id=tool_id))

    return {"messages": tool_messages}


# ═══════════════════════════════════════════════════════════════
# 第 3 步：条件边 —— 决定下一步走哪个节点
# ═══════════════════════════════════════════════════════════════

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    检查最后一条 AI 消息是否包含 tool_calls。
    有 → 走 tools 节点
    无 → 走 END（图结束）
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"


# ═══════════════════════════════════════════════════════════════
# 第 4 步：构建图 —— 把节点和边组装起来
# ═══════════════════════════════════════════════════════════════

def build_graph():
    """
    构建 LangGraph 图：

         START
           │
           ▼
        ┌──────┐  有tool_calls   ┌───────┐
        │ llm  ├────────────────→│ tools │
        └──┬───┘                 └───┬───┘
           │ 无tool_calls            │
           ▼                         │
         (END)  ←────────────────────┘

    和 manual_agent.py 的 while 循环完全等价。
    """
    workflow = StateGraph(AgentState)

    # 注册节点："llm" 和 "tools" 两个执行单元
    workflow.add_node("llm", call_llm)
    workflow.add_node("tools", call_tools)

    # 入口：从 llm 节点开始
    workflow.set_entry_point("llm")

    # 条件边：llm 执行完后，根据 should_continue 的结果分流
    workflow.add_conditional_edges("llm", should_continue, {
        "tools": "tools",  # 走向 tools 节点
        "end": END,        # 图结束
    })

    # 普通边：tools 执行完后，总是回到 llm
    workflow.add_edge("tools", "llm")

    # checkpointer 让图在执行过程中自动保存快照，
    # 支持断点续传和 human-in-the-loop
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════
# 第 5 步：运行 —— 和图交互
# ═══════════════════════════════════════════════════════════════

def run_agent_loop(user_query: str, max_iterations: int = 10):
    print(f"用户请求: {user_query}\n")
    print("=" * 60)

    graph = build_graph()

    # ═════════════════════════════════════════════════════════
    # 阶段一：图执行 —— 自动循环 LLM ↔ Tools
    # ═════════════════════════════════════════════════════════
    #
    # .stream() 每次状态变化时 yield 一个事件。
    # stream_mode="values" 表示每步输出完整的当前 State。
    #
    # thread_id 是 checkpointer 的会话标识，同一个 thread_id
    # 的多次调用共享状态（实现跨请求记忆）。
    #
    # LangGraph 内置了 max_steps 保护，等效于 max_iterations。

    config = {"configurable": {"thread_id": "session-1"}, "recursion_limit": max_iterations * 2}

    final_messages = []
    try:
        for event in graph.stream(
            {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]},
            config,
            stream_mode="values",
        ):
            # event 是当前的完整 AgentState
            final_messages = event["messages"]
            iteration = event.get("iteration_count", 0)
            if iteration:
                print(f"  [状态] 消息历史长度: {len(final_messages)} 条, 迭代: {iteration}")

    except Exception as e:
        print(f"\n⚠️ 图执行出错或达到最大步数: {e}")

    # ═════════════════════════════════════════════════════════
    # 阶段二：结构化输出
    # ═════════════════════════════════════════════════════════
    #
    # DeepSeek 的 Anthropic 兼容 API 不支持 with_structured_output(),
    # 所以在最后加一条消息，直接要求模型输出合法 JSON，然后手动解析。

    print("\n" + "=" * 60)
    print("[阶段二] 生成 JSON 简报\n")

    format_instruction = HumanMessage(
        content=(
            "请根据以上搜索结果，直接输出一个标准 JSON 对象，"
            "字段包含：标题、关键词、核心配置推荐、性能对比分析、价格分析、结论与建议、参考来源。"
            "只输出 JSON，不要包裹在 ```json``` 代码块中，不要加任何解释文字。"
        )
    )
    final_messages.append(format_instruction)

    try:
        response = model.invoke(final_messages)
        raw = response.content

        # DeepSeek 返回的 content 可能是 list[dict]（content blocks）或 str
        if isinstance(raw, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw
            )
        else:
            content = str(raw)

        # 去掉可能的 markdown 代码块标记
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:]) if len(lines) > 1 else content
        if content.endswith("```"):
            content = content[:-3].strip()

        return json.loads(content)
    except Exception as e:
        return {"error": str(e), "raw": str(response.content)[:500]}


# ═══════════════════════════════════════════════════════════════
# 第 6 步：运行
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    user_topic = "分析AMD平台最新的cpu详细参数，以及目前的市场价格。"
    result = run_agent_loop(user_topic)

    print("\n" + "=" * 60)
    print("最终输出（结构化 JSON）:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
