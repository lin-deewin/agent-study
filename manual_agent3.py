"""
LangGraph 版 Agent + Human-in-the-Loop（人工审批）

基于 manual_agent2.py，在 call_tools 节点里加了 interrupt()，
让 Agent 每次调工具前暂停，等你审批通过后再执行。

和 manual_agent2.py 唯一的区别（除了新增的交互逻辑）：
    call_tools 节点开头多了 5 行 interrupt 代码

核心新概念：
    interrupt()     — 在节点内部暂停图执行，等待外部输入
    Command(resume) — 恢复暂停的图，把用户决定传回 interrupt() 的返回值
    checkpointer    — 暂停期间持久化 State（manual_agent2.py 已配置好）

交互流程：
    llm 决策要搜 → tools 节点入口 ⏸ 暂停 → 你看到工具清单 →
        Y: 批准 → 执行搜索 → 结果喂回 llm → 继续...
        N: 拒绝 → 返回拒绝消息 → llm 调整策略 →
        A: 改为全自动 → 后续不再暂停 →
    llm 决定不再搜索 → END → 生成 JSON
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
from langgraph.types import interrupt, Command


# ═══════════════════════════════════════════════════════════════
# 第 0 步：准备 LLM 和工具
# ═══════════════════════════════════════════════════════════════

class BriefingOutput(BaseModel):
    """硬件配置简报的结构化输出"""
    年份: str = Field(description="年份")
    厂家: list[str] = Field(description="标题")
    型号: str = Field(description="型号")
    频率: str = Field(description="频率")
    内存容量: str = Field(description="内存容量")
    价格: float = Field(description="价格")
    参考来源: list[str] = Field(description="参考来源")


model = ChatAnthropic(
    model_name="deepseek-v4-pro",
    temperature=0,
    max_tokens=8192,
    base_url="https://api.deepseek.com/anthropic",
    api_key=SecretStr(os.environ["DS_API_KEY"]),
    timeout=120,
) # pyright: ignore[reportCallIssue]

search_tool = TavilySearch(max_results=5)
tools_by_name = {search_tool.name: search_tool}

model_with_tools = model.bind_tools([search_tool])

system_prompt = (
    "你是一名资深的硬件配置大师，"
    "你的任务是针对用户提出的主题，通过搜索引擎获取最新、最准确的信息。"
    "要求：\n"
    "1. 必须根据搜索到的实际数据撰写简报。\n"
    "2. 去咸鱼，拼多多，京东等平台搜索。\n"
    "3. 必须列出参考的信息来源。\n"
    "4. 最终输出必须是一个标准 JSON。"
)


# ═══════════════════════════════════════════════════════════════
# 第 1 步：State（和 manual_agent2.py 一样）
# ═══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration_count: int
    auto_mode: bool  # True = 不再询问，全自动执行


# ═══════════════════════════════════════════════════════════════
# 第 2 步：节点函数
# ═══════════════════════════════════════════════════════════════

def call_llm(state: AgentState) -> dict:
    iteration = state.get("iteration_count", 0) + 1
    messages = state["messages"]

    print(f"\n>>> 第 {iteration} 轮 LLM 调用 <<<")
    print("  [思考] ", end="", flush=True)

    full_message: AIMessage | None = None
    for chunk in model_with_tools.stream(messages):
        # 只打印文本内容，过滤掉 thinking 块（减少控制台噪音）
        content = chunk.content
        if isinstance(content, list):
            text_parts = [
                b.get("text", "") if isinstance(b, dict) else ""
                for b in content
                if isinstance(b, dict) and b.get("type") in ("text", "text_delta")
            ]
            text = "".join(text_parts)
            if text:
                print(text, end="", flush=True)
        elif isinstance(content, str) and content:
            print(content, end="", flush=True)
        full_message = chunk if full_message is None else full_message + chunk # type: ignore

    print()

    # 在调用完LLM后，检查是否有工具调用计划划。
    # 没有的话就结束搜索阶段，有的话就进入工具调用节点。
    has_tools = bool(full_message.tool_calls)
    if not has_tools:
        print("  [决策] 不再需要工具，搜索阶段结束")
    else:
        tool_names = [tc["name"] for tc in full_message.tool_calls]
        print(f"  [决策] 计划调用工具: {', '.join(tool_names)}")

    return {
        "messages": [full_message],
        "iteration_count": iteration,
    }


def call_tools(state: AgentState) -> dict:
    """
    工具节点。新增 HITL 逻辑：
      入口处调用 interrupt() 暂停 → 等待用户审批 →
        批准 → 正常执行工具
        拒绝 → 返回 ToolMessage(拒绝原因)
        全自动 → 后续不再暂停
    """
    messages = state["messages"]
    last_message = messages[-1]
    tool_calls = last_message.tool_calls

    # ─── HITL: 非自动模式下，进入工具节点前暂停 ───
    auto_mode = state.get("auto_mode", False)
    if not auto_mode:
        # interrupt() 在这里暂停整个图。
        # 用户通过 Command(resume=decision) 恢复后，
        # decision 就是这个函数的返回值。
        decision = interrupt({
            "tool_calls": [
                {"name": tc["name"], "args": tc["args"]}
                for tc in tool_calls
            ],
            "iteration": state.get("iteration_count", 0),
        })

        # 根据用户决定处理
        if decision.get("action") == "reject":
            print("  [HITL] 用户拒绝，返回拒绝消息给 LLM")
            return {"messages": [
                ToolMessage(
                    content=f"用户拒绝执行此工具调用。请换一种方式搜索或直接基于已有信息回答。",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]}

        if decision.get("action") == "quit":
            print("  [HITL] 用户终止执行")
            return {"messages": [
                ToolMessage(
                    content="用户终止了搜索。请直接基于已有信息生成最终 JSON。",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]}

        if decision.get("action") == "auto":
            print("  [HITL] 用户切换为全自动模式")
            # 更新 auto_mode，后续不再询问
            return {"auto_mode": True}

        # action == "approve" 或默认：继续执行
        print("  [HITL] 用户批准 ✓")

    # ─── 执行工具（和 manual_agent2.py 一样） ───
    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        print(f"  → 执行: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

        tool = tools_by_name[tool_name]
        result = tool.invoke(tool_args)

        result_str = json.dumps(result, ensure_ascii=False, indent=2)
        preview = result_str[:200]
        print(f"  ← 返回: {preview}...")

        truncated = result_str[:3000]
        tool_messages.append(ToolMessage(content=truncated, tool_call_id=tool_id))

    return {"messages": tool_messages}


# ═══════════════════════════════════════════════════════════════
# 第 3 步：条件边
# ═══════════════════════════════════════════════════════════════

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"


# ═══════════════════════════════════════════════════════════════
# 第 4 步：构建图
# ═══════════════════════════════════════════════════════════════

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("llm", call_llm)
    workflow.add_node("tools", call_tools)

    workflow.set_entry_point("llm")

    workflow.add_conditional_edges("llm", should_continue, {
        "tools": "tools",
        "end": END,
    })

    workflow.add_edge("tools", "llm")

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════
# 第 5 步：交互式运行循环
# ═══════════════════════════════════════════════════════════════

def show_interrupt_info(interrupt_value):
    """展示暂停信息 —— Agent 想调什么工具"""
    tc_list = interrupt_value.get("tool_calls", [])
    iteration = interrupt_value.get("iteration", 0)
    print("\n" + "─" * 55)
    print(f"⚠️  Agent 第 {iteration} 轮请求调用以下工具：")
    for i, tc in enumerate(tc_list):
        print(f"   [{i+1}] {tc['name']}")
        print(f"       参数: {json.dumps(tc['args'], ensure_ascii=False)}")
    print("─" * 55)


def ask_user_decision():
    """询问用户决定"""
    while True:
        choice = input("\n📋 [Y]批准执行  [N]拒绝本次  [A]改为全自动  [Q]退出: ").strip().upper()
        if choice == "Y":
            return {"action": "approve"}
        elif choice == "N":
            return {"action": "reject"}
        elif choice == "A":
            return {"action": "auto"}
        elif choice == "Q":
            return {"action": "quit"}
        print("  无效选择，请输入 Y / N / A / Q")


def run_agent_loop(user_query: str, max_iterations: int = 10):
    print(f"用户请求: {user_query}\n")
    print("=" * 60)
    print("模式: Human-in-the-Loop（每次工具调用前需要审批）")
    print("=" * 60)

    graph = build_graph()

    config = {
        "configurable": {"thread_id": "session-hitl"},
        "recursion_limit": max_iterations * 2,
    }

    initial_state = {
        "messages": [SystemMessage(content=system_prompt), HumanMessage(content=user_query)],
        "iteration_count": 0,
        "auto_mode": False,
    }

    final_messages = None

    # ─── 主循环：stream 生成器 + 中断处理 ─── 
    # 思路：维护一个 stream 生成器。正常事件 → 更新状态。
    # 检测到 __interrupt__ → 询问用户 → 用 Command 创建新生成器 → 继续。

    current_state = dict(initial_state)

    try:
        # 启动第一次 stream
        stream_gen = graph.stream(current_state, config, stream_mode="values")

        while True:
            try:
                for event in stream_gen:
                    # ─── 检测中断 ───
                    if "__interrupt__" in event:
                        interrupt_obj = event["__interrupt__"][0]
                        interrupt_value = interrupt_obj.value
                        show_interrupt_info(interrupt_value)
                        decision = ask_user_decision()

                        print()  # 空行分隔
                        # 用 Command 恢复，创建新的生成器
                        stream_gen = graph.stream(
                            Command(resume=decision), config, stream_mode="values"
                        )
                        break  # 跳出当前 for，用新生成器继续 while

                    # ─── 正常事件 ───
                    current_state.update(event)
                    iteration = event.get("iteration_count", 0)
                    if iteration:
                        print(f"  [状态] 消息数: {len(event['messages'])} 条")
                else:
                    # for-else: stream 正常结束，没有 break
                    final_messages = current_state.get("messages", [])
                    break

            except Exception as inner_e:
                inner_msg = str(inner_e)
                if "Recursion" in inner_msg or "recursion" in inner_msg:
                    print(f"\n  [系统] 达到最大步数限制，搜索阶段结束")
                    final_messages = current_state.get("messages", [])
                    break
                raise  # 非 recursion 错误，抛到外层

    except Exception as e:
        error_msg = str(e)
        print(f"\n  [系统] 图执行异常: {error_msg[:150]}")
        final_messages = current_state.get("messages", [])

    # ═════════════════════════════════════════════════════════
    # 阶段二：结构化 JSON 生成
    # ═════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("[阶段二] 生成 JSON 简报\n")

    if not final_messages:
        final_messages = current_state.get("messages", [])

    # 清理未执行的 tool_calls：如果最后一条 AIMessage 有 tool_calls
    # 但没有对应的 ToolMessage，API 会拒绝。移除这些"悬空"的 AIMessage。
    cleaned = []
    pending_ids = set()
    # 先收集所有已完成工具调用的 ID
    for msg in final_messages:
        if isinstance(msg, ToolMessage):
            pending_ids.add(msg.tool_call_id)
    # 重建消息列表，跳过含有未执行 tool_calls 的 AIMessage
    for msg in final_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            unanswered = [tc["id"] for tc in msg.tool_calls if tc["id"] not in pending_ids]
            if unanswered:
                print(f"  [清理] 移除含 {len(unanswered)} 个未执行工具调用的 AIMessage")
                continue
        cleaned.append(msg)
    final_messages = cleaned

    format_instruction = HumanMessage(
        content=(
            "请根据以上搜索结果，直接输出一个标准 JSON 对象，"
            "字段包含：标题、关键词、核心配置推荐、性能对比分析、价格分析、结论与建议、参考来源。"
            "只输出 JSON，不要包裹在 ```json``` 代码块中，不要加任何解释文字。"
        )
    )
    final_messages.append(format_instruction)

    response = None
    try:
        response = model.invoke(final_messages)
        raw = response.content

        if isinstance(raw, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw
            )
        else:
            content = str(raw)

        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:]) if len(lines) > 1 else content
        if content.endswith("```"):
            content = content[:-3].strip()

        return json.loads(content)
    except Exception as e:
        raw_info = str(response.content)[:500] if response is not None else "response 未生成"
        return {"error": str(e), "raw": raw_info}


# ═══════════════════════════════════════════════════════════════
# 第 6 步：运行
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    user_topic = "统计内存条的历史价格数据， 统计不同厂家的内存，不同内存频率的，不同容量大小，在不同的电商平台上的价格， 统计时间从2024年一月到2026年5月，每个月的价格变动。"
    result = run_agent_loop(user_topic)

    print("\n" + "=" * 60)
    print("最终输出（结构化 JSON）:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
