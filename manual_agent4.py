"""
LangGraph Store（跨会话记忆）—— 让 Agent 记住上一次对话的内容

基于 manual_agent.py 的硬件简报场景，新增 Store 实现跨线程记忆。
和 manual_agent3.py 的 checkpointer 不同：
    checkpointer → 线程内短期记忆（保存同一对话的状态）
    Store       → 跨线程长期记忆（不同对话之间共享的知识）

核心新概念：
    InMemoryStore  — 内存中的 KV 存储，按 namespace 组织
    store.put()    — 写入记忆（namespace, key, value）
    store.search() — 按 namespace 和 filter 查询记忆
    store.get()    — 按 namespace 和 key 读取单条记忆

使用场景：
    用户两次查询不同主题，Agent 从 Store 中回忆起上次的用户偏好
    和搜索结果，自动关联到新查询中。

交互流程：
    用户查询1 → 搜索 Intel CPU → 生成简报 → 写入 Store ✍️
    用户查询2 → 从 Store 读取上次偏好 📖 → 搜索兼容 GPU → 生成简报
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
from langgraph.store.memory import InMemoryStore


# ═══════════════════════════════════════════════════════════════
# 第 0 步：准备 LLM 和工具
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
    max_tokens=8192,
    base_url="https://api.deepseek.com/anthropic",
    api_key=SecretStr(os.environ["DS_API_KEY"]),
    timeout=120,
) # pyright: ignore[reportCallIssue]

search_tool = TavilySearch(max_results=5)
tools_by_name = {search_tool.name: search_tool}

model_with_tools = model.bind_tools([search_tool])
model_structured = model.with_structured_output(BriefingOutput)

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
# 第 1 步：State（和 manual_agent3.py 一样，无 auto_mode/iteration_count 干扰）
# ═══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str  # 用于区分不同用户的 Store namespace


# ═══════════════════════════════════════════════════════════════
# 第 2 步：Store —— 跨会话记忆的核心
# ═══════════════════════════════════════════════════════════════

# InMemoryStore: 所有记忆存内存里，进程重启就消失。
# 生产环境可以换成 PostgresStore / RedisStore。
store = InMemoryStore()


def save_memory(user_id: str, topic: str, findings: dict):
    """
    写入 Store：把本次搜索结果的关键信息存起来。

    Store 的组织结构：
      namespace = ("memories", user_id)  — 类似"文件夹"，按用户隔离
      key       = str                     — 这条记忆的唯一 ID
      value     = dict                    — 实际内容

    这里用搜索主题的 hash 做 key，避免同一主题重复存储。
    """
    namespace = ("memories", user_id)
    key = f"search_{hash(topic) % 100000}"

    store.put(
        namespace=namespace,
        key=key,
        value={
            "topic": topic,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "title": findings.get("标题", ""),
            "keywords": findings.get("关键词", []),
            "recommendation": findings.get("核心配置推荐", ""),
            "price_analysis": findings.get("价格分析", ""),
        },
    )
    print(f"  [Store] ✍️  已写入记忆: namespace={namespace}, key={key}")
    print(f"  [Store]    内容摘要: {findings.get('标题', '')[:60]}")


def recall_memories(user_id: str, current_topic: str = "") -> list[dict]:
    """
    从 Store 中读取过往记忆，按与 current_topic 的相关性排序。

    相关性打分：当前主题的词和历史记忆的 topic + keywords 之间的交集越大，分越高。
    完全不相关的记忆直接过滤掉。
    """
    namespace = ("memories", user_id)
    items = store.search(namespace)
    raw_memories = [item.value for item in items]

    if not raw_memories:
        print(f"  [Store] 📖 namespace={namespace} 暂无记忆")
        return []

    # 用当前主题的所有非虚词做匹配关键字集合
    query_words = {w for w in current_topic if len(w) >= 2}
    import re
    query_tokens = set(re.findall(r"[一-鿿]+|[a-zA-Z0-9]+", current_topic.lower()))

    def relevance(memory: dict) -> int:
        """计算记忆和当前查询的相关性分数"""
        memory_text = (memory.get("topic", "") + " " + " ".join(memory.get("keywords", []))).lower()
        memory_tokens = set(re.findall(r"[一-鿿]+|[a-zA-Z0-9]+", memory_text))
        return len(query_tokens & memory_tokens)

    # 按相关性排序，过滤掉完全无关的记忆
    scored = [(relevance(m), m) for m in raw_memories]
    scored = [(s, m) for s, m in scored if s > 0]
    scored.sort(key=lambda x: x[0], reverse=True)

    if scored:
        print(f"  [Store] 📖 从 namespace={namespace} 读取到 {len(scored)} 条相关记忆:")
        for score, m in scored:
            print(f"  [Store]    • (相关度 {score}) {m.get('topic', '未知')[:50]}")
    else:
        print(f"  [Store] 📖 namespace={namespace} 有 {len(raw_memories)} 条记忆，但与当前主题均不相关")

    return [m for _, m in scored]


def build_enriched_system_prompt(user_id: str, current_topic: str) -> str:
    """
    在基础 system_prompt 之上，追加从 Store 读取的历史记忆。
    让 LLM 知道用户之前查过什么、有什么偏好。
    """
    base = system_prompt
    memories = recall_memories(user_id, current_topic)

    if not memories:
        return base

    # 把历史记忆格式化为上下文
    context_parts = []
    for i, m in enumerate(memories):
        context_parts.append(
            f"## 历史记录 {i+1}\n"
            f"- 用户上次查询的主题: {m.get('topic', '未知')}\n"
            f"- 标题: {m.get('title', '未知')}\n"
            f"- 关键词: {', '.join(m.get('keywords', []))}\n"
            f"- 推荐配置: {m.get('recommendation', '无')[:100]}\n"
            f"- 价格分析: {m.get('price_analysis', '无')[:100]}\n"
        )

    history_block = (
        "\n\n---\n"
        "## 历史记忆（来自之前的对话）\n"
        "以下是你之前在同类查询中获得的信息，可以用于关联当前查询：\n\n"
        + "\n".join(context_parts)
        + "\n---\n"
        "请结合以上历史信息，在当前查询中做出更连贯的分析和推荐。"
    )

    return base + history_block


# ═══════════════════════════════════════════════════════════════
# 第 3 步：节点函数
# ═══════════════════════════════════════════════════════════════

def call_llm(state: AgentState) -> dict:
    messages = state["messages"]

    print(f"\n>>> LLM 调用 <<<")
    print("  [思考] ", end="", flush=True)

    full_message: AIMessage | None = None
    for chunk in model_with_tools.stream(messages):
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
        full_message = chunk if full_message is None else full_message + chunk  # type: ignore

    print()

    has_tools = bool(full_message.tool_calls)
    if not has_tools:
        print("  [决策] 不再需要工具，搜索阶段结束")
    else:
        tool_names = [tc["name"] for tc in full_message.tool_calls]
        print(f"  [决策] 计划调用工具: {', '.join(tool_names)}")

    return {"messages": [full_message]}


def call_tools(state: AgentState) -> dict:
    messages = state["messages"]
    last_message = messages[-1]
    tool_calls = last_message.tool_calls

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
# 第 4 步：条件边
# ═══════════════════════════════════════════════════════════════

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"


# ═══════════════════════════════════════════════════════════════
# 第 5 步：构建图
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
    return workflow.compile(checkpointer=checkpointer, store=store)


# ═══════════════════════════════════════════════════════════════
# 第 6 步：运行函数
# ═══════════════════════════════════════════════════════════════

def run_briefing(user_id: str, user_topic: str, thread_suffix: str = "main") -> dict:
    """
    在 LangGraph 图里执行搜索 → 结构化输出。

    和 manual_agent.py 的关键区别：
      - 第 1 行：build_enriched_system_prompt 从 Store 读取历史记忆
      - 最后一行：save_memory 把本次结果写入 Store
    """
    # ─── Store 读取：用历史记忆丰富系统提示词 ───
    enriched_prompt = build_enriched_system_prompt(user_id, user_topic)

    print(f"\n{'=' * 60}")
    print(f"用户: {user_id}")
    print(f"查询: {user_topic}")
    print(f"{'=' * 60}")

    graph = build_graph()

    config = {
        "configurable": {"thread_id": f"session-{user_id}-{thread_suffix}"},
        "recursion_limit": 20,
    }

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=enriched_prompt),
            HumanMessage(content=user_topic),
        ],
        "user_id": user_id,
    }

    final_state = None
    for event in graph.stream(initial_state, config, stream_mode="values"):
        final_state = event
        msg_count = len(event.get("messages", []))
        if msg_count:
            print(f"  [状态] 消息数: {msg_count} 条")

    # ─── 阶段二：结构化 JSON 生成 ───
    print(f"\n[阶段二] 生成 JSON 简报\n")

    messages = final_state.get("messages", [])

    try:
        briefing: BriefingOutput = model_structured.invoke(messages)
        result = briefing.model_dump()

        # ─── Store 写入：保存本次搜索结果 ───
        save_memory(user_id, user_topic, result)

        return result
    except Exception as e:
        error_result = {"error": str(e)}
        # 失败也写入 Store，记录尝试
        save_memory(user_id, user_topic, error_result)
        return error_result


# ═══════════════════════════════════════════════════════════════
# 第 7 步：演示 —— 两次调用展示跨会话记忆
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    user_id = "user_lqp"

    # ─── 第一轮：查 AMD CPU ───
    print("\n" + "█" * 60)
    print("█  第 1 次查询")
    print("█" * 60)

    result1 = run_briefing(
        user_id=user_id,
        user_topic="分析AMD平台最新的Ryzen CPU详细参数，以及目前的市场价格。",
        thread_suffix="round1",
    )

    print("\n" + "=" * 60)
    print("第 1 次查询结果（结构化 JSON）:")
    print(json.dumps(result1, ensure_ascii=False, indent=2))

    # ─── 显示 Store 中的内容 ───
    namespace = ("memories", user_id)
    stored = store.search(namespace)
    print(f"\n[Store 状态] namespace={namespace}: {len(stored)} 条记忆")
    for item in stored:
        print(f"  • key={item.key}, topic={item.value.get('topic', '?')[:60]}")

    # ─── 第二 轮：查兼容的 GPU（Store 已经有记忆了） ───
    print("\n\n" + "█" * 60)
    print("█  第 2 次查询（Store 中已有第 1 次的记忆）")
    print("█" * 60)

    result2 = run_briefing(
        user_id=user_id,
        user_topic="推荐与AMD Ryzen兼容的显卡，分析其性能和价格。",
        thread_suffix="round2",
    )

    print("\n" + "=" * 60)
    print("第 2 次查询结果（结构化 JSON）:")
    print(json.dumps(result2, ensure_ascii=False, indent=2))

    # ─── 最终 Store 状态 ───
    stored_final = store.search(namespace)
    print(f"\n\n[Store 最终状态] namespace={namespace}: {len(stored_final)} 条记忆")
    for item in stored_final:
        print(f"  • key={item.key}, topic={item.value.get('topic', '?')[:60]}")
