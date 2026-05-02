"""
手动 Agent 循环 —— 揭示 create_agent 底层的工作原理

Agent 的本质不是什么魔法，就是一个 while 循环：
  1. LLM 接收消息，决定是"直接回答"还是"调用工具"
  2. 如果要调工具 → 执行工具 → 把结果追加到消息列表 → 回到步骤 1
  3. 如果不调工具 → 输出最终回复 → 循环结束

这个文件用最少的代码把上面的循环写出来，方便和 main.py（黑盒版）对比学习。
"""

import json
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from pydantic import SecretStr

# ──────────────────────────────────────────────────────────
# 第 0 步：准备 LLM 和工具（和 main.py 一样）
# ──────────────────────────────────────────────────────────

model = ChatAnthropic(
    model_name="deepseek-v4-pro",
    temperature=0,
    base_url="https://api.deepseek.com/anthropic",
    api_key=SecretStr(os.environ["DS_API_KEY"]),
    timeout=60,
)

search_tool = TavilySearch(max_results=5)

# 把工具列表转换成 LLM 能理解的格式（JSON Schema）
# bind_tools 是 LangChain 的便捷方法，等价于手动构造 tools 参数
model_with_tools = model.bind_tools([search_tool])

# 方便按名字查找工具
tools_by_name = {tool.name: tool for tool in [search_tool]}

# ──────────────────────────────────────────────────────────
# 第 1 步：系统提示词
# ──────────────────────────────────────────────────────────

system_prompt = (
    "你是一名资深的硬件配置大师，知道最合理的电脑配置组装。"
    "你的任务是针对用户提出的主题，通过搜索引擎获取最新、最准确的信息。"
    "要求：\n"
    "1. 必须根据搜索到的实际数据撰写简报。\n"
    "2. 简报应包含：目前的硬件价格、硬件性能（要用最权威的软件评测结果），硬件参数。\n"
    "3. 必须列出参考的信息来源。\n"
    "4. 最终输出必须是一个标准 JSON。"
)

# ──────────────────────────────────────────────────────────
# 第 2 步：手写 Agent 循环（核心！）
# ──────────────────────────────────────────────────────────

def run_agent_loop(user_query: str, max_iterations: int = 10):
    """
    Agent 循环的本质：LLM 决策 → 执行工具 → 喂回结果 → 再决策 → ... → 输出最终答案
    """

    # 初始化消息列表：system + user（使用 LangChain 消息对象，不是裸 tuple）
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]

    print(f"用户请求: {user_query}\n")
    print("=" * 60)

    for turn in range(1, max_iterations + 1):
        print(f"\n>>> 第 {turn} 轮 LLM 调用 <<<")

        # ─── 2a: 调用 LLM ───
        # LLM 返回一个 AIMessage，可能包含 tool_calls（表示它想调工具）
        # 也可能只包含 content（表示它想直接回答）
        response: AIMessage = model_with_tools.invoke(messages)

        # ─── 2b: 如果 LLM 直接输出文本（不调工具）→ 循环结束 ───
        if not response.tool_calls:
            print("LLM 决策: 不再需要工具，直接输出最终回复\n")
            return response.content

        # ─── 2c: LLM 决定调用工具 ───
        print(f"LLM 决策: 需要调用 {len(response.tool_calls)} 个工具")

        # 把 LLM 的这轮回复追加到消息历史（包含 tool_calls 指令）
        messages.append(response)

        # ─── 2d: 逐个执行工具，把结果封装成 ToolMessage ───
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"  → 调用工具: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

            # 查找并执行工具
            tool = tools_by_name[tool_name]
            tool_result = tool.invoke(tool_args)

            # 截断显示（搜索结果可能很长）
            result_preview = str(tool_result)[:200]
            print(f"  ← 工具返回: {result_preview}...")

            # 把工具结果追加到消息历史
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))

        print(f"  消息历史长度: {len(messages)} 条")

    # 达到最大迭代次数仍未结束
    return "⚠️ Agent 达到最大迭代次数，未能完成简报。"


# ──────────────────────────────────────────────────────────
# 第 3 步：运行
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    user_topic = "分析2026性能最好的电脑主机，以及性价比最好的主机配置。简要回答。"
    result = run_agent_loop(user_topic)

    print("\n" + "=" * 60)
    print("最终输出:")
    print(result)
