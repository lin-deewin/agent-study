"""
手动 Agent 循环 —— 揭示 create_agent 底层的工作原理

Agent 的本质不是什么魔法，就是一个 while 循环：
  1. LLM 接收消息，决定是"直接回答"还是"调用工具"
  2. 如果要调工具 → 执行工具 → 把结果追加到消息列表 → 回到步骤 1
  3. 如果不调工具 → 输出最终回复 → 循环结束

v2 新增：
  - 流式输出：用 .stream() 逐 token 看到 LLM 的思考过程
  - 结构化输出：用 with_structured_output() 强制输出合法 JSON
"""

import json
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from pydantic import BaseModel, Field, SecretStr

# ──────────────────────────────────────────────────────────
# 结构化输出 Schema（和 main.py 一样）
# ──────────────────────────────────────────────────────────

class BriefingOutput(BaseModel):
    """硬件配置简报的结构化输出"""
    标题: str = Field(description="简报标题")
    关键词: list[str] = Field(description="3-5个核心关键词")
    核心配置推荐: str = Field(description="推荐的核心硬件配置及理由")
    性能对比分析: str = Field(description="各平台性能对比（含权威评测数据）")
    价格分析: str = Field(description="各配置当前市场价格分析")
    结论与建议: str = Field(description="最终结论和购买建议")
    参考来源: list[str] = Field(description="参考信息来源URL列表")


# ──────────────────────────────────────────────────────────
# 第 0 步：准备 LLM 和工具
# ──────────────────────────────────────────────────────────

model = ChatAnthropic(
    model_name="deepseek-v4-pro",
    temperature=0,
    base_url="https://api.deepseek.com/anthropic",
    api_key=SecretStr(os.environ["DS_API_KEY"]),
    timeout=120,
)

search_tool = TavilySearch(max_results=5)

# 两个模型变体，各司其职：
#   model_with_tools  → 带工具，用于搜索循环（流式）
#   model_structured  → 不带工具，用于最终 JSON 格式化
model_with_tools = model.bind_tools([search_tool])
model_structured = model.with_structured_output(BriefingOutput)

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
# 第 2 步：手写 Agent 循环 —— 加入流式 + 结构化输出
# ──────────────────────────────────────────────────────────

def run_agent_loop(user_query: str, max_iterations: int = 10):
    print(f"🚀 Agent 正在处理请求: {user_query}\n")
    """
    Agent 循环的本质：LLM 决策 → 执行工具 → 喂回结果 → 再决策 → ... → 输出最终答案

    流程分两阶段：
      阶段一（流式循环）：带工具的 LLM，逐 token 打印思考过程，必要时调用搜索工具
      阶段二（结构化收尾）：不带工具，但强制输出符合 BriefingOutput Schema 的 JSON
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]

    print(f"用户请求: {user_query}\n")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════
    # 阶段一：流式工具调用循环
    # ═══════════════════════════════════════════════════════

    for turn in range(1, max_iterations + 1):
        print(f"\n>>> 第 {turn} 轮 LLM 调用 <<<")

        # ─── 2a: 流式调用 LLM ───
        # .stream() 返回一个迭代器，每产生一个 token 就 yield 一个 AIMessageChunk
        # 我们需要把所有 chunk 拼起来，才能得到完整的 tool_calls 信息
        print("  [思考] ", end="", flush=True)
        full_message: AIMessage | None = None

        for chunk in model_with_tools.stream(messages):
            # 逐 token 打印，看到 LLM 的实时思考
            if chunk.content:
                print(chunk.content, end="", flush=True)

            # 累加 chunk：AIMessageChunk + AIMessageChunk → AIMessageChunk
            # 这会自动合并 content 和 tool_call_chunks
            full_message = chunk if full_message is None else full_message + chunk

        print()  # 换行

        # ─── 2b: 如果 LLM 直接输出文本（不调工具）→ 搜索阶段结束 ───
        if not full_message.tool_calls:
            print("  [决策] 不再需要工具，搜索阶段结束\n")
            # 把这轮 LLM 的回复也追加到消息历史（作为上下文）
            messages.append(full_message)
            break

        # ─── 2c: LLM 决定调用工具 ───
        print(f"  [决策] 需要调用 {len(full_message.tool_calls)} 个工具")

        # 把 LLM 的这轮回复追加到消息历史（包含 tool_calls 指令）
        messages.append(full_message)

        # ─── 2d: 逐个执行工具，把结果封装成 ToolMessage ───
        for tool_call in full_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"  → 调用工具: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

            tool = tools_by_name[tool_name]
            tool_result = tool.invoke(tool_args)

            result_preview = str(tool_result)[:200]
            print(f"  ← 工具返回: {result_preview}...")

            # 截断过长结果，避免请求体膨胀导致超时
            truncated = str(tool_result)[:3000]
            messages.append(ToolMessage(content=truncated, tool_call_id=tool_id))

        print(f"  消息历史长度: {len(messages)} 条")

    else:
        # Python 的 for-else：循环没有被 break 中断 → 达到最大迭代次数
        print(f"\n⚠️ 达到最大迭代次数 {max_iterations}，强制结束搜索阶段")

    # ═══════════════════════════════════════════════════════
    # 阶段二：结构化输出 —— 强制生成合法 JSON
    # ═══════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("[阶段二] 用 with_structured_output() 生成 JSON 简报\n")

    # model_structured 没有工具，只接收消息，输出一个 BriefingOutput Pydantic 对象
    # 这里传完整消息历史（含所有搜索结果），让 LLM 有足够素材
    try:
        briefing: BriefingOutput = model_structured.invoke(messages)
        return briefing.model_dump()
    except Exception as e:
        # 如果模型输出的 JSON 无法解析为 BriefingOutput，会进这里
        return {"error": str(e), "raw": messages[-1].content if messages[-1].content else "无内容"}


# ──────────────────────────────────────────────────────────
# 第 3 步：运行
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    user_topic = "分析AMD平台最新的cpu详细参数，以及目前的市场价格。"
    result = run_agent_loop(user_topic)

    print("\n" + "=" * 60)
    print("最终输出（结构化 JSON）:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
