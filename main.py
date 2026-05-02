import json
import os
from pydantic import BaseModel, Field, SecretStr
from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain.agents import create_agent


# 结构化输出：定义简报的标准 JSON 格式
class BriefingOutput(BaseModel):
    """硬件配置简报的结构化输出"""

    标题: str = Field(description="简报标题")
    关键词: list[str] = Field(description="3-5个核心关键词")
    核心配置推荐: str = Field(description="推荐的核心硬件配置及理由")
    性能对比分析: str = Field(description="各平台性能对比（含权威评测数据）")
    价格分析: str = Field(description="各配置当前市场价格分析")
    结论与建议: str = Field(description="最终结论和购买建议")
    参考来源: list[str] = Field(description="参考信息来源URL列表")

def run_briefing_agent(query: str):
    """
    开发一个能通过搜索工具写简报的单体 Agent
    """

    # 第一步：初始化大脑 (LLM)
    # 使用 temperature=0 确保生成的简报逻辑严密，不会胡编乱造
    model = ChatAnthropic(
        model_name="deepseek-v4-pro",
        temperature=0,
        base_url="https://api.deepseek.com/anthropic",
        api_key=SecretStr(os.environ["DS_API_KEY"]),
        timeout=60,
        stop=["\n"],
    )

    # 第二步：初始化工具 (Tools)
    # Tavily 是专为 AI 优化的搜索引擎
    search_tool = TavilySearch(max_results=5)
    tools = [search_tool]

    # 第三步：设定系统提示词 (System Prompt)
    # 这是 Agent 的"灵魂”，决定了它作为研究员的行为准则
    system_message = (
        "你是一名资深的硬件配置大师，知道最合理的电脑配置组装。 "
        "你的任务是针对用户提出的主题，通过搜索引擎获取最新、最准确的信息。 "
        "要求：\n"
        "1. 必须根据搜索到的实际数据撰写简报。\n"
        "2. 简报应包含：目前的硬件价格、硬件性能(要用最权威的软件评测结果)，硬件参数。\n"
        "3. 必须列出参考的信息来源。\n"
        "4. 最终输出必须是一个标准 JSON，字段包含：标题、关键词、核心配置推荐、性能对比分析、价格分析、结论与建议、参考来源。"
    )

    # 第四步：构建 Agent
    # response_format 让 Agent 自动返回结构化 JSON，不再是一段自由文本
    agent_executor = create_agent(
        model, tools,
        system_prompt=system_message,
        response_format=BriefingOutput,
    )

    # 第五步：执行任务
    print(f"🚀 Agent 正在处理请求: {query}\n")
    
    inputs = {"messages": [("user", query)]}
    
    # 使用流式输出，让你看到 Agent 的思考过程
    with open("out.txt", "w") as f:
        for event in agent_executor.stream(inputs, stream_mode="values"): # type: ignore
            f.write(str(event) + "\n")

            last_message = event["messages"][-1]
            # 结构化输出：最终回复是合法 JSON
            if last_message.type == "ai" and last_message.content:
                content = last_message.content
                # 尝试解析为 JSON 并美化打印
                try:
                    print(f"--- Agent 结构前回复 ---\n{str(content)}")
                    parsed = json.loads(content)
                    print(f"--- Agent 结构化回复 ---\n{json.dumps(parsed, ensure_ascii=False, indent=2)}\n")
                except (json.JSONDecodeError, TypeError):
                    print(f"--- Agent 回复 ---\n{content}\n")

if __name__ == "__main__":
    user_topic = "分析2026性能最好的电脑主机，以及性价比最好的主机配置，可以包括志强平台，amd平台。可以把x99,平台也加上，顺便聊聊你的看法，最后给出结论。"
    run_briefing_agent(user_topic)
