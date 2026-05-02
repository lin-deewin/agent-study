import os
from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

def run_briefing_agent(query: str):
    """
    开发一个能通过搜索工具写简报的单体 Agent
    """

    # 第一步：初始化大脑 (LLM)
    # 使用 temperature=0 确保生成的简报逻辑严密，不会胡编乱造
    model = ChatAnthropic(
        model="deepseek-v4-pro",
        temperature=0,
        base_url="https://api.deepseek.com/anthropic",
        api_key=os.environ["DS_API_KEY"],
    )

    # 第二步：初始化工具 (Tools)
    # Tavily 是专为 AI 优化的搜索引擎
    search_tool = TavilySearch(max_results=5)
    tools = [search_tool]

    # 第三步：设定系统提示词 (System Prompt)
    # 这是 Agent 的“灵魂”，决定了它作为研究员的行为准则
    system_message = (
        "你是一名资深的硬件配置大师，知道最合理的电脑配置组装。"
        "你的任务是针对用户提出的主题，通过搜索引擎获取最新、最准确的信息。"
        "要求：\n"
        "1. 必须根据搜索到的实际数据撰写简报。\n"
        "2. 简报应包含：目前的硬件价格、硬件性能(要用最权威的软件评测结果)，硬件参数。\n"
        "3. 必须列出参考的信息来源。"
    )

    # 第四步：构建 Agent
    # create_agent 会自动处理：
    # 思考(Thought) -> 调用工具(Action) -> 观察结果(Observation) 的循环
    agent_executor = create_agent(model, tools, system_prompt=system_message)

    # 第五步：执行任务
    print(f"🚀 Agent 正在处理请求: {query}\n")
    
    inputs = {"messages": [("user", query)]}
    
    # 使用流式输出，让你看到 Agent 的思考过程
    with open("out.ext", "w") as f:
        for event in agent_executor.stream(inputs, stream_mode="values"):
            f.write(str(event) + "\n")

            last_message = event["messages"][-1]
            # 我们只打印 Agent 的文字回复（过滤掉工具调用的中间代码）

            if last_message.type == "ai" and last_message.content:
                print(f"--- Agent 回复 ---\n{last_message.content}\n")

if __name__ == "__main__":
    user_topic = "分析2026性能最好的电脑主机，以及性价比最好的主机配置，可以包括志强平台，amd平台。可以把x99,平台也加上，顺便聊聊你的看法，最后给出结论。"
    run_briefing_agent(user_topic)
