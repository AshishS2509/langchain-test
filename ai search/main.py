from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()

model = ChatOllama(model="qwen3.5:0.8b")
agent = create_agent(model=model, tools=[TavilySearch(max_results=5, topic="general")])
data = agent.invoke(
    {
        "messages": [
            "Give me current weather in nashik. Use degree celsius. Remove all other irrelevent information"
        ]
    }
)
print(data)
