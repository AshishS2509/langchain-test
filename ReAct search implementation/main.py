from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()


@tool
def web_search(query: str):
    """Search the web and return a concise summary."""
    print("Searching:", query)
    return  tavily.search(query=query, max_results=5)


model = ChatOllama(model="qwen3.5:0.8b", temperature=0)
agent = create_agent(model=model, tools=[web_search])


def main():
    response = agent.invoke(
        {"messages": [HumanMessage(content="26 march 2026 gold price in India")]}
    )
    print(response)


if __name__ == "__main__":
    main()
