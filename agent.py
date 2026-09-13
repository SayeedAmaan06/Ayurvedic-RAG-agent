"""
agent.py
--------
A minimal LangChain agent that:
1. Answers general health/life-improvement chat directly.
2. Uses a RAG tool to look up Ayurvedic home remedies from local documents.
3. Uses a DuckDuckGo search tool for anything it doesn't know / for finding
   product links to buy.

Built with the new LangChain v1 `create_agent` API (runs on LangGraph under
the hood) and plain `@tool`-decorated functions.

This file only defines the agent. FastAPI wraps it in app.py.
"""

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

DB_DIR = "chroma_db"

# ---------------------------------------------------------------------
# 1. Set up the RAG retriever (reads the vector store built by build_rag.py)
# ---------------------------------------------------------------------
# FastEmbed is used instead of sentence-transformers because it has no
# PyTorch dependency, keeping memory usage low (important for free-tier
# hosting on Render, which caps at 512MB RAM).
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})


@tool
def ayurvedic_knowledge_base(query: str) -> str:
    """Search the local Ayurvedic documents for home remedies, herbs
    (like tulsi, ashwagandha, turmeric), and dosha-related advice.
    Use this FIRST for any Ayurveda-related question.
    Input should be a short search phrase, e.g. 'remedy for cough'.
    """
    results = retriever.invoke(query)
    if not results:
        return "No matching Ayurvedic document found."
    return "\n\n---\n\n".join(doc.page_content for doc in results)


# ---------------------------------------------------------------------
# 2. Set up the DuckDuckGo search tool (for general web info + product links)
# ---------------------------------------------------------------------
# DuckDuckGoSearchRun is already a ready-made LangChain tool - no need to
# wrap it in a custom @tool function.
search_tool = DuckDuckGoSearchRun(
    name="web_search",
    description=(
        "Search the web using DuckDuckGo. Useful for general health/"
        "lifestyle questions not covered by the Ayurvedic knowledge base, "
        "current information, and finding real product links when the "
        "user wants to buy something (e.g. 'buy ashwagandha powder online')."
    ),
)

tools = [ayurvedic_knowledge_base, search_tool]

# ---------------------------------------------------------------------
# 3. Set up the LLM and the agent
# ---------------------------------------------------------------------
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)

SYSTEM_PROMPT = """You are an Ayurvedic home-remedy and healthy-living assistant.

Guidelines:
- For questions about Ayurvedic herbs, remedies, or dosha balance, ALWAYS
  check the ayurvedic_knowledge_base tool first.
- For general health/lifestyle questions (sleep, diet, exercise, stress),
  you can answer directly from your own knowledge in a simple, practical way.
- If the user asks where to buy a product (herb, oil, powder, etc.), use
  web_search to find real, current links and share 2-3 of them.
- If it's a topic that could be serious (persistent symptoms, high fever,
  chest pain, pregnancy complications, etc.), advise the user to consult a
  qualified doctor in addition to any home remedy suggestion.
- Keep answers short, practical, and step-by-step where possible.
- You are not a licensed doctor. Make this clear when giving medical-adjacent advice.
"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


def ask_agent(question: str) -> str:
    """Simple helper: send a question to the agent, get back the answer text."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


# Quick manual test: `python agent.py`
if __name__ == "__main__":
    while True:
        q = input("\nAsk something (or 'exit'): ")
        if q.lower() == "exit":
            break
        print("\n>>>", ask_agent(q))
