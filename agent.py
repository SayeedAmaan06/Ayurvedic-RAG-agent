"""
agent.py
--------
A minimal LangChain agent that:
1. Answers general health/life-improvement chat directly.
2. Uses a RAG tool to look up Ayurvedic home remedies from local documents.
3. Uses a DuckDuckGo search tool for anything it doesn't know / for finding
   product links to buy.

This file only defines the agent. FastAPI wraps it in app.py.
"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ddgs import DDGS
from langchain.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

DB_DIR = "chroma_db"

# ---------------------------------------------------------------------
# 1. Set up the RAG retriever (reads the vector store built by build_rag.py)
# ---------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})


def rag_lookup(query: str) -> str:
    """Search the local Ayurvedic documents for relevant remedies."""
    results = retriever.invoke(query)
    if not results:
        return "No matching Ayurvedic document found."
    return "\n\n---\n\n".join(doc.page_content for doc in results)


rag_tool = Tool(
    name="ayurvedic_knowledge_base",
    func=rag_lookup,
    description=(
        "Use this tool FIRST for any question about Ayurvedic remedies, "
        "herbs (like tulsi, ashwagandha, turmeric), home remedies, or "
        "dosha-related advice. Input should be a short search phrase, "
        "e.g. 'remedy for cough' or 'ashwagandha benefits'."
    ),
)

# ---------------------------------------------------------------------
# 2. Set up the DuckDuckGo search tool (for general web info + product links)
# ---------------------------------------------------------------------
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return top results with links."""
    ddgs = DDGS()
    results = list(ddgs.text(query, max_results=5))
    if not results:
        return "No search results found."
    return "\n\n".join(
        f"{r['title']}\n{r['href']}\n{r['body']}" for r in results
    )


search_tool = Tool(
    name="web_search",
    func=web_search,
    description=(
        "Use this tool to search the internet. Useful for: general health "
        "or lifestyle questions not covered by the Ayurvedic knowledge "
        "base, current information, and finding real product links when "
        "the user wants to buy something (e.g. 'buy ashwagandha powder "
        "online')."
    ),
)

tools = [rag_tool, search_tool]

# ---------------------------------------------------------------------
# 3. Set up the LLM and the agent prompt
# ---------------------------------------------------------------------
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# ---------------------------------------------------------------------
# 4. Build the agent + executor
# ---------------------------------------------------------------------
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def ask_agent(question: str) -> str:
    """Simple helper: send a question to the agent, get back the answer text."""
    result = agent_executor.invoke({"input": question})
    return result["output"]


# Quick manual test: `python agent.py`
if __name__ == "__main__":
    while True:
        q = input("\nAsk something (or 'exit'): ")
        if q.lower() == "exit":
            break
        print("\n>>>", ask_agent(q))
