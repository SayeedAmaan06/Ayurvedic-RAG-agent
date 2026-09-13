"""
app.py
------
Minimal FastAPI wrapper around the LangChain agent.

Run locally with:
    uvicorn app:app --reload

Endpoints:
    GET  /            -> health check
    POST /chat        -> {"question": "..."} -> {"answer": "..."}
"""

from fastapi import FastAPI
from pydantic import BaseModel
from agent import ask_agent

app = FastAPI(title="Ayurvedic Home Remedy Agent")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Ayurvedic agent is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer = ask_agent(request.question)
    return ChatResponse(answer=answer)
