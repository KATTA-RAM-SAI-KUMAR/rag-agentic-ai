from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import ask_question

app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="RAG chatbot over the Agentic AI eBook",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float


@app.get("/")
def root():
    return {
        "message": "Agentic AI RAG Chatbot API is running"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = ask_question(request.query)

    retrieved_context_chunks = [
        document.page_content
        for document in result["context"]
    ]

    return {
        "query": request.query,
        "final_answer": result["answer"],
        "retrieved_context_chunks": retrieved_context_chunks,
        "confidence_score": result["confidence"],
    }