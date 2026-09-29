from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import ask_question


app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="RAG chatbot over the Agentic AI eBook",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: list[str]
    confidence: float


@app.get("/")
def root():
    return {
        "message": "Agentic AI RAG Chatbot API is running"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    result = ask_question(request.question)

    retrieved_chunks = [
        document.page_content
        for document in result["context"]
    ]

    return {
        "answer": result["answer"],
        "retrieved_chunks": retrieved_chunks,
        "confidence": result["confidence"],
    }