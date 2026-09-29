import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-ai-index"
)

PDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "Ebook-Agentic-AI.pdf"
)

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from .env")