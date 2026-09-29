import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# New index for Ollama's 384-dimensional embeddings
INDEX_NAME = "agentic-ai-local"

PDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "Ebook-Agentic-AI.pdf",
)


def run_ingestion():

    print("Loading PDF...")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    print("Splitting document into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("Connecting to Pinecone...")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [index.name for index in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:

        print(f"Creating Pinecone index: {INDEX_NAME}")

        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        print("Index created.")

    else:

        print(f"Using existing index: {INDEX_NAME}")

    print("Loading local embedding model...")

    embeddings = OllamaEmbeddings(
        model="all-minilm"
    )

    print("Uploading chunks to Pinecone...")

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )

    print()
    print("===================================")
    print("INGESTION COMPLETED SUCCESSFULLY!")
    print("===================================")
    print(f"Total chunks indexed: {len(chunks)}")
    print(f"Pinecone index: {INDEX_NAME}")


if __name__ == "__main__":
    run_ingestion()