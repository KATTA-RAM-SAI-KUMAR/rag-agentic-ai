import os

from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "agentic-ai-cloud"

PDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "Ebook-Agentic-AI.pdf",
)


class FastEmbedEmbeddings:
    def __init__(self):
        print("Loading FastEmbed model...")
        self.model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

    def embed_documents(self, texts):
        return [
            embedding.tolist()
            for embedding in self.model.embed(texts)
        ]

    def embed_query(self, text):
        return next(
            self.model.embed([text])
        ).tolist()


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

    embeddings = FastEmbedEmbeddings()

    print("Uploading chunks to Pinecone...")
    print(f"Index: {INDEX_NAME}")

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY,
    )

    print()
    print("===================================")
    print("CLOUD INGESTION COMPLETED!")
    print("===================================")
    print(f"Total chunks indexed: {len(chunks)}")
    print(f"Pinecone index: {INDEX_NAME}")


if __name__ == "__main__":
    run_ingestion()