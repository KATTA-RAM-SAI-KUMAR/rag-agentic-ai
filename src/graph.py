import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, START, END

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "agentic-ai-local"

# Minimum similarity required to answer
CONFIDENCE_THRESHOLD = 0.60


# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

embeddings = OllamaEmbeddings(
    model="all-minilm"
)


# --------------------------------------------------
# PINECONE VECTOR STORE
# --------------------------------------------------

vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY,
)


# --------------------------------------------------
# LOCAL LLM
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# --------------------------------------------------
# LANGGRAPH STATE
# --------------------------------------------------

class AgentState(TypedDict):
    question: str
    context: list
    answer: str
    confidence: float


# --------------------------------------------------
# RETRIEVE
# --------------------------------------------------

def retrieve(state: AgentState):

    question = state["question"]

    print("\nRetrieving relevant chunks...")

    results = vector_store.similarity_search_with_score(
        question,
        k=4,
    )

    documents = [doc for doc, score in results]
    scores = [score for doc, score in results]

    print(f"Retrieved {len(documents)} chunks.")

    print("\nSimilarity scores:")

    for i, score in enumerate(scores, 1):
        print(f"  Chunk {i}: {score:.4f}")

    if scores:

        confidence = max(scores)

    else:

        confidence = 0.0

    print(
        f"\nBest similarity score: {confidence:.4f}"
    )

    # --------------------------------------------------
    # STRICT GROUNDING CHECK
    # --------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        print(
            "\nConfidence below threshold."
        )

        print(
            "Question is outside the eBook context."
        )

        return {
            "context": [],
            "confidence": confidence,
        }

    print(
        "\nConfidence above threshold."
    )

    print(
        "Question is sufficiently related to the eBook."
    )

    return {
        "context": documents,
        "confidence": confidence,
    }


# --------------------------------------------------
# GENERATE
# --------------------------------------------------

def generate(state: AgentState):

    question = state["question"]
    documents = state["context"]

    confidence = state["confidence"]

    # --------------------------------------------------
    # OUT-OF-CONTEXT RESPONSE
    # --------------------------------------------------

    if not documents or confidence < CONFIDENCE_THRESHOLD:

        return {
            "answer": (
                "I cannot find this information "
                "in the provided eBook."
            )
        }

    # --------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------

    context_text = "\n\n".join(
        document.page_content
        for document in documents
    )

    # --------------------------------------------------
    # STRICT RAG PROMPT
    # --------------------------------------------------

    prompt = f"""
You are a question-answering assistant for an Agentic AI eBook.

STRICT RULES:

1. Answer ONLY using the provided context.
2. Do not use outside knowledge.
3. Do not invent information.
4. If the answer cannot be found in the context, say:
   "I cannot find this information in the provided eBook."
5. Keep the answer clear and concise.
6. You may synthesize information from multiple retrieved context passages when answering the question.
7. Do not add facts, assumptions, or outside knowledge that are not supported by the retrieved context.
8. If the retrieved context does not contain enough information to answer the question, say:
   "I cannot find this information in the provided eBook."

CONTEXT FROM THE EBOOK:

{context_text}

QUESTION:

{question}

ANSWER:
"""

    print("\nGenerating answer...")

    response = llm.invoke(prompt)

    answer = response.content

    return {
        "answer": answer,
    }


# --------------------------------------------------
# BUILD LANGGRAPH
# --------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node(
    "retrieve",
    retrieve
)

workflow.add_node(
    "generate",
    generate
)

workflow.add_edge(
    START,
    "retrieve"
)

workflow.add_edge(
    "retrieve",
    "generate"
)

workflow.add_edge(
    "generate",
    END
)

graph = workflow.compile()


# --------------------------------------------------
# ASK QUESTION
# --------------------------------------------------

def ask_question(question: str):

    result = graph.invoke(
        {
            "question": question,
            "context": [],
            "answer": "",
            "confidence": 0.0,
        }
    )

    return result


# --------------------------------------------------
# TERMINAL TEST
# --------------------------------------------------

if __name__ == "__main__":

    question = input(
        "\nAsk a question about the Agentic AI eBook: "
    )

    result = ask_question(question)

    print("\n==============================")
    print("ANSWER")
    print("==============================")

    print(result["answer"])

    print("\n==============================")
    print("CONFIDENCE")
    print("==============================")

    print(
        f"{result['confidence']:.4f}"
    )

    print("\n==============================")
    print("RETRIEVED CHUNKS")
    print("==============================")

    if result["context"]:

        for i, document in enumerate(
            result["context"],
            1
        ):

            print(
                f"\n--- Chunk {i} ---"
            )

            print(
                document.page_content[:500]
            )

    else:

        print(
            "No chunks returned because "
            "the confidence was below the threshold."
        )