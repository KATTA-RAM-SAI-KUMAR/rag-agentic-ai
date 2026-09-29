# 🤖 Agentic AI RAG Chatbot

A knowledge-grounded **Retrieval-Augmented Generation (RAG)** chatbot built over an Agentic AI eBook.

The system retrieves relevant content from the eBook using **Pinecone**, orchestrates the workflow with **LangGraph**, and generates answers with a locally running **Llama 3.2** model through **Ollama**. It exposes both a **FastAPI** backend and a **Streamlit** UI.

The chatbot answers **strictly from the provided eBook** and rejects questions that fall outside the knowledge base.

\---

## 📑 Table of Contents

* [Project Objective](#-project-objective)
* [Features](#-features)
* [Architecture](#-architecture)
* [Technologies Used](#-technologies-used)
* [Project Structure](#-project-structure)
* [System Requirements](#-system-requirements)
* [Installation](#-installation)
* [Configuration](#-configuration)
* [Data Ingestion](#-data-ingestion)
* [LangGraph RAG Workflow](#-langgraph-rag-workflow)
* [Grounding Protection](#-grounding-protection)
* [Running the Application](#-running-the-application)
* [API Reference](#-api-reference)
* [Streamlit Application](#-streamlit-application)
* [Testing](#-testing)
* [Security](#-security)
* [Project Configuration Summary](#-project-configuration-summary)
* [Why Local Ollama?](#-why-local-ollama)
* [Quick Start](#-quick-start)
* [Future Improvements](#-future-improvements)

\---

## 📌 Project Objective

Build a RAG-based chatbot that:

1. Loads and processes an Agentic AI eBook.
2. Splits the eBook into smaller chunks.
3. Generates vector embeddings for the chunks.
4. Stores the embeddings in Pinecone.
5. Retrieves the most relevant chunks for a user question.
6. Uses LangGraph to orchestrate retrieval and generation.
7. Generates answers using **only** the retrieved eBook context.
8. Prevents unsupported / out-of-context answers.
9. Returns the retrieved chunks and a confidence score.
10. Provides a REST API using FastAPI.
11. Provides an interactive UI using Streamlit.

\---

## ✨ Features

* 📚 PDF-based knowledge base
* 🔎 Semantic search
* 🧠 Local embeddings
* 🗄️ Pinecone vector database
* 🔄 LangGraph workflow
* 🤖 Local LLM using Ollama
* 🛡️ Strict grounding
* 📊 Confidence score
* 🚫 Out-of-context question protection
* ⚡ FastAPI REST API
* 💻 Streamlit web interface
* 🧪 Automated unit tests

\---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │   Agentic AI eBook  │
                    │        (PDF)        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     PyPDFLoader     │
                    │   Load PDF Pages    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Text Splitter    │
                    │     Chunk = 1000    │
                    │     Overlap = 200   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Ollama Embeddings  │
                    │      all-MiniLM     │
                    │    384 dimensions   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       Pinecone      │
                    │   Vector Database   │
                    └──────────┬──────────┘
                               │
                               │ User Question
                               ▼
                    ┌─────────────────────┐
                    │      LangGraph      │
                    │                     │
                    │  START              │
                    │    ↓                │
                    │  Retrieve           │
                    │    ↓                │
                    │  Generate           │
                    │    ↓                │
                    │  END                │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Llama 3.2 3B    │
                    │       (Ollama)      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Answer + Confidence │
                    │  + Retrieved Chunks │
                    └─────────────────────┘
```

\---

## 🛠️ Technologies Used

|Technology|Purpose|
|-|-|
|Python|Core programming language|
|LangChain|RAG components|
|LangGraph|RAG workflow orchestration|
|Pinecone|Vector database|
|Ollama|Local AI model runtime|
|all-MiniLM|Embedding model|
|Llama 3.2 3B|Local language model|
|PyPDF|PDF processing|
|FastAPI|REST API|
|Streamlit|Web interface|
|Pytest|Testing|
|python-dotenv|Environment configuration|

\---

## 📁 Project Structure

```text
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── src/
│   ├── \_\_init\_\_.py
│   ├── config.py
│   ├── ingestion.py
│   └── graph.py
│
├── tests/
│   └── test\_rag.py
│
├── venv/
│
├── .env
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── app.py
└── streamlit\_app.py
```

|File|Description|
|-|-|
|`data/Ebook-Agentic-AI.pdf`|Source Agentic AI eBook|
|`src/config.py`|Project configuration|
|`src/ingestion.py`|PDF processing and Pinecone ingestion|
|`src/graph.py`|LangGraph RAG workflow|
|`tests/test\_rag.py`|RAG unit tests|
|`app.py`|FastAPI application|
|`streamlit\_app.py`|Streamlit frontend|
|`.env`|API keys and local configuration|
|`.env.example`|Example environment configuration|
|`requirements.txt`|Python dependencies|
|`.gitignore`|Files excluded from Git|
|`README.md`|Project documentation|

\---

## ⚙️ System Requirements

* Python 3.10+
* [Ollama](https://ollama.com)
* Pinecone account and API key
* Git
* Internet connection (for Pinecone)
* Windows / Linux / macOS

> The LLM and embedding model run \*\*locally\*\* using Ollama.

\---

## 🚀 Installation

### 1\. Clone the repository

```bash
git clone <YOUR\_GITHUB\_REPOSITORY\_URL>
cd rag-agentic-ai
```

### 2\. Create a virtual environment

**Windows**

```powershell
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4\. Install Ollama

Install Ollama from the [official website](https://ollama.com), then verify:

```bash
ollama --version
```

### 5\. Download the required models

```bash
# Embedding model
ollama pull all-minilm

# Language model
ollama pull llama3.2:3b
```

Verify the models:

```bash
ollama list
```

Expected models:

```text
all-minilm
llama3.2:3b
```

\---

## 🔑 Configuration

### Environment variables

Create a Pinecone API key, then create a `.env` file in the project root (use `.env.example` as a template):

```env
PINECONE\_API\_KEY=your\_pinecone\_api\_key
PINECONE\_INDEX\_NAME=agentic-ai-local
```

> ⚠️ Never commit `.env` to GitHub. It is excluded via `.gitignore`.

### Pinecone index

The local embedding model produces **384-dimensional** vectors, so the Pinecone index is configured as:

|Setting|Value|
|-|-|
|Index Name|`agentic-ai-local`|
|Dimension|384|
|Metric|cosine|
|Cloud|AWS|
|Region|us-east-1|

The original OpenAI-based 1536-dimensional index is **not** modified.

\---

## 📚 Data Ingestion

The project uses the Agentic AI eBook located at `data/Ebook-Agentic-AI.pdf`.

**Pipeline:**

```text
PDF
 ↓
PyPDFLoader
 ↓
Document Pages
 ↓
RecursiveCharacterTextSplitter
 ↓
1000-character chunks (200-character overlap)
 ↓
all-MiniLM embeddings
 ↓
Pinecone
```

**Run ingestion:**

```bash
python -m src.ingestion
```

**Expected output:**

```text
Loading PDF...
Loaded 60 pages.

Splitting document into chunks...
Created 119 chunks.

Connecting to Pinecone...
Using existing index: agentic-ai-local

Loading local embedding model...

Uploading chunks to Pinecone...

===================================
INGESTION COMPLETED SUCCESSFULLY!
===================================
Total chunks indexed: 119
Pinecone index: agentic-ai-local
```

\---

## 🔄 LangGraph RAG Workflow

```text
START → Retrieve → Generate → END
```

### Retrieve node

1. Receives the user question.
2. Performs semantic similarity search in Pinecone.
3. Retrieves the **top 4** relevant chunks.
4. Calculates a confidence value.
5. Checks whether the confidence exceeds the threshold (**0.60**).

### Generate node

If the retrieved information is sufficiently relevant, the generation node sends the retrieved context to Llama 3.2. The model is instructed to:

* Use only the retrieved context.
* Not use outside knowledge.
* Not invent information.
* Return a fixed response when the answer is not available.

For unsupported questions, the chatbot returns:

```text
I cannot find this information in the provided eBook.
```

\---

## 🛡️ Grounding Protection

Preventing hallucinated answers is a core requirement. The chatbot uses a confidence threshold:

```python
CONFIDENCE\_THRESHOLD = 0.60
```

If the retrieval confidence is below the threshold, the retrieved documents are **not** passed to the generation stage:

```text
Question
   ↓
Pinecone Retrieval
   ↓
Confidence Check
   ↓
Below 0.60
   ↓
Reject Question
   ↓
"I cannot find this information in the provided eBook."
```

This adds a second grounding layer for questions outside the eBook.

### Example questions

**Answerable:**

* What is Agentic AI?
* What are the key components of an agentic AI system?
* How are agents different from traditional automation?
* What role does memory play in an AI agent?

**Rejected:**

* *Who won the FIFA World Cup?* → `I cannot find this information in the provided eBook.`

### Example RAG result

For the question **"What is Agentic AI?"**, the system retrieved 4 chunks:

|Chunk|Similarity Score|
|-|-|
|Chunk 1|0.8170|
|Chunk 2|0.7813|
|Chunk 3|0.7516|
|Chunk 4|0.7377|

Confidence: **0.8170** — the answer is generated from the retrieved eBook context.

\---

## ▶️ Running the Application

### Complete request flow

```text
User
 │
 ▼
Streamlit / FastAPI
 │
 ▼
LangGraph
 │
 ▼
Retrieve from Pinecone
 │
 ▼
Top 4 Relevant Chunks
 │
 ▼
Confidence Check
 │
 ├── Low Confidence ──► Reject
 │
 └── High Confidence
          │
          ▼
     Context + Question
          │
          ▼
      Llama 3.2
          │
          ▼
     Final Answer
          │
          ▼
 Answer + Chunks + Confidence
```

### Terminal 1 — Verify Ollama models

```bash
ollama list
```

Make sure `all-minilm` and `llama3.2:3b` are listed.

### Terminal 2 — Start FastAPI

```powershell
.\\venv\\Scripts\\Activate.ps1
uvicorn app:app --reload
```

* API: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Terminal 3 — Start Streamlit

```powershell
.\\venv\\Scripts\\Activate.ps1
streamlit run streamlit\_app.py
```

* UI: [http://localhost:8501](http://localhost:8501)

\---

## 🔌 API Reference

### `GET /`

Health check to verify the API is running.

**Response:**

```json
{
  "message": "Agentic AI RAG Chatbot API is running"
}
```

### `POST /chat`

Ask a question about the eBook.

**Request:**

```json
{
  "question": "What is Agentic AI?"
}
```

**Response:**

```json
{
  "answer": "Agentic AI refers to a type of artificial intelligence that enables autonomous decision-making and action.",
  "retrieved\_chunks": \[
    "Relevant eBook content..."
  ],
  "confidence": 0.817
}
```

|Field|Description|
|-|-|
|`answer`|Generated answer (or the rejection message)|
|`retrieved\_chunks`|Chunks retrieved from Pinecone|
|`confidence`|Retrieval confidence score|

\---

## 💻 Streamlit Application

Start it with:

```bash
streamlit run streamlit\_app.py
```

Then open [http://localhost:8501](http://localhost:8501). The UI provides:

* Example questions
* Question input
* Generated answer
* Retrieval confidence
* Number of retrieved chunks
* Knowledge-base status
* Confidence progress indicator
* Retrieved eBook context

\---

## 🧪 Testing

Run the test suite:

```bash
python -m pytest -v
```

Expected result: `6 passed`

|Test|Purpose|
|-|-|
|`test\_retrieve\_relevant\_question`|Checks retrieval for relevant questions|
|`test\_retrieve\_irrelevant\_question`|Checks rejection of irrelevant retrieval|
|`test\_generate\_answer\_with\_context`|Checks answer generation with context|
|`test\_generate\_without\_context`|Checks out-of-context protection|
|`test\_generate\_low\_confidence`|Checks the confidence threshold|
|`test\_ask\_question`|Checks the complete graph invocation|

\---

## 🔐 Security

Sensitive credentials must never be committed to GitHub. Local secrets live in `.env`, and `.gitignore` includes:

```text
.env
venv/
.venv/
\_\_pycache\_\_/
.pytest\_cache/
```

Before pushing, verify that `.env` and `venv/` are **not** staged:

```bash
git status
```

\---

## 📌 Project Configuration Summary

|Setting|Value|
|-|-|
|PDF Pages|60|
|Text Chunks|119|
|Chunk Size|1000|
|Chunk Overlap|200|
|Embedding Model|all-MiniLM|
|Embedding Dimension|384|
|Vector Database|Pinecone|
|Pinecone Index|`agentic-ai-local`|
|Similarity Metric|Cosine|
|Retrieved Chunks|Top 4|
|Confidence Threshold|0.60|
|LLM|Llama 3.2 3B|
|LLM Runtime|Ollama|
|Workflow|LangGraph|
|API|FastAPI|
|UI|Streamlit|
|Testing|Pytest|

\---

## 🧠 Why Local Ollama?

The original assignment specifies OpenAI embeddings and an OpenAI LLM. This implementation uses Ollama for both embeddings (`all-minilm`) and generation (`llama3.2:3b`), so the complete RAG pipeline runs locally without requiring OpenAI API credits.

Because `all-minilm` produces 384-dimensional vectors, the project uses a separate Pinecone index configured for 384 dimensions.

\---

## 📋 Quick Start

```bash
git clone <YOUR\_GITHUB\_REPOSITORY\_URL>
cd rag-agentic-ai

python -m venv venv
.\\venv\\Scripts\\Activate.ps1        # Linux/macOS: source venv/bin/activate

python -m pip install -r requirements.txt

ollama pull all-minilm
ollama pull llama3.2:3b

python -m src.ingestion

python -m pytest -v

uvicorn app:app --reload
```

In another terminal:

```bash
streamlit run streamlit\_app.py
```

\---

## 📈 Future Improvements

* Conversation history
* Streaming LLM responses
* Source / page citations in answers
* Better retrieval and reranking
* Hybrid keyword + semantic retrieval
* Evaluation metrics for RAG quality
* More comprehensive benchmark testing
* Authentication for the API
* Docker deployment
* Cloud deployment
* Improved observability and logging

\---

## 🎯 Project Outcome

This project demonstrates a complete RAG pipeline:

```text
Document Ingestion → Chunking → Embeddings → Vector Database
→ Semantic Retrieval → LangGraph Orchestration → Grounding Check
→ Local LLM Generation → FastAPI / Streamlit
```

The resulting chatbot provides answers grounded in the Agentic AI eBook while rejecting questions that fall outside the available knowledge base.

\---

**Built with:** Python · LangChain · LangGraph · Pinecone · Ollama · Llama 3.2 · all-MiniLM · FastAPI · Streamlit · Pytest

