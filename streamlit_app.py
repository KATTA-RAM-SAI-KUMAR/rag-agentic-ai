import streamlit as st
from src.graph import ask_question, CONFIDENCE_THRESHOLD


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Agentic AI RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# SIMPLE STYLING
# =========================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background: #080b14;
    }

    /* Remove top Streamlit whitespace/header */
    header[data-testid="stHeader"] {
        background: #080b14 !important;
        border: none !important;
        height: 0rem !important;
    }

    [data-testid="stToolbar"] {
        display: none !important;
    }

    /* Main content */
    .block-container {
        max-width: 1200px !important;
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0c101b;
    }

    /* Headings */
    h1 {
        color: #ffffff !important;
        font-size: 42px !important;
        font-weight: 800 !important;
        margin-top: 0 !important;
        margin-bottom: 10px !important;
    }

    h2 {
        color: #ffffff !important;
    }

    h3 {
        color: #ffffff !important;
    }

    /* Text */
    p {
        color: #aab3c5;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 12px;
        border: none;
        background: linear-gradient(
            90deg,
            #6366f1,
            #8b5cf6
        );
        color: white;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #716cf5,
            #9b6af7
        );
        color: white;
    }

    /* Text input */
    .stTextInput input {
        background: #111827 !important;
        color: white !important;
        border-radius: 12px !important;
        border: 1px solid #30384d !important;
    }

    .stTextInput input:focus {
        border: 1px solid #7c3aed !important;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: #111827;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #252d40;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: #111827;
        border: 1px solid #252d40;
        border-radius: 12px;
    }

    /* Horizontal line */
    hr {
        border-color: #202638 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🤖 Agentic AI")

    st.caption("Knowledge-grounded RAG Assistant")

    st.divider()

    st.success("Local AI system online")

    st.subheader("📚 Knowledge Base")
    st.write("Agentic AI eBook")
    st.caption("60 pages • 119 chunks")

    st.subheader("🔎 Retrieval")
    st.write("Pinecone")
    st.caption("Top 4 relevant chunks")

    st.subheader("🧠 Embeddings")
    st.write("all-MiniLM")
    st.caption("384-dimensional vectors")

    st.subheader("⚡ Generation")
    st.write("Llama 3.2 • 3B")
    st.caption("Running locally with Ollama")

    st.subheader("🛡️ Grounding")
    st.write("Strict eBook-only answers")
    st.caption("Confidence threshold: 0.60")


# =========================================================
# HEADER
# =========================================================

st.caption("KNOWLEDGE-GROUNDED AI")

st.title("Agentic AI RAG Assistant")

st.write(
    "Ask questions about the Agentic AI eBook. "
    "Answers are generated only from the provided knowledge base."
)

st.divider()


# =========================================================
# EXAMPLE QUESTIONS
# =========================================================

st.subheader("✨ Try an example")

examples = [
    "What is Agentic AI?",
    "What are the key components of an agentic AI system?",
    "How are agents different from traditional automation?",
    "What role does memory play in an AI agent?",
]

columns = st.columns(4)

for i, example in enumerate(examples):

    with columns[i]:

        if st.button(
            example,
            key=f"example_{i}",
        ):
            st.session_state["question"] = example


# =========================================================
# QUESTION INPUT
# =========================================================

st.subheader("💬 Ask your question")

if "question" not in st.session_state:
    st.session_state["question"] = ""

question = st.text_input(
    "Question",
    value=st.session_state["question"],
    placeholder="e.g. What is Agentic AI?",
    label_visibility="collapsed",
)

st.session_state["question"] = question


# =========================================================
# ASK BUTTON
# =========================================================

if st.button(
    "🚀 Ask the Agentic AI Assistant",
    type="primary",
):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "🔎 Searching the knowledge base..."
        ):

            result = ask_question(
                question.strip()
            )

        st.session_state["result"] = result


# =========================================================
# RESULT
# =========================================================

if "result" in st.session_state:

    result = st.session_state["result"]

    answer = result["answer"]
    confidence = result["confidence"]
    documents = result["context"]


    # -----------------------------------------------------
    # ANSWER
    # -----------------------------------------------------

    st.divider()

    st.subheader("✨ Answer")

    st.info(answer)


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Retrieval Confidence",
            f"{confidence:.1%}",
        )

    with col2:

        st.metric(
            "Retrieved Chunks",
            len(documents),
        )

    with col3:

        if (
            documents
            and confidence >= CONFIDENCE_THRESHOLD
        ):
            status = "GROUNDED"
        else:
            status = "OUTSIDE KB"

        st.metric(
            "Knowledge Status",
            status,
        )


    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    st.subheader("📊 Confidence")

    confidence_value = min(
        max(float(confidence), 0.0),
        1.0,
    )

    st.progress(confidence_value)

    if confidence >= CONFIDENCE_THRESHOLD:

        st.success(
            f"Strong grounding — "
            f"{confidence:.1%} confidence"
        )

    else:

        st.warning(
            f"Low confidence — "
            f"{confidence:.1%}. "
            f"The question may be outside the eBook."
        )


    # -----------------------------------------------------
    # RETRIEVED CONTEXT
    # -----------------------------------------------------

    if documents:

        st.subheader("📚 Retrieved Context")

        st.caption(
            "These chunks were retrieved from Pinecone "
            "and provided to the language model."
        )

        for i, document in enumerate(
            documents,
            start=1,
        ):

            page = document.metadata.get(
                "page",
                "Unknown",
            )

            with st.expander(
                f"📄 Chunk {i} • Page {page}"
            ):

                st.write(
                    document.page_content
                )


    # -----------------------------------------------------
    # OUTSIDE KNOWLEDGE BASE
    # -----------------------------------------------------

    else:

        st.warning(
            "🛡️ Grounding protection activated.\n\n"
            "I cannot find this information in the "
            "provided eBook."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Agentic AI RAG Assistant • "
    "LangGraph • Pinecone • Ollama • Streamlit"
)