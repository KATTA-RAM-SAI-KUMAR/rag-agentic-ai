from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src import graph


# =========================================================
# TEST 1 — Relevant question should retrieve documents
# =========================================================

def test_retrieve_relevant_question():

    documents = [
        Document(page_content="Agentic AI enables autonomous decision making."),
        Document(page_content="Agents can perceive, reason and act."),
    ]

    mock_vector_store = MagicMock()

    mock_vector_store.similarity_search_with_score.return_value = [
        (documents[0], 0.82),
        (documents[1], 0.78),
    ]

    with patch.object(
        graph,
        "vector_store",
        mock_vector_store
    ):

        result = graph.retrieve({
            "question": "What is Agentic AI?",
            "context": [],
            "answer": "",
            "confidence": 0.0,
        })

    assert len(result["context"]) == 2
    assert result["confidence"] == 0.82


# =========================================================
# TEST 2 — Irrelevant question should be rejected
# =========================================================

def test_retrieve_irrelevant_question():

    mock_vector_store = MagicMock()

    mock_vector_store.similarity_search_with_score.return_value = [
        (
            Document(page_content="Some unrelated content"),
            0.10
        ),
        (
            Document(page_content="Another unrelated chunk"),
            0.08
        ),
    ]

    with patch.object(
        graph,
        "vector_store",
        mock_vector_store
    ):

        result = graph.retrieve({
            "question": "Who won the FIFA World Cup?",
            "context": [],
            "answer": "",
            "confidence": 0.0,
        })

    assert result["context"] == []
    assert result["confidence"] == 0.10


# =========================================================
# TEST 3 — Generate answer when context exists
# =========================================================

def test_generate_answer_with_context():

    mock_llm = MagicMock()

    mock_response = MagicMock()

    mock_response.content = (
        "Agentic AI enables autonomous decision-making."
    )

    mock_llm.invoke.return_value = mock_response

    documents = [
        Document(
            page_content=(
                "Agentic AI enables autonomous "
                "decision-making and action."
            )
        )
    ]

    with patch.object(
        graph,
        "llm",
        mock_llm
    ):

        result = graph.generate({
            "question": "What is Agentic AI?",
            "context": documents,
            "answer": "",
            "confidence": 0.82,
        })

    assert (
        result["answer"]
        == "Agentic AI enables autonomous decision-making."
    )

    mock_llm.invoke.assert_called_once()


# =========================================================
# TEST 4 — Generate should refuse without context
# =========================================================

def test_generate_without_context():

    result = graph.generate({
        "question": "Who won the FIFA World Cup?",
        "context": [],
        "answer": "",
        "confidence": 0.10,
    })

    assert (
        result["answer"]
        == "I cannot find this information in the provided eBook."
    )


# =========================================================
# TEST 5 — Low confidence should not generate an answer
# =========================================================

def test_generate_low_confidence():

    documents = [
        Document(
            page_content="Unrelated information."
        )
    ]

    result = graph.generate({
        "question": "Who won the FIFA World Cup?",
        "context": documents,
        "answer": "",
        "confidence": 0.20,
    })

    assert (
        result["answer"]
        == "I cannot find this information in the provided eBook."
    )


# =========================================================
# TEST 6 — Full graph should return expected result
# =========================================================

def test_ask_question():

    expected_result = {
        "question": "What is Agentic AI?",
        "context": [
            Document(
                page_content="Agentic AI enables autonomous action."
            )
        ],
        "answer": "Agentic AI enables autonomous action.",
        "confidence": 0.82,
    }

    with patch.object(
        graph,
        "graph"
    ) as mock_graph:

        mock_graph.invoke.return_value = expected_result

        result = graph.ask_question(
            "What is Agentic AI?"
        )

    assert result["answer"] == (
        "Agentic AI enables autonomous action."
    )

    assert result["confidence"] == 0.82

    assert len(result["context"]) == 1

    mock_graph.invoke.assert_called_once()