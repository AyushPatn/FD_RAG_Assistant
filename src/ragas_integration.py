from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddings
    from ragas.llms import LangchainLLM
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
except ImportError:  # pragma: no cover - handled gracefully for local/test environments
    Dataset = None
    evaluate = None
    LangchainEmbeddings = None
    LangchainLLM = None
    answer_relevancy = context_precision = context_recall = faithfulness = None

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def build_evaluation_dataset() -> List[Dict[str, str]]:
    """Create a small, review-friendly evaluation dataset for policy questions."""
    return [
        {
            "question": "What is the minimum age to open a Fixed Deposit?",
            "ground_truth": "Indian residents above 18 years can open an FD. Minors can open with a guardian.",
        },
        {
            "question": "What is the premature withdrawal penalty?",
            "ground_truth": "The premature withdrawal penalty is 1%.",
        },
        {
            "question": "When does TDS apply on FD interest?",
            "ground_truth": "TDS applies if interest exceeds 40,000.",
        },
    ]


def run_ragas_evaluation(
    retriever: Any,
    qa_chain: Any,
    samples: Optional[List[Dict[str, str]]] = None,
    llm: Optional[Any] = None,
    embeddings: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run a lightweight RAGAS evaluation over the provided sample questions."""
    if samples is None:
        samples = build_evaluation_dataset()

    if evaluate is None or Dataset is None or LangchainLLM is None or LangchainEmbeddings is None:
        return {
            "status": "skipped",
            "message": "RAGAS dependencies are not installed. Install ragas and datasets to enable evaluation.",
            "samples": len(samples),
        }

    rows: List[Dict[str, Any]] = []
    for sample in samples:
        question = sample["question"]
        docs = retriever.get_relevant_documents(question)
        answer = qa_chain.invoke({"query": question})["result"]

        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": [doc.page_content for doc in docs],
                "ground_truth": sample.get("ground_truth", ""),
            }
        )

    dataset = Dataset.from_list(rows)

    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if embeddings is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    llm_wrapper = LangchainLLM(llm=llm)
    embedding_wrapper = LangchainEmbeddings(embeddings=embeddings)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm_wrapper,
        embeddings=embedding_wrapper,
    )

    return {
        "status": "ok",
        "results": result,
        "samples": len(rows),
    }
