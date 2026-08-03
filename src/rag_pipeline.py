from pathlib import Path
import sys

from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from ragas_integration import build_evaluation_dataset, run_ragas_evaluation

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_documents():
    text_path = DATA_DIR / "fd_policy.txt"
    if text_path.exists():
        return TextLoader(str(text_path)).load()

    pdf_path = DATA_DIR / "HDFCT2_.pdf"
    if pdf_path.exists():
        return PyPDFLoader(str(pdf_path)).load()

    raise FileNotFoundError("No policy document found in the data folder.")


def build_rag_components():
    documents = load_documents()

    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt_template = """Use the context below to answer the question.
Give a short, clear answer. Do NOT repeat the full context.

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )

    return retriever, qa


def ask_loop():
    retriever, qa = build_rag_components()
    query = input("Ask Questions related to MF -  ")

    response = qa.invoke({"query": query})

    print("\nAnswer:")
    print(response["result"])


def run_evaluation():
    retriever, qa = build_rag_components()
    samples = build_evaluation_dataset()
    results = run_ragas_evaluation(retriever=retriever, qa_chain=qa, samples=samples)

    print("\nRAGAS evaluation results:")
    if results.get("status") == "ok":
        print(results["results"])
    else:
        print(results["message"])


if __name__ == "__main__":
    if "--evaluate" in sys.argv:
        run_evaluation()
    else:
        ask_loop()
