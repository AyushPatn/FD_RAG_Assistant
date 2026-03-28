import os
from dotenv import load_dotenv

# LangChain Core Imports (new standard)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --------------------------------------------------
# 1. Load Environment
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# 2. Load Documents
# --------------------------------------------------
def load_documents():
    loader = TextLoader("data/fd_policy.txt")
    return loader.load()

# --------------------------------------------------
# 3. Split Documents
# --------------------------------------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

# --------------------------------------------------
# 4. Create Vector Store
# --------------------------------------------------
def create_vectorstore(docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(docs, embeddings)

# --------------------------------------------------
# 5. Create Retriever
# --------------------------------------------------
def create_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 6}
    )

# --------------------------------------------------
# 6. Create LLM
# --------------------------------------------------
def create_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

# --------------------------------------------------
# 7. Prompt Template
# --------------------------------------------------
def get_prompt():
    template = """
You are a banking assistant.

Answer ONLY from the given context.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{input}

Answer:
"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "input"]
    )

# --------------------------------------------------
# 8. Build RAG Chain (LCEL STANDARD)
# --------------------------------------------------
def build_rag_chain(retriever, llm, prompt):
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

# --------------------------------------------------
# 9. Main Execution
# --------------------------------------------------
def main():
    documents = load_documents()
    docs = split_documents(documents)

    vectorstore = create_vectorstore(docs)
    retriever = create_retriever(vectorstore)

    llm = create_llm()
    prompt = get_prompt()

    rag_chain = build_rag_chain(retriever, llm, prompt)

    while True:
        query = input("\nAsk your FD question (type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        response = rag_chain.invoke({"input": query})

        print("\nAnswer:")
        print(response["answer"])

# --------------------------------------------------
# Entry Point
# --------------------------------------------------
if __name__ == "__main__":
    main()