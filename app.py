from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

# OpenAI imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# Load data
loader = PyPDFLoader("data/HDFCT2_.pdf")
documents = loader.load()

# Split text
text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
docs = text_splitter.split_documents(documents)

# Use OpenAI Embeddings (UPDATED)
embeddings = OpenAIEmbeddings()

# Vector DB
vectorstore = FAISS.from_documents(docs, embeddings)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# GPT-3.5 LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Custom prompt
prompt_template = """Use the context below to answer the question.
Give a short, clear answer. Do NOT repeat the full context.

Context:
{context}

Question:
{question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)

# Query
query = input("Ask Questions related to MF -  ")

response = qa.invoke({"query": query})

print("\nAnswer:")
print(response["result"])