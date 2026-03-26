from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline

# Load data
loader = TextLoader("data/fd_policy.txt")
documents = loader.load()

# Split text (smaller chunks)
text_splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
docs = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
vectorstore = FAISS.from_documents(docs, embeddings)

# 🔥 Limit retrieved docs
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Local LLM
pipe = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=100   # 🔥 limit output
)

llm = HuggingFacePipeline(pipeline=pipe)

# 🔥 Custom prompt (MAIN FIX)
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
query = input("Ask your FD question: ")

response = qa.invoke({"query": query})

print("\nAnswer:")
print(response["result"])