from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# -------------- STEP 1 : Read Document -------------

file_path = Path(__file__).parent/"nodejs.pdf"


loader = PyPDFLoader(file_path= file_path)
docs = loader.load() # Read PDF file

# print("\n---- Docs[5] : ----\n ", docs[5])



# -------------- STEP 2 : Chunking -------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 400
)

split_docs = text_splitter.split_documents(documents=docs)


# -------------- STEP 3 : Vector Embedding -------------

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# Using [embedding_model] create embeddings of [split_docs] and store in vector DB

vector_store = QdrantVectorStore.from_documents(
        documents=split_docs,
        url = "http://localhost:6333",
        collection_name = "learning_vectors",
        embedding=embedding_model
)

print("Indexing of Documents Done........")


