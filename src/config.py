import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME=os.getenv("PINECONE_INDEX_NAME","agentic-ai-rag")


PDF_PATH='data/Ebook-Agentic-AI.pdf'

CHUNK_SIZE=1000
CHUNK_OVERLAP=200

TOP_K=4

EMBEDDING_DIMENSION=1536

GEMINI_MODEL = "gemini-3.5-flash-lite"