from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.gemini_embeddings import GeminiEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

from src.config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    GEMINI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSION,
)


def load_pdf():
    pdf_path = Path(PDF_PATH)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    print(f"Pages loaded: {len(documents)}")

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    print(f"Total chunks created: {len(chunks)}")

    return chunks

def filter_noise_chunks(chunks):
    noise_phrases = [
        "scan this qr code",
        "scan the qr",
        "e-copy of this book",
    ]

    filtered_chunks = []

    for chunk in chunks:
        text = chunk.page_content.lower()

        if any(phrase in text for phrase in noise_phrases):
            continue

        filtered_chunks.append(chunk)

    print(
        f"Removed {len(chunks) - len(filtered_chunks)} "
        f"noise chunks."
    )

    return filtered_chunks

def get_embeddings():
    return GeminiEmbeddings()

def create_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:

        print("Creating Pinecone index...")

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        print("Pinecone index created.")

    else:
        print("Pinecone index already exists.")

    return pc


def upload_to_pinecone(chunks):

    embeddings = get_embeddings()

    create_pinecone_index()

    print("Uploading chunks to Pinecone...")

    vector_store = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )

    vector_store.add_documents(chunks)

    print(f"Uploaded {len(chunks)} chunks to Pinecone.")

    return vector_store


def ingest():

    print("\n========== PDF INGESTION ==========\n")

    documents = load_pdf()

    print("\n========== CHUNKING ==========\n")

    chunks = split_documents(documents)

    print("\n========== FILTERING NOISE ==========\n")

    chunks = filter_noise_chunks(chunks)

    print("\n========== PINECONE ==========\n")

    upload_to_pinecone(chunks)

    print("\n========== INGESTION COMPLETE ==========\n")


if __name__ == "__main__":
    ingest()