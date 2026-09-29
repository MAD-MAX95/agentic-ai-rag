from typing import List

from openai import OpenAI
from langchain_core.embeddings import Embeddings

from src.config import GEMINI_API_KEY, EMBEDDING_DIMENSION


class GeminiEmbeddings(Embeddings):

    def __init__(self):
        self.client = OpenAI(
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def embed_query(self, text: str) -> List[float]:

        response = self.client.embeddings.create(
            model="gemini-embedding-001",
            input=text,
            dimensions=EMBEDDING_DIMENSION,
        )

        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:

        embeddings = []

        for i, text in enumerate(texts):

            print(f"Embedding chunk {i + 1}/{len(texts)}")

            response = self.client.embeddings.create(
                model="gemini-embedding-001",
                input=text,
                dimensions=EMBEDDING_DIMENSION,
            )

            embeddings.append(response.data[0].embedding)

        return embeddings