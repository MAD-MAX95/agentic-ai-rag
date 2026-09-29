from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from pydantic import BaseModel, Field

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    PINECONE_INDEX_NAME,
    TOP_K,
)

from src.gemini_embeddings import GeminiEmbeddings


# ============================================================
# 1. STATE
# ============================================================

class AgentState(TypedDict):
    question: str
    context: list[str]
    answer: str
    score: float


# ============================================================
# 2. STRUCTURED OUTPUT
# ============================================================

class GenerationResult(BaseModel):
    answer: str = Field(
        description="Answer based only on the provided context."
    )

    confidence_score: float = Field(
        description="Confidence that the answer is directly supported "
                    "by the provided context. Must be between 0 and 1."
    )


# ============================================================
# 3. PINECONE RETRIEVER
# ============================================================

embeddings = GeminiEmbeddings()

vector_store = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
)


# ============================================================
# 4. GEMINI CLIENT
# ============================================================

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


# ============================================================
# 5. RETRIEVE NODE
# ============================================================

def retrieve(state: AgentState):

    question = state["question"]

    print(f"\nRetrieving context for: {question}")

    results = vector_store.similarity_search_with_score(
        question,
        k=TOP_K,
    )

    context = []

    RELEVANCE_THRESHOLD = 0.60

    for doc, score in results:

        print(
            f"Page: {doc.metadata.get('page', 'unknown')} | "
            f"Chunk: {doc.metadata.get('chunk_id', 'unknown')} | "
            f"Score: {score:.4f}"
        )

        if score < RELEVANCE_THRESHOLD:
            continue

        page = doc.metadata.get("page", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")

        formatted_chunk = (
            f"[Page: {page} | Chunk: {chunk_id} | "
            f"Relevance: {score:.4f}]\n"
            f"{doc.page_content}"
        )

        context.append(formatted_chunk)

    print(f"Retrieved {len(context)} relevant chunks.")

    return {
        "context": context
    }


# ============================================================
# 6. GENERATE NODE
# ============================================================

def generate(state: AgentState):

    question = state["question"]
    context = state["context"]

    if not context:

        return {
            "answer": "I don't have enough information in the provided eBook to answer that.",
            "score": 0.0,
        }

    context_text = "\n\n".join(context)

    system_prompt = """
You are a RAG assistant answering questions about an Agentic AI eBook.

IMPORTANT RULES:

1. Answer ONLY using the provided context.
2. Do NOT use your outside knowledge.
3. If the context does not contain enough information to answer
   the question, say:

   "I don't have enough information in the provided eBook to answer that."

4. Never invent facts.
5. Never use information that is not present in the retrieved context.
6. The confidence score must represent how strongly the retrieved
   context supports the answer.
7. Confidence must be between 0 and 1.
"""

    user_prompt = f"""
CONTEXT FROM THE EBOOK:

{context_text}

QUESTION:

{question}

Return:
- answer
- confidence_score
"""

    response = client.beta.chat.completions.parse(
        model=GEMINI_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        response_format=GenerationResult,
    )

    result = response.choices[0].message.parsed

    if result is None:
        raise ValueError('Gemini did not return a structred response')

    return {
        "answer": result.answer,
        "score": max(0.0, min(1.0, result.confidence_score)),
    }


# ============================================================
# 7. BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()