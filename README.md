# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built for an AI Engineer
interview assignment. The application answers questions strictly from
the provided **Agentic AI eBook** and refuses questions when the
required information is not present in the retrieved context.

## Features

-   PDF ingestion using PyPDFLoader
-   Recursive text chunking with 1000-character chunks and 200-character
    overlap
-   Noise filtering for irrelevant PDF content
-   Vector embeddings using Google's `gemini-embedding-001`
-   Pinecone vector database for similarity search
-   LangGraph workflow for retrieval and answer generation
-   Grounded answers based only on retrieved eBook context
-   Confidence score for generated answers
-   Out-of-scope question refusal
-   FastAPI backend with structured JSON responses
-   Streamlit frontend for interactive use

## Architecture

``` text
                    Agentic AI eBook PDF
                            |
                            v
                    PyPDFLoader
                            |
                            v
                Text Splitting / Cleaning
                1000 chars / 200 overlap
                            |
                            v
                 Gemini Embeddings
                            |
                            v
                       Pinecone
                    Vector Database
                            |
                            v
                        LangGraph
                  +-------------------+
                  |                   |
                  |     Retrieve      |
                  |         |         |
                  |         v         |
                  |     Generate      |
                  |         |         |
                  +---------+---------+
                            |
                            v
                         FastAPI
                            |
                    +-------+-------+
                    |               |
                    v               v
                 Swagger        Streamlit
```

## Project Structure

``` text
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── gemini_embeddings.py
│   ├── ingestion.py
│   └── graph.py
│
├── app.py
├── streamlit_app.py
├── tests_sample_queries.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Technologies

-   Python
-   LangChain
-   LangGraph
-   Pinecone
-   Google Gemini API
-   FastAPI
-   Streamlit
-   PyPDF
-   Pydantic

## Embedding Implementation Note

The assignment specifies OpenAI embeddings with a 1536-dimensional
Pinecone index.

For this implementation, Google's `gemini-embedding-001` is used
instead, with the embedding output configured to **1536 dimensions**.
Pinecone is still used as the vector database and the retrieval workflow
remains the same.

The implementation keeps API credentials separate from the source code
using environment variables.

## 1. Clone the Repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd rag-agentic-ai
```

## 2. Create a Virtual Environment

Windows:

``` bash
python -m venv venv
```

Activate it:

``` bash
venv\Scripts\activate
```

## 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Create a `.env` file in the project root:

``` env
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-rag
```

Do not commit `.env` to GitHub.

Use `.env.example` as the template.

## 5. Create and Populate the Pinecone Index

Make sure the Agentic AI eBook is available at:

``` text
data/Ebook-Agentic-AI.pdf
```

Run:

``` bash
python -m src.ingestion
```

The ingestion pipeline:

1.  Loads the PDF using `PyPDFLoader`
2.  Splits the document into chunks
3.  Uses a 1000-character chunk size
4.  Uses 200 characters of overlap
5.  Removes known PDF noise such as QR-code content
6.  Generates embeddings
7.  Stores vectors and metadata in Pinecone

The Pinecone index uses:

``` text
Dimension: 1536
Metric: cosine
```

## 6. Run the FastAPI Backend

Start the API:

``` bash
uvicorn app:app --reload
```

Open Swagger UI:

``` text
http://127.0.0.1:8000/docs
```

The main endpoint is:

``` text
POST /chat
```

Request:

``` json
{
  "query": "What is Agentic AI according to the eBook?"
}
```

Response:

``` json
{
  "query": "What is Agentic AI according to the eBook?",
  "final_answer": "Answer generated from the retrieved eBook context.",
  "retrieved_context_chunks": [
    "Relevant eBook chunk 1...",
    "Relevant eBook chunk 2..."
  ],
  "confidence_score": 0.8
}
```

## 7. Run the Streamlit Frontend

Keep the FastAPI server running and open another terminal.

Run:

``` bash
streamlit run streamlit_app.py
```

The Streamlit application provides:

-   Question input
-   Generated answer
-   Confidence score
-   Retrieved context chunks

The frontend communicates with the FastAPI backend rather than directly
accessing the vector database or LLM.

## LangGraph Workflow

The chatbot uses a simple LangGraph pipeline:

``` text
START
  |
  v
Retrieve
  |
  v
Generate
  |
  v
END
```

### Retrieve Node

The retrieve node:

-   Receives the user's question
-   Performs similarity search against Pinecone
-   Retrieves the top relevant chunks
-   Applies a relevance threshold
-   Passes relevant context to the generation node

### Generate Node

The generate node:

-   Receives the question and retrieved context
-   Generates an answer using Gemini
-   Is instructed to use only the supplied eBook context
-   Returns a confidence score
-   Refuses to answer when sufficient information is not available

## Grounding and Out-of-Scope Handling

The chatbot is designed to avoid answering from general world knowledge.

If relevant eBook context cannot be retrieved, the application returns:

``` text
I don't have enough information in the provided eBook to answer that.
```

This is important for questions unrelated to the source document.

## Validation Queries

The implementation was validated using the following queries:

### 1. Definition

``` text
What is Agentic AI according to the eBook?
```

### 2. Comparison

``` text
How do AI agents differ from traditional automation systems?
```

### 3. Architecture

``` text
What are the core components of an Agentic Architecture?
```

### 4. Memory

``` text
What role does memory play in Agentic AI workflows?
```

### 5. Out-of-Scope Question

``` text
Who won the 2022 FIFA World Cup?
```

The final query is intentionally outside the scope of the eBook and
should be refused rather than answered using external knowledge.

## API Response Format

The `/chat` endpoint returns:

``` json
{
  "query": "...",
  "final_answer": "...",
  "retrieved_context_chunks": [
    "..."
  ],
  "confidence_score": 0.0
}
```

### Fields

  -----------------------------------------------------------------------
  Field                               Description
  ----------------------------------- -----------------------------------
  `query`                             User's original question

  `final_answer`                      Grounded answer generated from
                                      retrieved context

  `retrieved_context_chunks`          Relevant chunks retrieved from
                                      Pinecone

  `confidence_score`                  Model-generated confidence value
                                      between 0 and 1
  -----------------------------------------------------------------------

## Security

API credentials are loaded from environment variables.

The following files are intentionally excluded from Git:

``` text
.env
venv/
.venv/
__pycache__/
```

Never commit real Gemini or Pinecone API keys to a public repository.

## Running the Complete Application

Terminal 1:

``` bash
venv\Scripts\activate
uvicorn app:app --reload
```

Terminal 2:

``` bash
venv\Scripts\activate
streamlit run streamlit_app.py
```

Then open the Streamlit URL shown in the terminal.

## Future Improvements

Possible extensions include:

-   Reranking retrieved chunks
-   More advanced retrieval strategies
-   Citation/source highlighting in the UI
-   Retrieval evaluation metrics
-   Automated evaluation test suite
-   LangGraph conditional routing
-   Better confidence calibration
-   Conversation memory
-   Production deployment

## Author

**Tejeswar Dora**

AI / Data Engineering Project
