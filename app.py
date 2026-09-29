from fastapi import FastAPI
from pydantic import BaseModel,Field
from src.graph import graph

app=FastAPI(
  title='Agentic AI RAG Chatbot',
  description='LangGraph + Pinecone RAG chatbot for the Agentic AI eBook',
  version='1.0.0'
)

# Request Model

class ChatRequest(BaseModel):
  query:str=Field(
    ...,
    min_length=1,
    description='Question About the Agentic AI eBook'
  )

# Response Model
class ChatResponse(BaseModel):
  query: str
  final_answer:str
  retrieved_context_chunks:list[str]
  confidence_score:float

# Health Check
@app.get("/")
def root():
  return{
    'message':'Agentic AI RAG Chatbot API is running'
  }

# Chat Endpoint
@app.post('/chat',response_model=ChatResponse)
def chat(request:ChatRequest):
    try:

        result = graph.invoke({
            "question": request.query,
            "context": [],
            "answer": "",
            "score": 0.0,
        })

        return ChatResponse(
            query=request.query,
            final_answer=result["answer"],
            retrieved_context_chunks=result["context"],
            confidence_score=result["score"],
        )

    except Exception as e:

        print("\n========== CHAT ERROR ==========")
        print(type(e).__name__)
        print(str(e))
        print("================================\n")

        raise
  