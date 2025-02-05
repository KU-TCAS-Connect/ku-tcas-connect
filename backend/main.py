from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from datetime import datetime
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer


vec = VectorStore()
app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost:3000",  # React dev server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Define the request model
class QueryRequest(BaseModel):
    query: str

# Define the response model
class QueryResponse(BaseModel):
    response: str

def similarlity_search(query):
    relevant_question = query
    results = vec.search(relevant_question, limit=3)

    response = Synthesizer.generate_response(question=relevant_question, context=results)
    return response.answer

# Mock function for RAG process (replace with actual implementation)
def rag_pipeline(query: str) -> str:
    generated_response = similarlity_search(query)
    return generated_response

@app.post("/rag-query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    try:
        response = rag_pipeline(request.query)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Hello World"}
