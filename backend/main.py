from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from search_csv import compute_sparse_vector, create_dataframe_from_results, generate_bge_embedding
from services.llm_synthesizer import Synthesizer
from uuid import uuid4
from dotenv import load_dotenv
from database.connectdb import VectorStore
from qdrant_client import QdrantClient, models
import os

load_dotenv()

app = FastAPI()

# In-memory chat history storage
chat_histories: Dict[str, List[Dict[str, str]]] = {}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_class = VectorStore()
client = vector_class.qdrant_client
models = vector_class.qdrant_model

client = QdrantClient("http://localhost:6333")

class QueryRequest(BaseModel):
    session_id: str  # Unique session ID for tracking history
    query: str

class QueryResponse(BaseModel):
    response: str

def search(query):
    query_indices, query_values = compute_sparse_vector(query)

    search_result = client.query_points(
        collection_name=vector_class.col_setting.collection_name["csv"],
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(indices=query_indices, values=query_values),
                using="keywords",
                limit=1,
            ),
            models.Prefetch(
                query=generate_bge_embedding(query),  # <-- dense vector using BGE model
                using="",
                limit=1,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
    )

    return search_result

def rag_pipeline(query: str, session_id: str) -> str:
    """Generates a response while keeping track of conversation history."""
    history = chat_histories.get(session_id, [])

    search_result = search(query)
    context = create_dataframe_from_results(search_result)
    response = Synthesizer.generate_response(question=query, context=context, history=history)

    # Store the conversation history
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": response.answer})
    chat_histories[session_id] = history

    return response.answer

@app.post("/rag-query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    try:
        response = rag_pipeline(request.query, request.session_id)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/new-session")
async def new_session():
    """Creates a new session ID for tracking history."""
    session_id = str(uuid4())
    chat_histories[session_id] = []
    return {"session_id": session_id}
