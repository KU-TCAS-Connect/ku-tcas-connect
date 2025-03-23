from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from main_question_extraction import question_extraction_csv, question_extraction_txt
from services.llm_question_extraction import QuestionExtraction
from services.llm_answer_not_related import AnswerQuestion
from search_txt import main_search_and_answer_txt
from main_query_classification import query_classification
from search_csv import main_search_and_answer_csv
from services.llm_synthesizer import Synthesizer
from uuid import uuid4
from dotenv import load_dotenv
from database.connectdb import VectorStore
from qdrant_client import QdrantClient, models
from services.logfile import save_log_infile
import os
import datetime

load_dotenv()

app = FastAPI()

# In-memory chat history storage
chat_histories: Dict[str, List[Dict[str, str]]] = {}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_class = VectorStore()
client = vector_class.qdrant_client
models = vector_class.qdrant_model

class QueryRequest(BaseModel):
    session_id: str  # Unique session ID for tracking history
    query: str

class QueryResponse(BaseModel):
    response: str

def rag_pipeline_csv(query: str, session_id: str, round_metadata: int, filename: str) -> str:
    """Generates a response while keeping track of conversation history."""
    history = chat_histories.get(session_id, [])

    answer = main_search_and_answer_csv(query, history, round_metadata=round_metadata)
    answer_answer = answer["answer"]
    answer_log = answer["log"]
    save_log_infile(filename=filename, content=answer_log)
    
    chat_histories[session_id] = history
    return answer_answer

def rag_pipeline_txt(query: str, session_id: str, round_metadata: int | None, filename: str) -> str:
    """Generates a response while keeping track of conversation history."""
    history = chat_histories.get(session_id, [])

    answer =  main_search_and_answer_txt(query, history, round_metadata=round_metadata, filename=filename)
    answer_answer = answer["answer"]
    answer_log = answer["log"]
    save_log_infile(filename=filename, content=answer_log)
    
    chat_histories[session_id] = history

    return answer_answer

def llm_completion(query: str, session_id: str) -> str:
    history = chat_histories.get(session_id, [])
    
    response = AnswerQuestion.generate_response(question=query, history=history)
    chat_histories[session_id] = history
    
    return response.answer

# def rag_pipeline_llm(query: str, session_id: str) -> str:
#     """Generates an open-ended response while keeping track of conversation history."""
#     history = chat_histories.get(session_id, [])
    
#     # Add the user query to the history
#     history.append({"role": "user", "content": query})

#     # Now, create the conversation history context for LLM
#     conversation_context = ""
#     for message in history:
#         conversation_context += f"{message['role']}: {message['content']}\n"
    
#     # Call the LLM to generate a response based on the conversation context
#     synthesizer = AnswerQuestion()  # Assuming Synthesizer handles LLM interaction
#     answer = synthesizer.get_response(conversation_context)  # Pass the entire history
    
#     # Store the assistant's response in history
#     history.append({"role": "assistant", "content": answer})
#     chat_histories[session_id] = history

#     return answer

@app.post("/rag-query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"log/output/{current_time}.txt"
    
    query = request.query

    # Query Classification
    classification_question = query_classification(query)
    search_table = classification_question["table"]
    classi_log = classification_question["log"]
    classi_log.append(f"query classify as: {search_table}" + "\n")
    
    save_log_infile(filename, classi_log)
    
    # print("query classify as:", search_table)
    
    if search_table == "csv":
        # Query Extraction
        is_complete, missing_fields, round_, log_list_extract_csv = question_extraction_csv(query)
        save_log_infile(filename=filename, content=log_list_extract_csv)
        if not is_complete:
            missing_str = ", ".join(missing_fields)
            return QueryResponse(response=f"โปรดเพิ่มให้ครบ ข้อมูลที่ขาดหายไปคือ {missing_str}")
        try:
            response = rag_pipeline_csv(query, request.session_id, round_metadata=int(round_), filename=filename)
            return QueryResponse(response=response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    elif search_table == "txt":
        round_, log_list_extract_txt = question_extraction_txt(query)
        save_log_infile(filename=filename, content=log_list_extract_txt)
        try:
            response = rag_pipeline_txt(query, request.session_id, round_metadata=round_, filename=filename)
            return QueryResponse(response=response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    elif search_table == "not_related":
        try:
            response = llm_completion(query, request.session_id)
            return QueryResponse(response=response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/new-session")
async def new_session():
    """Creates a new session ID for tracking history."""
    session_id = str(uuid4())
    chat_histories[session_id] = []
    return {"session_id": session_id}
