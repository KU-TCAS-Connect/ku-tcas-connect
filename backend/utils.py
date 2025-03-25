from FlagEmbedding import BGEM3FlagModel, FlagReranker
from services.llm_question_extraction import QuestionExtraction
from services.bge_embedding import FlagModel

import pandas as pd
import torch
import datetime
import os

device = "cuda" if torch.cuda.is_available() else "CPU"
print(f"Using device: {device}")

flag_class = FlagModel()
model = flag_class.bge_model

bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

def create_dataframe_from_results(results) -> pd.DataFrame:
    data = []
    for result in results.points:
        row = {
            "id": result.id,
            "score": result.score,
            "admission_program": result.payload.get("admission_program", ""),
            "content": result.payload.get("contents", ""),
            "reference": result.payload.get("reference", ""),
        }
        data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Convert id to string for better readability
    df["id"] = df["id"].astype(str)

    # Display the DataFrame
    return df

def create_dataframe_for_rerank(results_list) -> pd.DataFrame:
    data = []
    if (not results_list):
        return pd.DataFrame(data)
    for result in results_list:
        row = {
            "id": result.id,
            "score": result.score,
            "admission_program": result.payload.get("admission_program", ""),
            "content": result.payload.get("contents", ""),
            "reference": result.payload.get("reference", ""),
        }
        data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Convert id to string for better readability
    df["id"] = df["id"].astype(str)

    # Display the DataFrame
    return df

def generate_bge_embedding(text):
    try:
        sentences_1 = [text]
        
        output_1 = model.encode(
            sentences_1, 
            return_dense=True, 
            return_sparse=True, 
            return_colbert_vecs=False
        )

        dense_vector = torch.tensor(output_1['dense_vecs'][0], device=device)  # Move to GPU

        return dense_vector.cpu().numpy()  # Convert back to NumPy before sending to Qdrant
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None
    
def compute_sparse_vector(text):
    sentences_1 = [text]  # Use the content of the row for encoding
    output_1 = model.encode(
        sentences_1, 
        return_dense=True, 
        return_sparse=True, 
        return_colbert_vecs=False
    )

    lexical_weights = output_1['lexical_weights'][0]

    sparse_vector_dict = {token: weight for token, weight in lexical_weights.items()}

    indices = list(sparse_vector_dict.keys())
    values = [float(x) for x in list(sparse_vector_dict.values())]

    return indices, values

def reranker_process(query, document_list):
    keep_document_list = [[query, document] for document in document_list]
    scores = reranker.compute_score(keep_document_list, normalize=True)
    
    def sort_by_score(item):
        return item[1]

    sorted_scores = sorted(enumerate(scores), key=sort_by_score, reverse=True)

    return sorted_scores  # Returns a list of (index, score) tuples

def log_message_csv(message):
    """Logs a message to the same file used for main search logs and prints it to the terminal."""
    print(message)  # Print to terminal

    # Define log file path
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"log/output/csv/{current_time}.txt"

    # Ensure the file exists
    if not os.path.exists(filename):
        open(filename, 'w', encoding="utf-8").close()

    # Append log message to file
    with open(filename, "a", encoding="utf-8") as file:
        file.write(message + "\n")

def log_message_txt(message):
    """Logs a message to the same file used for main search logs and prints it to the terminal."""
    print(message)  # Print to terminal

    # Define log file path
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"log/output/txt/{current_time}.txt"

    # Ensure the file exists
    if not os.path.exists(filename):
        open(filename, 'w', encoding="utf-8").close()

    # Append log message to file
    with open(filename, "a", encoding="utf-8") as file:
        file.write(message + "\n")