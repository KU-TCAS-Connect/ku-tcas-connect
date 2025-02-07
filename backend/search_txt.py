import openai
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue
from FlagEmbedding import BGEM3FlagModel

from services.llm_retrieve_filter import RetrieveFilter
from services.question_extraction import QuestionExtraction, QuestionExtractionResponse
from database.connectdb import VectorStore
from services.bge_embedding import FlagModel
from services.llm_synthesizer import Synthesizer
import pandas as pd

load_dotenv()

vector_class = VectorStore()
client = vector_class.qdrant_client
models = vector_class.qdrant_model

flag_class = FlagModel()
model = flag_class.bge_model

client = QdrantClient("http://localhost:6333")

# Initialize the BGEM3 model
bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

def compute_sparse_vector(text):
    sentences_1 = [text]  # Use the content of the row for encoding
    output_1 = bge_model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    
    # Extract the lexical weights (this is your sparse representation)
    lexical_weights = output_1['lexical_weights'][0]

    # Convert the lexical weights into a dictionary (index: weight)
    sparse_vector_dict = {token: weight for token, weight in lexical_weights.items()}

    indices = list(sparse_vector_dict.keys())  # Indices of the sparse vector
    values = list(sparse_vector_dict.values())  # Values of the sparse vector
    native_floats = [float(x) for x in values]
    new_dict = dict(zip(indices, native_floats))
    return indices, native_floats

def generate_bge_embedding(text):
    try:
        # Generate dense embedding using BGE model
        sentences_1 = [text]  # The content you want to encode
        output_1 = bge_model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=False)

        # Extract the dense vector (embedding)
        dense_vector = output_1['dense_vecs'][0]

        return dense_vector
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def search_similar_vectors(query_text, top_k=5):
    query_embedding = generate_bge_embedding(query_text)  # Use BGE model for dense vector
    if not query_embedding:
        print("Failed to generate query embedding.")
        return

    search_results = client.search(
        collection_name=vector_class.col_setting.collection_name["txt"],
        query_vector=query_embedding,
        limit=top_k
    )

    for result in search_results:
        print(f"Found ID: {result.id}, Score: {result.score}, Metadata: {result.payload}")

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

#################### Main ####################
chat_history = []  # Initialize chat history

query = "อยากทราบกำหนดการการประกาศผลผู้ผ่านการคัดเลือก "
query_indices, query_values = compute_sparse_vector(query)

search_result = client.query_points(
    collection_name=vector_class.col_setting.collection_name["txt"],
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

################### Print the search results (Retrieve Document) ####################
print("#################### Print the search results (Retrieve Document) ####################")
for result in search_result.points:
    print(f"Score: {result.score}")
    print(f"""{result.payload["admission_program"]}\n{result.payload["contents"]}\n{result.payload["reference"]}""")
    print("---------------------------------")

# Extract retrieved documents from search_result
document_from_db_before_filter = []
for result in search_result.points:
    document_content = f"""{result.payload["admission_program"]}\n{result.payload["contents"]}\n{result.payload["reference"]}"""
    document_from_db_before_filter.append(document_content)

context_str_after_filtered = RetrieveFilter.filter(query=query, documents=document_from_db_before_filter)

print("--------------------------------- Print Filtered Document ---------------------------------")
print("Index of Filtered Document:\n", context_str_after_filtered.idx)
print("Filtered Document Conent:\n", context_str_after_filtered.content)
print("Reason why filter out:\n", context_str_after_filtered.reject_reasons)

################### QuestionExtraction ####################
# print("--------------------------------- QuestionExtraction ---------------------------------")
# thought_process, major, round_, program, program_type = QuestionExtraction.extract(query, QuestionExtractionResponse)
# print(f"Extract from User Question using LLM Question Checker")
# print(thought_process)
# print(f"Major: {major}")
# print(f"Round: {round_}")
# print(f"Program: {program}")
# print(f"Program Type: {program_type}")

################### Generate Answer by LLM ####################
print("--------------------------------- Generate Answer by LLM ---------------------------------")
print("First Question")
# First question
response1 = Synthesizer.generate_response(
    question=query, 
    context=create_dataframe_from_results(search_result), 
    history=chat_history
)
print("Answer Question:", response1.answer)

# # Second question (same chat session, keeps context)
# print("Second Question")
# response2 = Synthesizer.generate_response(
#     question="แล้วค่าเทอมเท่าไหร่?", 
#     context=create_dataframe_from_results(search_result), 
#     history=chat_history  # Keeps previous messages
# )
# print("Answer Secodn Question", response2.answer)
