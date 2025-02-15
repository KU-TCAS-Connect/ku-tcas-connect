from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue

from utils import compute_sparse_vector, create_dataframe_from_results, generate_bge_embedding
from services.llm_answer import AnswerQuestion
from services.llm_retrieve_filter import RetrieveFilter
from services.llm_question_extraction import QuestionExtraction, QuestionExtractionResponse
from database.connectdb import VectorStore
from services.llm_synthesizer import Synthesizer

load_dotenv()

vector_class = VectorStore()
client = vector_class.qdrant_client
models = vector_class.qdrant_model

client = QdrantClient("http://localhost:6333")

def hybrid_search_txt_documents(query, top_k=1):
    query_indices, query_values = compute_sparse_vector(query)

    search_result = client.query_points(
        collection_name=vector_class.col_setting.collection_name["txt"],
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(indices=query_indices, values=query_values),
                using="keywords",
                limit=top_k,
            ),
            models.Prefetch(
                query=generate_bge_embedding(query),  # <-- dense vector using BGE model
                using="",
                limit=top_k,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
    )

    return search_result

#################### Main ####################

def main_search_and_answer_txt(user_question, chat_history):
    chat_history_list = chat_history  # Initialize chat history

    # query = "อยากทราบกำหนดการการประกาศผลผู้ผ่านการคัดเลือก"
    query = user_question
    print(f"Received query: {query}")

    search_result = hybrid_search_txt_documents(query=query, top_k=1)

    ################### Print the search results (Retrieve Document) ####################
    print("#################### Print the search results (Retrieve Document) ####################")
    for result in search_result.points:
        print(f"Score: {result.score}")
        print(f"""{result.payload["admission_program"]}\n{result.payload["contents"]}\n{result.payload["reference"]}""")
        print("---------------------------------")

    print("---------------- Extract retrieved documents from search_result --------------------")
    document_from_db_before_filter = []
    for result in search_result.points:
        document_content = f"""{result.payload["admission_program"]}\n{result.payload["contents"]}\n{result.payload["reference"]}"""
        document_from_db_before_filter.append(document_content)

    context_str_after_filtered = RetrieveFilter.filter(query=query, documents=document_from_db_before_filter)

    print("--------------------------------- Print Filtered Document ---------------------------------")
    print("Index of Filtered Document:\n", context_str_after_filtered.idx)
    print("Filtered Document Content:\n", context_str_after_filtered.content)
    print("Reason why filter out:\n", context_str_after_filtered.reject_reasons)

    print("--------------------------------- Prepare filtered documents before send to LLM ---------------------------------")
    filtered_indices_list = context_str_after_filtered.idx
    filtered_indices_list = [(int(x) - 1) for x in filtered_indices_list]
    df_of_search_result = create_dataframe_from_results(search_result)
    df_filtered = df_of_search_result.loc[df_of_search_result.index.isin(filtered_indices_list)]
    print("df_of_search_result", df_of_search_result)
    print("df_filterd", df_filtered)

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
    response = AnswerQuestion.generate_response(
        question=query, 
        context=df_filtered, 
        history=chat_history_list
    )
    print("Answer Question:", response.answer)
    return response.answer

if __name__ == "__main__":
    question = "ถามคำถาม"
    chat_history = []
    main_search_and_answer_txt(user_question=question, chat_history=chat_history)