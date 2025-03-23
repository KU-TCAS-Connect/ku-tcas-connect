from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue

import datetime

from utils import compute_sparse_vector, create_dataframe_from_results, generate_bge_embedding, reranker_process
from services.llm_answer_txt import AnswerQuestion
from services.llm_retrieve_filter import RetrieveFilter
from services.llm_question_extraction import QuestionExtraction, QuestionExtractionResponse
from database.connectdb import VectorStore
from services.llm_synthesizer import Synthesizer
import os

load_dotenv()

vector_class = VectorStore()
client = vector_class.qdrant_client
models = vector_class.qdrant_model

client = QdrantClient("http://localhost:6333")

def hybrid_search_txt_documents(query, round_metadata, top_k=1):
    query_indices, query_values = compute_sparse_vector(query)

    # Base search query
    search_filter = []
    if round_metadata is not None:
        search_filter.append(
            models.FieldCondition(
                key="admission_round",
                match=models.MatchValue(value=round_metadata),
            )
        )
    
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
        query_filter=models.Filter(must=search_filter) if search_filter else None,  # Apply filter only if needed
    )
    
    return search_result

#################### Main ####################

def main_search_and_answer_txt(user_question, chat_history, round_metadata, filename):
    chat_history_list = chat_history  # Initialize chat history

    # query = "อยากทราบกำหนดการการประกาศผลผู้ผ่านการคัดเลือก"
    query = user_question
    print(f"Received query: {query}")

    search_result = hybrid_search_txt_documents(query=query, round_metadata=round_metadata, top_k=2)
    sorted_list_of_index_and_score_rerank = reranker_process(query=query, document_list=[result.payload["contents"] for result in search_result.points])

    ################### Print the search results (Retrieve Document) ####################
    print("#################### Print the search results (Retrieve Document) ####################")
    for result in search_result.points:
        print(f"Score: {result.score}")
        print(f"""{result.payload["admission_program"]}\n{result.payload["contents"]}\n{result.payload["reference"]}""")
        print("---------------------------------")

    ################### Print the rerank results ####################
    print("#################### Print the rerank results ####################")
    print("sorted_list_of_index_and_score_rerank:", sorted_list_of_index_and_score_rerank)

    ################### Extract reranked documents ###################
    document_from_db_after_rerank = []
    for index, rerank_score in sorted_list_of_index_and_score_rerank:
        result = search_result.points[index]
        document_content = f"""{result.payload["admission_program"]}\n{result.payload["contents"]}\n{result.payload["reference"]}"""
        document_from_db_after_rerank.append(document_content)

        print(f"Rerank Score: {rerank_score}")
        print(document_content)
        print("---------------------------------")

    ################### Send reranked documents to filter ###################
    context_str_after_filtered = RetrieveFilter.filter(query=query, documents=document_from_db_after_rerank)

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
    print("df_filtered", df_filtered)

    ################### Generate Answer by LLM ####################
    print("--------------------------------- Generate Answer by LLM ---------------------------------")
    response = AnswerQuestion.generate_response(
        question=query, 
        context=df_filtered, 
        history=chat_history_list
    )
    print("Answer Question:", response.answer)
    
    ####### LOG #######
    # current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    # filename = f"log/output/txt/{current_time}.txt"

    # if not os.path.exists(filename):
    #     open(filename, 'w', encoding="utf-8").close()

    # with open(f"{filename}", "a",  encoding="utf-8") as file:
    #     file.write(f"User Question: {user_question}" + "\n" + "\n")
    #     file.write(f"########### Search results (Retrieve Document) ###########" + "\n")
    #     for result in search_result.points:
    #         file.write(f"Score: {result.score}" + "\n")
    #         # file.write(f"""{result.payload["admission_program"]}\n{result.payload["admission_round"]}\n{result.payload["contents"]}\n{result.payload["reference"]}""" + "\n")
    #         file.write(f"""{result.payload.get("admission_program", "")}\n{result.payload.get("admission_round", "N/A")}\n{result.payload.get("contents", "")}\n{result.payload.get("reference", "")}\n""")
    #         file.write(f"---------------------------------" + "\n")
        
    #     file.write(f"########### Sorted list of index and score rerank ###########" + "\n")
    #     file.write(f"sorted_list_of_index_and_score_rerank:, {sorted_list_of_index_and_score_rerank}")


    #     for index, rerank_score in sorted_list_of_index_and_score_rerank:
    #         result = search_result.points[index]
    #         document_content = f"""{result.payload["admission_program"]}\n{result.payload["contents"]}\n{result.payload["reference"]}"""
    #         document_from_db_after_rerank.append(document_content)

    #         print(f"Rerank Score: {rerank_score}")
    #         print(document_content)
    #         print("---------------------------------")
            
    #     file.write(f"########### Rerank results ###########" + "\n")
    #     for index, rerank_score in sorted_list_of_index_and_score_rerank:
    #         result = search_result.points[index]
    #         document_content = f"""{result.payload["admission_program"]}\n{result.payload["contents"]}\n{result.payload["reference"]}\n"""

    #         file.write(f"Rerank Score: {rerank_score}" + "\n")
    #         file.write(f"{document_content}" + "\n")
    #         file.write("---------------------------------" + "\n")

    #     file.write(f"--------------------------------- Print Filtered Document ---------------------------------"+"\n")
    #     file.write(f"Index of Filtered Document:\n")
    #     file.write(str(context_str_after_filtered.idx))
    #     file.write("\n")
    #     file.write(f"Filtered Document Content:\n")
    #     file.write(str(context_str_after_filtered.content))
    #     file.write("\n")
    #     file.write(f"Reason why filter out:\n")
    #     file.write(str(context_str_after_filtered.reject_reasons))
    #     file.write("\n")
    #     file.write(f"--------------------------------- Prepare filtered documents before send to LLM ---------------------------------"+"\n")
    #     file.write(f"df_filtered")
    #     file.write(str(df_filtered))
    #     file.write("\n")
    #     file.write(f"--------------------------------- Generate Answer by LLM ---------------------------------"+"\n")
    #     file.write(f"Answer Question:\n")
    #     file.write(str(response.answer))

    log_list = []

    log_list.append(f"User Question: {user_question}" + "\n" + "\n")
    log_list.append(f"########### Search results (Retrieve Document) ###########" + "\n")
    search_result_point = []    
    for result in search_result.points:
        search_result_point.append(f"Score: {result.score}" + "\n")
        # file.write(f"""{result.payload["admission_program"]}\n{result.payload["admission_round"]}\n{result.payload["contents"]}\n{result.payload["reference"]}""" + "\n")
        search_result_point.append(f"""{result.payload.get("admission_program", "")}\n{result.payload.get("admission_round", "N/A")}\n{result.payload.get("contents", "")}\n{result.payload.get("reference", "")}\n""")
        search_result_point.append(f"---------------------------------" + "\n")
    log_list.append(search_result_point)
    log_list.append(f"########### Sorted list of index and score rerank ###########" + "\n")
    log_list.append(f"sorted_list_of_index_and_score_rerank:, {sorted_list_of_index_and_score_rerank}")

    rerank_with_doc = []
    for index, rerank_score in sorted_list_of_index_and_score_rerank:
        result = search_result.points[index]
        document_content = f"""{result.payload["admission_program"]}\n{result.payload["contents"]}\n{result.payload["reference"]}"""
        document_from_db_after_rerank.append(document_content)

        rerank_with_doc.append(f"Rerank Score: {rerank_score}")
        rerank_with_doc.append(document_content)
        rerank_with_doc.append("---------------------------------")
    log_list.append(rerank_with_doc)
    
    rerank_result = []
    log_list.append(f"########### Rerank results ###########" + "\n")
    for index, rerank_score in sorted_list_of_index_and_score_rerank:
        result = search_result.points[index]
        document_content = f"""{result.payload["admission_program"]}\n{result.payload["contents"]}\n{result.payload["reference"]}\n"""

        rerank_result.append(f"Rerank Score: {rerank_score}" + "\n")
        rerank_result.append(f"{document_content}" + "\n")
        rerank_result.append("---------------------------------" + "\n")
    log_list.append(rerank_result)
    
    log_list.append(f"--------------------------------- Print Filtered Document ---------------------------------"+"\n")
    log_list.append(f"Index of Filtered Document:\n")
    log_list.append(str(context_str_after_filtered.idx))
    log_list.append("\n")
    log_list.append(f"Filtered Document Content:\n")
    log_list.append(str(context_str_after_filtered.content))
    log_list.append("\n")
    log_list.append(f"Reason why filter out:\n")
    log_list.append(str(context_str_after_filtered.reject_reasons))
    log_list.append("\n")
    log_list.append(f"--------------------------------- Prepare filtered documents before send to LLM ---------------------------------"+"\n")
    log_list.append(f"df_filtered")
    log_list.append(str(df_filtered))
    log_list.append("\n")
    log_list.append(f"--------------------------------- Generate Answer by LLM ---------------------------------"+"\n")
    log_list.append(f"Answer Question:\n")
    log_list.append(str(response.answer))

    return {
        "answer":response.answer,
        "log": log_list
        }