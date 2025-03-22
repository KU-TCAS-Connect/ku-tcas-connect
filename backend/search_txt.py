from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue

import datetime

from utils import compute_sparse_vector, create_dataframe_from_results, generate_bge_embedding
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

    search_result = hybrid_search_txt_documents(query=query, top_k=2)

    ################### Print the search results (Retrieve Document) ####################
    print("#################### Print the search results (Retrieve Document) ####################")
    for result in search_result.points:
        print(f"Score: {result.score}")
        print(f"""{result.payload["admission_program"]}\n{result.payload["contents"]}\n{result.payload["reference"]}""")
        print("---------------------------------")

    # print("---------------- Extract retrieved documents from search_result --------------------")
    # document_from_db_before_filter = []
    # for result in search_result.points:
    #     document_content = f"""{result.payload["admission_program"]}\n{result.payload["contents"]}\n{result.payload["reference"]}"""
    #     document_from_db_before_filter.append(document_content)

    # context_str_after_filtered = RetrieveFilter.filter(query=query, documents=document_from_db_before_filter)

    # print("--------------------------------- Print Filtered Document ---------------------------------")
    # print("Index of Filtered Document:\n", context_str_after_filtered.idx)
    # print("Filtered Document Content:\n", context_str_after_filtered.content)
    # print("Reason why filter out:\n", context_str_after_filtered.reject_reasons)

    # print("--------------------------------- Prepare filtered documents before send to LLM ---------------------------------")
    # filtered_indices_list = context_str_after_filtered.idx
    # filtered_indices_list = [(int(x) - 1) for x in filtered_indices_list]
    # df_of_search_result = create_dataframe_from_results(search_result)
    # df_filtered = df_of_search_result.loc[df_of_search_result.index.isin(filtered_indices_list)]
    # print("df_of_search_result", df_of_search_result)
    # print("df_filterd", df_filtered)

    # ################### QuestionExtraction ####################
    # # print("--------------------------------- QuestionExtraction ---------------------------------")
    # # thought_process, major, round_, program, program_type = QuestionExtraction.extract(query, QuestionExtractionResponse)
    # # print(f"Extract from User Question using LLM Question Checker")
    # # print(thought_process)
    # # print(f"Major: {major}")
    # # print(f"Round: {round_}")
    # # print(f"Program: {program}")
    # # print(f"Program Type: {program_type}")

    # ################### Generate Answer by LLM ####################
    # print("--------------------------------- Generate Answer by LLM ---------------------------------")
    # response = AnswerQuestion.generate_response(
    #     question=query, 
    #     context=df_filtered, 
    #     history=chat_history_list
    # )
    # print("Answer Question:", response.answer)
    
    # ####### LOG #######
    # current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    # filename = f"log/output/txt/{current_time}.txt"

    # if not os.path.exists(filename):
    #     open(filename, 'w', encoding="utf-8").close()

    # with open(f"{filename}", "a",  encoding="utf-8") as file:
    #     for result in search_result.points:
    #         file.write(f"Score: {result.score}" + "\n")
    #         # file.write(f"""{result.payload["admission_program"]}\n{result.payload["admission_round"]}\n{result.payload["contents"]}\n{result.payload["reference"]}""" + "\n")
    #         file.write(f"""{result.payload.get("admission_program", "")}\n{result.payload.get("admission_round", "N/A")}\n{result.payload.get("contents", "")}\n{result.payload.get("reference", "")}\n""")
    #         file.write(f"---------------------------------" + "\n")
            
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
    #     file.write(f"df_filterd")
    #     file.write(str(df_filtered))
    #     file.write("\n")
    #     file.write(f"--------------------------------- Generate Answer by LLM ---------------------------------"+"\n")
    #     file.write(f"Answer Question:\n")
    #     file.write(str(response.answer))

    return "response.answer"