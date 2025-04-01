from services.llm_question_classification import QueryClassification
import os

def query_classification(query):
    log_list = []
    query_classification = QueryClassification()
    search_table = ""
    category = query_classification.classify(query)
    log_list.append(f"คำถาม: {query} → จัดหมวดหมู่เป็น: {category}")
    

    category = category.intent

    os.environ["QUERY"] = query  # Set environment variable

    if category == "general_info":
        log_list.append("Search in table txt")
        search_table = "txt"
    elif category == "admission_criteria":
        log_list.append("Search in table csv")
        search_table = "csv"
    elif category == "not_related":
        log_list.append("Bypass search in Vector Database")
        search_table = "not_related"
    return {"table":search_table,"log": log_list}
