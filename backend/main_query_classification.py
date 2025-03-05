from services.llm_question_classification import QueryClassification
import os

def query_classification(query):
    query_classification = QueryClassification()
    search_table = ""
    category = query_classification.classify(query)
    print(f"คำถาม: {query} → จัดหมวดหมู่เป็น: {category}")

    category = category.intent

    os.environ["QUERY"] = query  # Set environment variable

    if category == "general_info":
        print("Search in table txt")
        search_table = "txt"
    elif category == "admission_criteria":
        print("Search in table csv")
        search_table = "csv"
    elif category == "count_criteria":
        print("Search in table csv count_criteria")
        search_table = "csv_count_criteria"
    elif category == "not_related":
        print("Bypass search in Vector Database")
        search_table = "not_related"
    return search_table
