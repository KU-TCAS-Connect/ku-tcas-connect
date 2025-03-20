from services.llm_open_close_question_classify import QuestionTypeClassifier

def query_type_classification(query):
    question_type_classifier = QuestionTypeClassifier()
    question_type_llm_response = question_type_classifier.classify(query)
    question_type = question_type_llm_response.question_type
    print(f"คำถาม: {query} → จัดเป็นคำถามแบบ: {question_type}")

    return question_type
