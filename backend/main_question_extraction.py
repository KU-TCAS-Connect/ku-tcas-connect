from services.llm_question_extraction import QuestionExtraction

def question_extraction_csv(query):
    thought_process, major, round_, program, program_type, is_complete, missing_fields = QuestionExtraction.extract(query)
    print(f"Extract User Question using LLM")
    print(f"Thought Process: {thought_process}")
    print(f"Major: {major}")
    print(f"Round: {round_}")
    print(f"Program: {program}")
    print(f"Program Type: {program_type}")
    print(f"Query is complete: {is_complete}")
    print(f"Missing fields: {missing_fields}")
    return is_complete, missing_fields, round_

def question_extraction_txt(query):
    round_ = QuestionExtraction.extract_txt(query)
    print(f"Extract User Question using LLM")
    print(f"Round: {round_}")
    return round_