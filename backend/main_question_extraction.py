from services.llm_question_extraction import QuestionExtraction

def question_extraction_csv(query):
    thought_process, major, round_, program, program_type, is_complete, missing_fields = QuestionExtraction.extract(query)
    log_list = []
    log_list.append(f"Extract User Question using LLM")
    log_list.append(f"Thought Process: {thought_process}")
    log_list.append(f"Major: {major}")
    log_list.append(f"Round: {round_}")
    log_list.append(f"Program: {program}")
    log_list.append(f"Program Type: {program_type}")
    log_list.append(f"Query is complete: {is_complete}")
    log_list.append(f"Missing fields: {missing_fields}")
    return is_complete, missing_fields, round_, log_list

def question_extraction_txt(query):
    log_list = []
    round_ = QuestionExtraction.extract_txt(query)
    log_list.append(f"Extract User Question using LLM")
    log_list.append(f"Round: {round_}")
    return round_, log_list