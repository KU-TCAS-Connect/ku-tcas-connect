import os
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, Tuple, List
from typing import Any, Dict, Type
from services.llm_factory import LLMFactory

class QuestionExtractionResponse(BaseModel):
    thought_process: List[str] = Field(
        description="List of thoughts that the AI assistant had while extracting the user query"
    )
    major: str
    round: Optional[int] = Field(
        description="The admission round number, or None if not found",
        examples=["1", "2", "3"]
    )
    program: str
    program_type: str

class QuestionExtractionResponseTxt(BaseModel):
    round: Optional[int] = Field(
        description="The admission round number, or None if not found",
        examples=["1", "2", "3"]
    )

class QuestionExtraction:

    SYSTEM_PROMPT_CSV = """
        # Role and Purpose
        You are an AI assistant that extract user's query and check if a user's query has enough context to query a database. 
        Your task is to ensure that the query contains all necessary information based on specific rules 
        and guidelines provided below. Do not assume missing fields. If a field is not provided, leave it blank. Please respond in Thai language only.

        # Knowledge
        for example: สาขาวิชา: วศ.บ. สาขาวิชาวิศวกรรมเครื่องกล (ภาษาไทย ปกติ) รอบ 1/2 ช้างเผือก
        1. Major as สาขาวิชา (From example will be วิศวกรรมเครื่องกล). 
        The term 'major' may be written in different ways, such as 'สาขาวิชา', 'สาขา', or other equivalent terms.
        2. Round as รอบการคัดเลือก (From example will be 1/2). 
        The term 'round' may appear as 'รอบ', 'รอบการคัดเลือก', or other similar terms. There are 3 round
            - 1: can be 1, 1/1, or 1/2
            - 2
            - 3
        3. Program as โครงการ (From example will be ช้างเผือก). 
        The term 'program' may be written as 'โครงการ' or other synonyms.
        For round 1 (รอบ 1) have follwing program:
            - เรียนล่วงหน้า
            - นานาชาติและภาษาอังกฤษ
            - โอลิมปิกวิชาการ
            - ผู้มีความสามารถทางกีฬาดีเด่น
            - ช้างเผือก
        For round 2 (รอบ 2) have follwing program:
            - เพชรนนทรี
            - นานาชาติและภาษาอังกฤษ
            - ความร่วมมือในการสร้างเครือข่ายทางการศึกษากับมหาวิทยาลัยเกษตรศาสตร์
            - ลูกพระพิรุณ
            - โควตา 30 จังหวัด
            - รับนักเรียนดีเด่นจากโรงเรียนสาธิตแห่งมหาวิทยาลัยเกษตรศาสตร์
            - ผู้มีความสามารถทางกีฬา
        For round 3 (รอบ 3) have only one program:
            - Admission
        4. Program Type have follwing example (From example will be ภาษาไทย ปกติ or ปกติ:
            - ปกติ or ภาคปกติ
            - พิเศษ or ภาคพิเศษ
            - นานาชาติ or ภาคนานาชาติ
            - ภาษาไทย พิเศษ or ภาคภาษาไทย พิเศษ
            - ภาษาไทย ปกติ or ภาคภาษาไทย ปกติ
            - ภาษาอังกฤษ or ภาคภาษาอังกฤษ
            - ภาษาต่างประเทศ or ภาคภาษาต่างประเทศ
        
        # Rules
        1. Do not assume missing fields. If a field is not provided, leave it blank.
        2. Do not infer values unless explicitly mentioned.
        3. If "International Program" (นานาชาติ) appears in either Program or Program Type, assume:
            - Program Type = นานาชาติ
            - Program = นานาชาติและภาษาอังกฤษ
            or if "English Program" (ภาษาต่างประเทศ, ภาษาอังกฤษ) appears in either Program or Program Type, assume:
            - Program Type = ภาษาต่างประเทศ
            - Program = นานาชาติและภาษาอังกฤษ
        4. Round 3 always has Program = Admission (if Round 3 is specified).
        5. User DOES NOT NEED to input Condtion (เงื่อนไขขั้นต่ำ) and Criteria (เกณฑ์การพิจารณา).

        Additionally, extract and return the following fields from the user's query:
        1. **Major** (สาขาวิชา/สาขา)
        2. **Round** (รอบการคัดเลือก/รอบ)
        3. **Program** (โครงการ)
        4. **Program type** (ภาค)

        Ensure that you mention which information is missing and what the user needs to add to complete the query.
        """

    SYSTEM_PROMPT_TXT = """
        # Role and Purpose
        You are an AI assistant that extracts **only the Round number** from a user's query.  
        You must identify and return one of the following rounds: **1, 1/1, 1/2, 2, 3**.
        If the round is not found, return an empty value.
    """

    @staticmethod
    def extract(text) -> Tuple[List[str], str, int, str, str]:
        llm = LLMFactory("openai")
        # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # llm_response = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
            # messages=[
            #     {"role": "system", "content": QuestionExtraction.SYSTEM_PROMPT},
            #     {"role": "user", "content": "Extract: " + text},            
            # ],
        #     max_tokens=3000,
        #     # max_retries=3,
        #     temperature=0.1,
        #     response_model=template,
        # )
        
        # completion_params = {
        #     "model": "gpt-3.5-turbo",
        #     "temperature": 0.1,
        #     # "max_retries": 3,
        #     "max_tokens": 3000,
        #     "response_model": template,
        #     "messages":[
        #         {"role": "system", "content": QuestionExtraction.SYSTEM_PROMPT},
        #         {"role": "user", "content": "Extract: " + text},            
        #     ],
        # }
        llm_response = llm.create_completion(
            messages=[
                {"role": "system", "content": QuestionExtraction.SYSTEM_PROMPT_CSV},
                {"role": "user", "content": "Extract: " + text},
            ],
            response_model=QuestionExtractionResponse,
        )
        thought_process = llm_response.thought_process
        major = llm_response.major
        round_ = llm_response.round
        print("round_", round_)
        program = llm_response.program
        program_type = llm_response.program_type

        # Check completeness
        missing_fields = []
        if not major:
            missing_fields.append("Major (สาขาวิชา)")
        if not round_:
            missing_fields.append("Round (รอบการคัดเลือก)\n  รอบการคัดเลือกที่มี : 1, 1/1, 1/2, 2, 3")
        if not program and round_ != 3:  # Assume round 3 defaults to "Admission"
            missing_fields.append("Program (โครงการ)\nโครงการรอบ1 : เรียนล่วงหน้า, นานาชาติและภาษาอังกฤษ, โอลิมปิกวิชาการ, ผู้มีความสามารถทางกีฬาดีเด่น, ช้างเผือก\nโครงการรอบ2 : เพชรนนทรี, นานาชาติและภาษาอังกฤษ, ความร่วมมือในการสร้างเครือข่ายทางการศึกษากับมหาวิทยาลัยเกษตรศาสตร์, ลูกพระพิรุณ, โควตา 30 จังหวัด, รับนักเรียนดีเด่นจากโรงเรียนสาธิตแห่งมหาวิทยาลัยเกษตรศาสตร์, ผู้มีความสามารถทางกีฬา\nโครงการรอบ3 : Admission")
        if not program_type:
            missing_fields.append("Program Type (ภาค): ปกติ, พิเศษ, นานาชาติ, ภาษาอังกฤษ")

        is_complete = len(missing_fields) == 0
        
        return thought_process, major, round_, program, program_type, is_complete, missing_fields

    @staticmethod
    def extract_txt(text) -> Optional[int]:
        llm = LLMFactory("openai")
        llm_response = llm.create_completion(
            messages=[
                {"role": "system", "content": QuestionExtraction.SYSTEM_PROMPT_TXT},
                {"role": "user", "content": "Extract: " + text},
            ],
            response_model=QuestionExtractionResponseTxt,
        )

        round_ = llm_response.round
        return round_ if round_ else None