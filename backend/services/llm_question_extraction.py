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
    round: int = Field(
        description="The admission round number",
        examples=["1", "1/1", "1/2", "2", "3"]
    )
    program: str
    program_type: str
    
class QuestionExtraction:

    SYSTEM_PROMPT = """
        # Role and Purpose
        You are an AI assistant that extract user's query and check if a user's query has enough context to query a database. 
        Your task is to ensure that the query contains all necessary information based on specific rules 
        and guidelines provided below.

        # Guidelines:
        1. If the user asks in Thai language, please respond in Thai.
        2. If the user asks in English, please respond in English.

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
        * Remark that if user type international program (นานาชาติ) only by not providing Program or Program Type, 
        please assume to use program and program type as international or นานาชาติ to both.
        
        # Rules
        1. If the user does not provide a **major**, ask the user to provide a major first.
        2. If the user does not provide a **round**, ask the user to provide a round first.
        3. If the user does not provide a **program**:
            - If it's round 3, assume the program is Admission.
            - For other rounds, ask the user to provide a program.
        4. If the user does not provide a **program type**, ask the user to provide the program type first.
        5. User DOES NOT NEED to input Condtion (เงื่อนไขขั้นต่ำ) and Criteria (เกณฑ์การพิจารณา).

        Your response should clearly indicate if the query is complete or if additional information is needed. If additional information is required, specify exactly what the user is missing and ask them to provide it.  
        And if user put question that hard to extract what it is, please tell user to put in format like
        "โปรดให้ข้อมูลตามรูปแบบ สาขาวิชา, ภาค(ปกติ, พิเศษ, นานาชาติ, ภาษาอังกฤษ), รอบการคัดเลือก, และโครงการ เช่นตัวอย่าง
        สาขาวิชา: วศ.บ. สาขาวิชาวิศวกรรมเครื่องกล (ภาษาไทย ปกติ) รอบ 1/2 ช้างเผือก"

        Additionally, extract and return the following fields from the user's query:
        1. **Major** (สาขาวิชา/สาขา)
        2. **Round** (รอบการคัดเลือก/รอบ)
        3. **Program** (โครงการ)
        4. **Program type** (ภาค)
        5. If Major, Round, Program, Program type are missing, provide the specific feedback about what the user should add.

        Ensure that you mention which information is missing and what the user needs to add to complete the query.
        """

    @staticmethod
    def extract(text, template) -> Tuple[List[str], str, int, str, str]:
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
                {"role": "system", "content": QuestionExtraction.SYSTEM_PROMPT},
                {"role": "user", "content": "Extract: " + text},
            ],
            response_model=template,
        )
        thought_process = llm_response.thought_process
        major = llm_response.major
        round_ = llm_response.round
        program = llm_response.program
        program_type = llm_response.program_type

        # Check completeness
        missing_fields = []
        if not major:
            missing_fields.append("Major (สาขาวิชา)")
        if not round_:
            missing_fields.append("Round (รอบการคัดเลือก)")
        if not program and round_ != 3:  # Assume round 3 defaults to "Admission"
            missing_fields.append("Program (โครงการ)")
        if not program_type:
            missing_fields.append("Program Type (ภาค)")

        is_complete = len(missing_fields) == 0
        
        return thought_process, major, round_, program, program_type, is_complete, missing_fields
