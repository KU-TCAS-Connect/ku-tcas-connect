from pydantic import BaseModel, Field
from typing import Literal
from services.llm_factory import LLMFactory

class QueryClassificationResponse(BaseModel):
    intent: Literal["general_info", "admission_criteria", "count_criteria", "not_related"]

class QueryClassification:
    SYSTEM_PROMPT = """
    You are an AI assistant that classifies user questions into three categories:

    ### Categories:  
    1. "general_info" - for questions about KU TCAS processes, schedules, general admission information, or general information about Kasetsart University.
    - Example: "กำหนดการรับสมัครของคณะวิศวกรรมซอฟต์แวร์ รอบ1/1 โครงการนานาชาติเป็นอย่างไร"
    - Example: "คุณสมบัติผู้สมัครเบื้องต้นของรอบ1 โครงการนักกีฬาทีมชาติคืออะไร"

    2. "admission_criteria" - for questions about admission criteria, eligibility, or required documents.
    - Example: "ถ้าจะสมัครวิศวกรรมศาสตร์ต้องใช้คะแนนอะไรบ้าง?"
    - Example: "เกณฑ์การรับสมัครรอบที่ 3 เป็นอย่างไร?"
    - Example: "เกณฑ์ในการสอบเข้าวิศวะซอฟต์แวร์รอบที่1 มีทั้งหมดกี่โครงการ"

    3. "count_criteria" - for questions asking about the number of admission criteria.
    - Example: "วิศวะเครื่องกล รอบ2 มีกี่เกณฑ์?"
    - Example: "รอบที่ 3 ของวิศวะซอฟต์แวร์มีเกณฑ์ทั้งหมดกี่ข้อ?"
    - Example: "มีเงื่อนไขกี่ข้อสำหรับรอบ1 สาขาวิศวกรรมโยธา?"
    
    4. "not_related" - for questions that are not related to KU TCAS, not related to admission, and not related to university-related topics.
    - Example: "อากาศวันนี้เป็นยังไง?"
    - Example: "แนะนำร้านกาแฟหน่อย"

    ### Response Format:
    Respond must be a JSON object containing only an "intent" field.
    """

    @staticmethod
    def classify(question: str) -> QueryClassificationResponse:
        """Classifies the user's question into one of the three categories."""
        
        # Construct message history with system prompt and user question
        messages = [
            {"role": "system", "content": QueryClassification.SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        # Call the LLM using a helper factory
        llm = LLMFactory("openai")
        response = llm.create_completion(
            response_model=QueryClassificationResponse,
            messages=messages,
        )

        # Return the classified intent
        return response