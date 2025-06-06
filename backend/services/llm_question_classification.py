from pydantic import BaseModel, Field
from typing import Literal
from services.llm_factory import LLMFactory

class QueryClassificationResponse(BaseModel):
    intent: Literal["general_info", "admission_criteria", "not_related"]

class QueryClassification:
    SYSTEM_PROMPT = """
    You are an AI assistant that classifies user questions into three categories:

    ### Categories:  
    1. "general_info" - for questions about Kasetsart University TCAS processes, schedules, qualification, general admission information, required documents.
    Most common keywords: กำหนดการ คุณสมบัติ วันที่เท่าไหร่ สมัครยังไง คุณสมบัติเฉพาะ
    - Example: "กำหนดการรับสมัครของคณะวิศวกรรมซอฟต์แวร์ รอบ1/1 โครงการนานาชาติเป็นอย่างไร"
    - Example: "คุณสมบัติผู้สมัครเฉพาะถ้าอยากเข้าโครงการนักกีฬาทีมชาติคืออะไร"
    - Example: "รอบ2 MOU ต้องมีคุณสมบัติอะไรบ้าง"
    - Example: "เว็บไซด์ที่ประกาศผลของเกษตรชื่ออะไร"
    
    2. "admission_criteria" - for questions about admission criteria.
    Most common keywords: เกณฑ์ คะแนน พอร์ต รับกี่คน
    - Example: "ถ้าจะสมัครวิศวกรรมศาสตร์ต้องใช้คะแนนอะไรบ้าง?"
    - Example: "เกณฑ์การรับสมัครรอบที่ 3 เป็นอย่างไร?"
    - Example: "เกณฑ์ในการสอบเข้าวิศวะซอฟต์แวร์รอบที่1 มีทั้งหมดกี่โครงการ"

    3. "not_related" - for other questions that not in 2 category which are not related to Kasetsart University TCAS, not related to admission, and not related to university-related topics.
    - Example: "อากาศวันนี้เป็นยังไง?"
    - Example: "แนะนำร้านกาแฟหน่อย"
    - Example: "อุณหภูมิตอนนี้"    
    - Example: "วันนี้วันที่เท่าไหร่"

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