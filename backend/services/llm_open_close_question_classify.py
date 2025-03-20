from pydantic import BaseModel, Field
from typing import Literal
from services.llm_factory import LLMFactory

class QuestionTypeResponse(BaseModel):
    question_type: Literal["closed_ended", "open_ended"]

class QuestionTypeClassifier:
    SYSTEM_PROMPT = """
    You are an AI assistant that classifies user questions into two categories:

    ### Categories:
    1. **"closed_ended"** - Questions that expect a definite or specific answer, such as Yes/No, a number, a name, or a factual detail.  
    - Example: "วันปิดรับสมัครรอบที่ 2 คือเมื่อไหร่?"  
    - Example: "วิศวะเครื่องกล รอบ3 ใช้คะแนน TPAT1 มั้ย?"  
    - Example: "มหาวิทยาลัยเกษตรศาสตร์มีหลักสูตรนานาชาติไหม?"

    2. **"open_ended"** - Questions that require an explanation, opinion, or a more detailed response.  
    - Example: "ความแตกต่างระหว่างหลักสูตรนานาชาติกับหลักสูตรปกติคืออะไร?"  
    - Example: "ฉันควรรู้อะไรก่อนสมัครเข้าคณะวิศวกรรมซอฟต์แวร์?"  
    - Example: "หลักสูตรวิศวกรรมศาสตร์ของมหาวิทยาลัยเกษตรศาสตร์ดีแค่ไหน?"  

    ### Instructions:
    - Categorize each question into **only one** category.  
    - Respond with a **JSON object** containing only the `"question_type"` field.  
    - Always respond in **English**.
    """

    @staticmethod
    def classify(question: str) -> QuestionTypeResponse:
        """Classifies the user's question as 'closed_ended' or 'open_ended'."""
        
        # Construct message history with system prompt and user question
        messages = [
            {"role": "system", "content": QuestionTypeClassifier.SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        # Call the LLM using a helper factory
        llm = LLMFactory("openai")
        response = llm.create_completion(
            response_model=QuestionTypeResponse,
            messages=messages,
        )

        # Return the classified question type
        return response
