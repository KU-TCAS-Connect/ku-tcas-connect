from typing import List, Dict
from pydantic import BaseModel, Field
from services.llm_factory import LLMFactory

class FilteredDocumentResponse(BaseModel):
    """A model to represent the filtered documents and their associated details."""
    idx: List[int] = Field(
        description="List of indices of the documents that were kept", examples=[1, 2, 3, 4, 5, 6, 7]
    )
    content: List[str] = Field(
        description=" List of full content of the documents that were kept"
    )
    reject_reasons: List[str] = Field(
        description="List of thoughts and reasons why documents were filtered out or kept"
    )
    
class RetrieveFilter:
    """Utility class for filtering and retaining only the documents relevant to a user's query."""

    SYSTEM_PROMPT = """
    # Role and Purpose
    คุณเป็น AI ผู้ช่วยที่ทำหน้าที่คัดกรองผลลัพธ์ที่ดึงมาจากฐานข้อมูลเวกเตอร์ 
    หน้าที่ของคุณคือการวิเคราะห์ผลลัพธ์ที่ได้รับ และเก็บเฉพาะเอกสารที่เกี่ยวข้องโดยตรงกับคำถามของผู้ใช้ 
    กรองเอกสารที่ไม่ตรงประเด็นออก เพื่อให้แน่ใจว่าข้อมูลที่ให้ไปนั้นถูกต้องและมีประโยชน์มากที่สุด
    
    # กฎการคัดกรอง
    - ตอบกลับทุกอย่างเป็นภาษาไทย
    - สาขาวิชาวิศวกรรมศาสตร์ในคำถามของผู้ใช้และเอกสารที่ดึงมาจะต้องตรงกัน
    - เนื้อหาโดยรวมของเอกสารที่ดึงมาจะต้องตรงกับคำถามของผู้ใช้
    - ห้ามเดาหรือเลือกเอกสารที่ดูเหมือนจะเกี่ยวข้องแบบสุ่ม

    # ข้อยกเว้น
    - หากคำถามที่มีภาคไม่ตรงกับเอกสารที่ดึงมาไม่ต้องคัดเอกสารนั้นออก
    """

    @staticmethod
    def filter(query: str, documents: List[str]) -> FilteredDocumentResponse:
        """Filters the retrieved documents to keep only those relevant to the user's query.

        Args:
            query: The user's query.
            documents: The list of documents retrieved from the database.

        Returns:
            A FilteredDocumentResponse instance with details about kept or filtered-out documents.
        """
        if not documents:
            return FilteredDocumentResponse(idx=[], content=[], reasons=[])

        document_list = "\n".join([f"เอกสารอันที่ {idx + 1}\n{doc}" for idx, doc in enumerate(documents)])

        user_message = f"""
จากเอกสารที่ดึงมา กรุณาคัดกรองและเก็บเฉพาะเอกสารที่เกี่ยวข้องกับคำถามของผู้ใช้:
คำถามของผู้ใช้: "{query}"

เอกสารที่ดึงมา:
{document_list}
"""

        print("Send this User Message to LLM:", user_message)        
        messages = [
            {"role": "system", "content": RetrieveFilter.SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        llm = LLMFactory("openai")
        response = llm.create_completion(
            messages=messages,
            response_model=FilteredDocumentResponse,
        )
        return response
