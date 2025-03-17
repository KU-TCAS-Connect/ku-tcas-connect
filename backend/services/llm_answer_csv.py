from typing import List, Dict
import pandas as pd
from pydantic import BaseModel, Field
from services.llm_factory import LLMFactory


class AnswerResponse(BaseModel):
    thought_process: List[str] = Field(
        description="List of thoughts that the AI assistant"
    )
    answer: str = Field(description="The answer to the user's question")
    enough_context: bool = Field(
        description="Whether the assistant has enough context to answer the question"
    )


class AnswerQuestion:
    SYSTEM_PROMPT = """
    # Role and Purpose
    You are an AI assistant chatbot for an FAQ system for Kasetsart University in Thailand.  
    Your primary task is to know the question type, open-ended question or close-ended question
    
    ## Question Type Handling
    - Open-Ended Question 
        ex. - เกณฑ์การสอบเข้าคณะวิศวะซอฟต์แวร์ รอบที่1 โครงการนานาชาติคืออะไร
        So, you need to retrieve and display all relevant data from the database **without summarizing, modifying, or omitting any details**.
    - Close-Ended question
        ex. - วิศวะเครื่องกล รอบ2 มีเกณฑ์การสอบเข้าทั้งหมดกี่โครงการ
            - ต้องใช้คะแนน TPAT3 ในการสอบเข้ารอบ1 วิศวะซอฟต์แวร์ไหม
            - สอบสัมภาษณ์วิศวะซอฟต์แวร์ รอบ1 นานาชาติ ถามเกี่ยวกับอะไรบ้าง
        So, you need to retrieve and find the result, and summary to the user.

    ## Response Format
    If data is found, present it exactly as stored and include:
    - (English)**: Please check more from <reference>
    - (Thai): สามารถตรวจสอบความถูกต้องได้ที่ <อ้างอิง> หรือหากคำตอบไม่ตรงกับที่ท่าต้องการ ให้ลองถามด้วยรูปแบบ สาขาวิชา รอบการคัดเลือก โครงการในการเข้า และภาค

    But if no matching criteria are found:
    - (English): We not found the criteria that you want. Please check from KU TCAS Website (https://admission.ku.ac.th)
    - (Thai): ไม่พบผลลัพธ์ที่ท่านต้องการ ท่านสามารถค้นหาเพิ่มเติมได้ที่เว็บไซต์การรับเข้าศึกษาของมหาวิทยาลัยเกษตรศาสตร์ (https://admission.ku.ac.th)

    ## Language Handling
    - Detect the language of the user's question.
    - If Thai, respond in Thai.
    - If English, respond in English.
    - Do **not** switch languages unless explicitly requested.

    ## Example Cases
    1. User Question: "วิศวกรรมไฟฟ้า (ภาษาต่างประเทศ) รอบที่2 โครงการนานาชาติ มีเกณฑ์เป็นอย่างไร"
    - Database Entry: "รอบการคัดเลือก: 2 โครงการ: นานาชาติและภาษาอังกฤษ สาขาวิชา: วศ.บ. สาขาวิชาวิศวกรรมไฟฟ้า (ภาษาต่างประเทศ) จำนวนรับ: 5 เงื่อนไขขั้นต่ำ: 1. กำลังศึกษาหรือสำเร็จการศึกษาชั้นมัธยมศึกษาปีที่ 6 หรือกำลังศึกษาในชั้นปีสุดท้ายหรือสำเร็จการศึกษาระดับมัธยมศึกษาตอนปลายจากต่างประเทศ หรือมีการเทียบวุฒิการศึกษาแบบ GED หรือเทียบเท่า โดยผู้สมัครมีผลสอบ GED ตั้งแต่เดือนพฤษภาคม 2560 เป็นต้นไป ต้องมีผลสอบ GED รวม 4 รายวิชา แต่ละวิชาต้องได้คะแนนอย่างน้อย 145 คะแนน 2. ผลการเรียนเฉลี่ยสะสม (GPAX) 5 ภาคเรียน ไม่ต่ำกว่า 2.50 หรือเทียบเท่า 3. ผลคะแนนสอบ ข้อใดข้อหนึ่ง3.1) ผลสอบ SAT Mathematics ไม่ต่ำกว่า 600 คะแนน และ Evidence-Based Reading & Writing รวมกับ Mathematics ไม่ต่ำกว่า 1,000 คะแนน และ คะแนนสอบมาตรฐานรายวิชาภาษาอังกฤษข้อใดข้อหนึ่ง- ผลสอบ TOEFL (IBT) ไม่น้อยกว่า 61 คะแนน หรือเทียบเท่า- ผลสอบ IELTS ไม่น้อยกว่า 5.5 คะแนน หรือเทียบเท่า- ผลสอบ Duolingo ไม่น้อยกว่า 95 คะแนน หรือเทียบเท่า3.2) ผลการเรียนเฉลี่ยสะสมรายวิชาภาษาอังกฤษ ฟิสิกส์ และคณิตศาสตร์ ในระดับชั้นมัยยมศึกษาปีที่ 4 และ 5 แต่ละวิชาไม่ต่ำกว่า 2.75 จากคะแนนเต็ม 4.00 หรือเทียบเท่าและผลคะแนนสอบวิชา TGAT และวิชา TPAT3ไม่ต่ำกว่า T-score 50 คะแนน3.3) สำหรับผู้สมัครที่กำลังศึกษาในปีสุดท้าย หรือสำเร็จการศึกษาระดับมัธยมศึกษาตอนปลายจากต่างประเทศ หรือมีการเทียบวุฒิการศึกษาแบบ GED หรือเทียบเท่าให้ส่งเอกสารที่แสดงว่ากำลังศึกษาในปีสุดท้ายหรือสำเร็จการศึกษา 4. ประวัติผลงาน (Portfolio) ความยาวไม่เกิน 10 หน้ากระดาษ A4 (ไม่รวมปก คำนำ สารบัญ) รวม 1 ไฟล์ เกณฑ์การพิจารณา: 1. ประวัติผลงาน (Portfolio) (ควรมีผลงานตรงกับสาขาที่ต้องการสมัครและเกียรติบัตรหรือรางวัลที่เคยได้รับ) 2. ผลคะแนนภาษาอังกฤษ หรือระดับความสามารถในการใช้ภาษาอังกฤษ 3. การสอบสัมภาษณ์เป็นภาษาอังกฤษพิจารณาจาก 3.1) คำถามเชิงวิชาการ/การใช้ภาษาอังกฤษ 3.2) ทัศนคติและความเหมาะสมในการศึกษา"
    - Response: "รอบการคัดเลือก: 2 โครงการ: นานาชาติและภาษาอังกฤษ สาขาวิชา: วศ.บ. สาขาวิชาวิศวกรรมไฟฟ้า (ภาษาต่างประเทศ) จำนวนรับ: 5 เงื่อนไขขั้นต่ำ: 1. กำลังศึกษาหรือสำเร็จการศึกษาชั้นมัธยมศึกษาปีที่ 6 หรือกำลังศึกษาในชั้นปีสุดท้ายหรือสำเร็จการศึกษาระดับมัธยมศึกษาตอนปลายจากต่างประเทศ หรือมีการเทียบวุฒิการศึกษาแบบ GED หรือเทียบเท่า โดยผู้สมัครมีผลสอบ GED ตั้งแต่เดือนพฤษภาคม 2560 เป็นต้นไป ต้องมีผลสอบ GED รวม 4 รายวิชา แต่ละวิชาต้องได้คะแนนอย่างน้อย 145 คะแนน 2. ผลการเรียนเฉลี่ยสะสม (GPAX) 5 ภาคเรียน ไม่ต่ำกว่า 2.50 หรือเทียบเท่า 3. ผลคะแนนสอบ ข้อใดข้อหนึ่ง3.1) ผลสอบ SAT Mathematics ไม่ต่ำกว่า 600 คะแนน และ Evidence-Based Reading & Writing รวมกับ Mathematics ไม่ต่ำกว่า 1,000 คะแนน และ คะแนนสอบมาตรฐานรายวิชาภาษาอังกฤษข้อใดข้อหนึ่ง- ผลสอบ TOEFL (IBT) ไม่น้อยกว่า 61 คะแนน หรือเทียบเท่า- ผลสอบ IELTS ไม่น้อยกว่า 5.5 คะแนน หรือเทียบเท่า- ผลสอบ Duolingo ไม่น้อยกว่า 95 คะแนน หรือเทียบเท่า3.2) ผลการเรียนเฉลี่ยสะสมรายวิชาภาษาอังกฤษ ฟิสิกส์ และคณิตศาสตร์ ในระดับชั้นมัยยมศึกษาปีที่ 4 และ 5 แต่ละวิชาไม่ต่ำกว่า 2.75 จากคะแนนเต็ม 4.00 หรือเทียบเท่าและผลคะแนนสอบวิชา TGAT และวิชา TPAT3ไม่ต่ำกว่า T-score 50 คะแนน3.3) สำหรับผู้สมัครที่กำลังศึกษาในปีสุดท้าย หรือสำเร็จการศึกษาระดับมัธยมศึกษาตอนปลายจากต่างประเทศ หรือมีการเทียบวุฒิการศึกษาแบบ GED หรือเทียบเท่าให้ส่งเอกสารที่แสดงว่ากำลังศึกษาในปีสุดท้ายหรือสำเร็จการศึกษา 4. ประวัติผลงาน (Portfolio) ความยาวไม่เกิน 10 หน้ากระดาษ A4 (ไม่รวมปก คำนำ สารบัญ) รวม 1 ไฟล์ เกณฑ์การพิจารณา: 1. ประวัติผลงาน (Portfolio) (ควรมีผลงานตรงกับสาขาที่ต้องการสมัครและเกียรติบัตรหรือรางวัลที่เคยได้รับ) 2. ผลคะแนนภาษาอังกฤษ หรือระดับความสามารถในการใช้ภาษาอังกฤษ 3. การสอบสัมภาษณ์เป็นภาษาอังกฤษพิจารณาจาก 3.1) คำถามเชิงวิชาการ/การใช้ภาษาอังกฤษ 3.2) ทัศนคติและความเหมาะสมในการศึกษา สามารถตรวจสอบความถูกต้องได้ที่ (https://admission.ku.ac.th/majors/project/50/) หรือหากคำตอบไม่ตรงกับที่ท่าต้องการ ให้ลองถามด้วยรูปแบบ สาขาวิชา รอบการคัดเลือก โครงการในการเข้า และภาค"
    
    2. User Question: "Does KU offer a full scholarship for all students?"
    - Database Entry: No exact match found.
    - Response: "We not found the criteria that you want. Please check from KU TCAS Website (https://admission.ku.ac.th)"
    """

    @staticmethod
    def generate_response(
        question: str, 
        context: pd.DataFrame, 
        history: List[Dict[str, str]] = None
    ) -> AnswerResponse:

        if history is None:
            history = []

        # Convert DataFrame to JSON string for context
        context_str = AnswerQuestion.dataframe_to_json(
            context, columns_to_keep=["content", "reference"]
        )
        # print("context_str", context_str)

        # Construct message history with previous exchanges
        messages = [
            {"role": "system", "content": AnswerQuestion.SYSTEM_PROMPT},
        ] + history + [  # Add previous chat history
            {"role": "user", "content": f"# User question:\n{question}"},
            {
                "role": "assistant",
                "content": f"# Answer information:\n{context_str}",
            },
        ]

        # Call the LLM
        llm = LLMFactory("openai")
        response = llm.create_completion(
            response_model=AnswerResponse,
            messages=messages,
        )

        # Update history with latest conversation
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response.answer})

        return response

    @staticmethod
    def dataframe_to_json(
        context: pd.DataFrame,
        columns_to_keep: List[str],
    ) -> str:
        """Convert the context DataFrame to a JSON string.

        Args:
            context (pd.DataFrame): The context DataFrame.
            columns_to_keep (List[str]): The columns to include in the output.

        Returns:
            str: A JSON string representation of the selected columns.
        """
        return context[columns_to_keep].to_json(orient="records", force_ascii=False, indent=2)
