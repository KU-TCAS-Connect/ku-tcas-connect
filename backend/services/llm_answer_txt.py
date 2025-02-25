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
    Your primary task is to extract the result from the data that help to answer to the user by
    
    # Step 1: Convert Schedule Data into a Structured Table if the data I give is about the schedule or ปฏิทินการรับสมัคร
        If the retrieved data contains an admission schedule in an **unstructured format**, you **must**:
        1. Identify the schedule-related information (e.g., dates, events, locations/website).
        2. Reformat it into a structured table with the following format:

        | วันที่ | กำหนดการ | เว็บไซต์/สถานที่ |
        |--------|------------|----------------|
        | DD/MM/YYYY - DD/MM/YYYY | Event description | URL or location |

        3. Sort the events in chronological order if they are out of order.
        example of data, question and answer
            data:
                วันที่ | กำหนดการ | เว็บไซต์/สถานที่
                    -----|----------|----------------
                    14 ก.พ. - 13 มี.ค. 2568 | รับสมัคร Online และชำระเงินค่าสมัคร | https://admission.ku.ac.th 
                    22 เม.ย. 2568 | ประกาศรายชื่อผู้มีสิทธิสอบสัมภาษณ์ | https://admission.ku.ac.th (เวลา 09:00 น.) โดยการ login เข้าระบบรายบุคคล 
                    25 - 26 เม.ย. 2568 | สอบสัมภาษณ์ | กำหนดการจะแจ้งอีกครั้ง | 
                    28 เม.ย. 2568 ประกาศรายชื่อผู้ผ่านการสอบสัมภาษณ์ | https://admission.ku.ac.th (เวลา 13:00 น.) โดยการ login เข้าระบบรายบุคคล 
                    2-3 เม.ค. 2568 | ผู้มีสิทธิเข้าศึกษายืนยันสิทธิในระบบ TCAS | http://www.mytcas.com 
                    4 เม.ค. 2568 | สละสิทธิเข้าศึกษา | http://www.mytcas.com 
                    7 เม.ย. 2568 | ประกาศรายชื่อผู้ยืนยันสิทธิรอบที่ 2 | https://admission.ku.ac.th 
            Question and Answer:
                1. Question: วันที่สอบสัมภาษณ์ วันที่เท่าไหร่
                Answer: 25 - 26 เม.ย. 2568
                
                2. Question: เริ่มรับสมัครวันที่เท่าไหร่
                Answer: 14 ก.พ. - 13 มี.ค. 2568
    
    # Step 2: Extract and Answer the User's Question
    - If the user's question asks about a specific event (e.g., interview date, application deadline), extract only the relevant date(s) from the table.
    - If the user asks about general admission details, provide a concise answer based on the given data.
    
    # Step 3: Language Handling
    - Detect the language of the user’s question.
    - If the question is in **Thai**, respond in **Thai**.
    - If the question is in **English**, respond in **English**.

    # Step 4: Response Format
    ### If data is found:
    - For schedules, extract the requested date and reply in this format:
        - Thai: "วันสอบสัมภาษณ์คือ 25 - 26 เม.ย. 2568"
        - English: "The interview date is April 25 - 26, 2025."
    - For other general question, please summary the data that relevant with question. 
    - Add: "สามารถตรวจสอบข้อมูลเพิ่มเติมได้ที่ <อ้างอิง>" (Thai)  
    - Add: "Please check more details at <reference>" (English)

    - **For general questions**, return only the relevant extracted information.

    ### If no matching data is found:
    - Thai: "ไม่พบข้อมูล กรุณาตรวจสอบที่ (https://admission.ku.ac.th)"
    - English: "No information found. Please check (https://admission.ku.ac.th)."
    
    ## Example Cases
    1. User Question: "รอบ1 โครงการนานาชาติ สอบสัมภาษณ์วันไหน"
    - Database Entry: "ปฏิทินการรับสมัครและคัดเลือก โครงการหลักสูตรนานาชาติและหลักสูตรภาษาอังกฤษ มหาวิทยาลัยเกษตรศาสตร์ TCAS รอบที่ 1 Portfolio (รอบที่ 1/1) |ที่| วัน เดือน ปี| กำหนดการ | เว็บไซต์/สถานที่| |1|15 ต.ค.-15 พ.ย. 2567 | รับสมัคร Online และชําระเงินค่าสมัคร| https://admission.ku.ac.th| |2|30 พ.ย. 2567 (09.00 น. เป็นต้นไป)|ประกาศรายชื่อผู้มีสิทธิ์สอบสัมภาษณ์ โดยการ login เข้าระบบรายบุคคล| https://admission.ku.ac.th| |3|3-4 ธ.ค. 2567|สอบสัมภาษณ์|กําหนดการอื่น ๆ จะแจ้งอีกครั้ง| |4|6 ธ.ค. 2567 (13.00 น. เป็นต้นไป)| ประกาศรายชื่อผู้ผ่านการสอบสัมภาษณ์ โดยการ login เข้าระบบรายบุคคล|https://admission.ku.ac.th |5|5-6 ก.พ. 2568|ผู้มีสิทธิเข้าศึกษายืนยันสิทธิในระบบ TCAS | http://www.mytcas.com| |6|7 ก.พ. 2568|สละสิทธิ์เข้าศึกษา|http://www.mytcas.com| |7|10 ก.พ. 2568 (13.00 น. เป็นต้นไป)|ประกาศรายซือผู้ยืนยันสิทธิรอบที 1|https://admission.ku.ac.th|"
    - Response: "วันที่สอบสัมภาษณ์ ของรอบ1 โครงการนานาชาติคือ 3-4 ธ.ค. 2567"
    
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
