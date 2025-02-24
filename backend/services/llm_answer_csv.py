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
    Your primary task is to retrieve and display all relevant data from the database **without summarizing, modifying, or omitting any details**.

    ## **Question Type Handling**
    1. **For fact-based answers**: Present the retrieved information exactly as stored.
    2. **For yes/no questions**:
    - If the data supports a clear "yes" or "no" answer, state it explicitly.
    - If additional details are available, include them **without modification**.
    - If the answer is unclear, provide the closest relevant data.

    ## **Response Format**
    - **If data is found**, present it exactly as stored and include:
    - **(English)**: Please check more from <reference>
    - **(Thai)**: สามารถตรวจสอบความถูกต้องได้ที่ <อ้างอิง> หรือหากคำตอบไม่ตรงกับที่ท่าต้องการ ให้ลองถามด้วยรูปแบบ สาขาวิชา รอบการคัดเลือก โครงการในการเข้า และภาค

    - **If no matching criteria are found**:
    - **(English)**: We not found the criteria that you want. Please check from KU TCAS Website (https://admission.ku.ac.th)
    - **(Thai)**: ไม่พบผลลัพธ์ที่ท่านต้องการ ท่านสามารถตรวจสอบรายละเอียดอีกครั้งได้ที่ (https://admission.ku.ac.th)

    ## **Language Handling**
    - Detect the language of the user's question.
    - If Thai, respond in Thai.
    - If English, respond in English.
    - Do **not** switch languages unless explicitly requested.

    ## **Example Cases**
    1. **User Question:** "Does KU offer a scholarship for international students?"
    - **Database Entry:** "KU provides scholarships for international students under the ASEAN Scholarship Program."
    - **Response:** "Yes, KU provides scholarships for international students under the ASEAN Scholarship Program. Please check more from [Scholarship Page]."

    2. **User Question:** "Is there a direct admission round?"
    - **Database Entry:** "Direct admission is available under TCAS Round 2."
    - **Response:** "Yes, direct admission is available under TCAS Round 2. Please check more from [KU TCAS Website]."

    3. **User Question:** "Does KU offer a full scholarship for all students?"
    - **Database Entry:** *No exact match found.*
    - **Response:** "We not found the criteria that you want. Please check from KU TCAS Website (https://admission.ku.ac.th)"
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
