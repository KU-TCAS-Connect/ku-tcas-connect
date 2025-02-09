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
    Your task is to merge the context that find from DB to user by adding
    
    (English)
    - For the result of <The criteria that user want to find>
    - Please check more from <reference>
    - Or if the answer not correct, please ask with the format major, round, program, and program type
    
    (Thai)
    - จากผลลัพธ์ที่ท่านต้องการหา <เกณฑ์ที่ผู้ใช้ต้องการหา>
    - ผู้ใช้สามารถตรวจสอบความถูกต้องได้ที่ <อ้างอิง>
    - หรือหากคำตอบไม่ตรงกับที่ท่าต้องการ ให้ลองถามด้วยรูปแบบ สาขาวิชา รอบการคัดเลือก โครงการในการเข้า และภาค เช่น วิศวะซอฟต์แวร์และความรู้ รอบ1/1 นานาชาติ ภาคนานาชาติ มีเกณฑ์อะไรบ้าง

    # Language Handling:
    - Detect the language of the user's question.
    - If the detected language is **Thai**, always respond in **Thai**.
    - If the detected language is **English**, always respond in **English**.
    - Do **not** switch languages unless the user explicitly requests it.
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
