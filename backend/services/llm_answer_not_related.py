from typing import List, Dict
import pandas as pd
from pydantic import BaseModel, Field
from services.llm_factory import LLMFactory


class AnswerResponse(BaseModel):
    thought_process: List[str] = Field(
        description="List of thoughts that the AI assistant"
    )
    answer: str = Field(description="The answer to the user's question")


class AnswerQuestion:
    SYSTEM_PROMPT = """
    # Role and Purpose
    You are the ChatGPT to help people to answer the general question. Please help them.
    """

    @staticmethod
    def generate_response(
        question: str, 
        history: List[Dict[str, str]] = None
    ) -> AnswerResponse:

        if history is None:
            history = []

        # Construct message history with previous exchanges
        messages = [
            {"role": "system", "content": AnswerQuestion.SYSTEM_PROMPT},
        ] + history + [  # Add previous chat history
            {"role": "user", "content": f"# User question:\n{question}"},
        ]

        # Call the LLM
        llm = LLMFactory("openai")
        response = llm.create_completion(
            response_model=AnswerResponse,
            messages=messages,
        )

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
