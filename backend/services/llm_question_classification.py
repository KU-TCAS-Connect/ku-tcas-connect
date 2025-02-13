from pydantic import BaseModel, Field
from typing import Literal
from services.llm_factory import LLMFactory

class QueryClassificationResponse(BaseModel):
    intent: Literal["general_info", "admission_criteria", "not_related"]

class QueryClassification:
    SYSTEM_PROMPT = """
    You are an AI assistant that classifies user questions into three categories:
    
    1. "general_info" - for questions about KU TCAS processes, schedules, or general admission information.
    2. "admission_criteria" - for questions about admission criteria, eligibility, or required documents.
    3. "not_related" - for questions that are unrelated to KU TCAS, admission, or university-related topics.

    Respond with a JSON object containing only an "intent" field.
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