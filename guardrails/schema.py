from pydantic import BaseModel, Field
from typing import List

class Citation(BaseModel):
    doc_id: str
    chunk_id: int

class AnswerSchema(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: List[Citation] = []
    evidence: List[str] = []  # exact quotes from CONTEXT