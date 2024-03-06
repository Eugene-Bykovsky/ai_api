from typing import List, Optional
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class GenerationRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 50
