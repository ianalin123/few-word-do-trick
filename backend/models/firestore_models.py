from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class EmotionData(BaseModel):
    label: str
    confidence: Optional[float] = None


class Message(BaseModel):
    id: str
    conversation_id: str
    speaker: str  # "USER" or "OTHER"
    text: str
    timestamp: datetime
    emotion: Optional[EmotionData] = None


class MessageCreate(BaseModel):
    speaker: str
    text: str
    timestamp: datetime
    emotion: Optional[EmotionData] = None


class Conversation(BaseModel):
    id: str
    user_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    status: str = "active"  # "active" or "ended"
