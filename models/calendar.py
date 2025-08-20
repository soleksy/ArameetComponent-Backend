# models/calendar.py
from pydantic import BaseModel, Field
from typing import List, Optional

# ---------- Final backend response types (what FastAPI will return) ----------

class Recommendation(BaseModel):
    title: str
    suggestion: str

class Meeting(BaseModel):
    title: str
    start_time: str
    end_time: str
    should_be_done_asynchronously: bool

class BackendResponse(BaseModel):
    calendar_detected: bool
    total_meetings_detected: int
    total_meetings_to_be_done_asynchronously: int
    meetings: List[Meeting]
    total_meeting_hours: Optional[float] = None
    potential_savings_hours: Optional[float] = None
    recommendations: List[Recommendation] = Field(default_factory=list)

# ---------- Intermediate pipeline types (used only inside analyzer) ----------

class ExtractedMeeting(BaseModel):
    title: str
    start_time: str
    end_time: str

class ExtractionResult(BaseModel):
    calendar_detected: bool
    meetings: List[ExtractedMeeting] = Field(default_factory=list)

class AsyncGrading(BaseModel):
    should_be_done_asynchronously: List[bool]

class RecommendationsEnvelope(BaseModel):
    recommendations: List[Recommendation] = Field(default_factory=list)