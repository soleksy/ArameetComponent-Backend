from pydantic import BaseModel
from typing import List, Optional

class Recommendation(BaseModel):	
    title: str
    suggestion: str
    format: Optional[str] = None  

class Meeting(BaseModel):
    title: str
    start_time: str
    end_time: str
    should_be_done_asynchronously: bool
    reason: Optional[str] = None

class CalendarAnalysis(BaseModel):
    calendar_detected: bool
    total_meetings_detected: int
    total_meetings_to_be_done_asynchronously: int
    meetings: List[Meeting] = []
    total_meeting_hours: Optional[float] = 0.0
    potential_savings_hours: Optional[float] = 0.0
    recommendations: List[Recommendation] = []
