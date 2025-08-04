from pydantic import BaseModel
from typing import List, Optional

class Meeting(BaseModel):
    title: str
    start_time: str
    end_time: str
    is_valuable: bool
    reason: Optional[str] = None

class CalendarAnalysis(BaseModel):
    calendar_detected: bool
    meetings: List[Meeting] = []
    total_meeting_hours: Optional[float] = 0.0
    potential_savings_hours: Optional[float] = 0.0
    recommendations: Optional[List[str]] = []
