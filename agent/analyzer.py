# agent/analyzer.py
import base64
import os
from datetime import datetime
from typing import List, Tuple

import dotenv
from openai import OpenAI

from models.calendar import (
    BackendResponse,
    ExtractionResult,
    ExtractedMeeting,
    AsyncGrading,
    RecommendationsEnvelope,
    Meeting,
)

import re
from datetime import datetime, time as dtime, date as ddate, timezone, timedelta

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
EXTRACT_MODEL = os.getenv("ARAMEET_MODEL_EXTRACT", "gpt-4o") 

# ----------------------- Utils -----------------------

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def _parse_iso(dt: str):
    try:
        return datetime.fromisoformat(dt)
    except Exception:
        return None

def _duration_hours(start_iso: str, end_iso: str) -> float:
    s = _parse_iso(start_iso)
    e = _parse_iso(end_iso)
    if not s or not e:
        return 0.0
    delta = (e - s).total_seconds()
    return max(0.0, round(delta / 3600.0, 3))

def _aggregate_hours(meetings: List[Meeting]) -> Tuple[float, float]:
    total = 0.0
    savings = 0.0
    for m in meetings:
        d = _duration_hours(m.start_time, m.end_time)
        total += d
        if m.should_be_done_asynchronously:
            savings += d
    return round(total, 3), round(savings, 3)


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_TIME_24H_RE = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*$", re.I)
_TIME_12H_RE = re.compile(r"^\s*\d{1,2}:\d{2}\s*(am|pm)\s*$", re.I)

def _normalize_dt_string(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if _ISO_DATE_RE.match(s) and " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return s

def _parse_datetime_loose(s: str) -> datetime | None:
    if not s:
        return None
    s = _normalize_dt_string(s)

    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass

    if _TIME_24H_RE.match(s):
        try:
            t = dtime.fromisoformat(s)
            return datetime.combine(ddate.today(), t)
        except Exception:
            pass

    if _TIME_12H_RE.match(s):
        try:
            t = datetime.strptime(s.strip().lower(), "%I:%M %p").time()
            return datetime.combine(ddate.today(), t)
        except Exception:
            pass

    try:
        return datetime.strptime(s, "%Y/%m/%dT%H:%M")
    except Exception:
        pass

    return None

def _duration_hours(start_iso: str, end_iso: str) -> float:
    s = _parse_datetime_loose(start_iso)
    e = _parse_datetime_loose(end_iso)

    # If either failed, return 0 (keeps behavior but now succeeds more often)
    if not s or not e:
        return 0.0

    # If model accidentally swapped, fix it
    if e < s:
        s, e = e, s

    delta = (e - s).total_seconds()
    return max(0.0, round(delta / 3600.0, 3))

def _normalize_extracted(meetings: list[ExtractedMeeting]) -> list[ExtractedMeeting]:
    out = []
    for m in meetings:
        start = (m.start_time or "").strip()
        end = (m.end_time or "").strip()

        # Accept "30m" / "45 minutes" if the model emitted a duration
        dur_min = None
        for pat in (r"^(\d+)\s*m(in(ute)?s?)?$", r"^(\d+)\s*minutes?$"):
            m2 = re.match(pat, end, re.I)
            if m2:
                dur_min = int(m2.group(1))
                break
        # If only start time is present + duration, fabricate an end
        if dur_min is not None and start:
            sd = _parse_datetime_loose(start)
            if sd:
                end = (sd + timedelta(minutes=dur_min)).isoformat()

        # Normalize trailing Z / space separators etc.
        start = _normalize_dt_string(start)
        end = _normalize_dt_string(end)

        out.append(ExtractedMeeting(title=m.title, start_time=start, end_time=end))
    return out

# ---------------- Stage 1: Extract meetings (image only) ----------------

def _extract_meetings(image_path: str) -> ExtractionResult:
    """
    Uses EXTRACT_MODEL to: detect calendar + extract (title, start_time, end_time)
    No grading here.
    """
    base64_image = encode_image(image_path)
    completion = client.chat.completions.parse(
        model=EXTRACT_MODEL,
        response_format=ExtractionResult,
        messages=[
            {
                "role": "system",
                "content": (
                    """You extract meetings from calendar screenshots.\n
                    EXTRACT ALL TEXT FROM THE IMAGE\n
                    Return ONLY JSON matching the schema. For each meeting:\n
                    - start_time and end_time MUST be full RFC3339 datetimes (example: 2025-08-19T13:30:00+00:00).\n
                    - If the calendar shows only times, assume today's date and still return full RFC3339.\n
                    - DO NOT return duration strings like '30m' or natural language.\n
                    If start and end hours are missing, estimate and assume < 45 minutes).\n
                    If it's not a calendar, set calendar_detected=false and return meetings=[]."""
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract meetings from this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        'detail': "high"
                    },
                ],
            },
        ],
    )
    return completion.choices[0].message.parsed

# ----------- Stage 2: Grade meetings (should be async?) -----------------

def _grade_meetings_async(extracted: List[ExtractedMeeting]) -> List[bool]:
    """
    Uses gpt-4o-mini to assign should_be_done_asynchronously for each meeting.
    Rule: TRUE unless ANY of the negative criteria match (then FALSE).
    """
    # Prepare compact meeting list for the model
    as_plain = [
        {"title": m.title, "start_time": m.start_time, "end_time": m.end_time}
        for m in extracted
    ]

    grading = client.chat.completions.parse(
        model="gpt-4o",
        response_format=AsyncGrading,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are grading meetings for asynchronous suitability.\n"
                    "Return an object with 'should_be_done_asynchronously': boolean[] aligned to the input order.\n"
                    "Set a meeting's value to FALSE (i.e., should NOT be async) if ANY of these negative criteria are met:\n"
                    "1) Duration > 3 hours.\n"
                    "2) Non-meeting/personal blocks (gym, focus/deep work, hairdresser/barber, doctor/dentist, lunch/breakfast/dinner/coffee,\n"
                    "   kids/school pick-up, PTO/OOO/vacation/leave, travel/trip).\n"
                    "3) External/sales/PR/investor context (sales, discovery, demo call, prospect, client/customer, QBR, renewal,\n"
                    "   success, account, vendor, partner, agency, press/media, PR, investor, fundraising, due diligence).\n"
                    "4) Public events (webinar, conference, meetup, podcast, event, community).\n"
                    "5) Recruiting (interview, screening) unless explicitly a 'debrief'.\n"
                    "6) Offline room hints (room/meeting room/boardroom, office).\n"
                    "7) Signal collision → negative wins (e.g., 'Sprint review with Client X', 'Product demo with Contoso').\n"
                    "8) **Interactive working sessions (require real-time coordination):** hands-on session, workshop, lab, live pairing,\n"
                    "   onboarding/training sessions, troubleshooting sessions.\n"
                    "9) Busy"
                    "Otherwise set to TRUE (i.e., should be async)."
                ),
            },
            {
                "role": "user",
                "content": f"Meetings (ordered): {as_plain}",
            },
        ],
    ).choices[0].message.parsed

    return grading.should_be_done_asynchronously

# ---------------- Stage 3: Recommendations (uses gpt-4o) ----------------

def _make_recommendations(final_meetings: List[Meeting]) -> List[dict]:
    """
    Uses gpt-4o to produce practical, action-oriented recommendations.
    Each recommendation must include at least one concrete example referencing
    an actual meeting title from `final_meetings`.
    """
    mini_view = [
        {
            "title": m.title,
            "start_time": m.start_time,
            "end_time": m.end_time,
            "should_be_done_asynchronously": m.should_be_done_asynchronously,
        }
        for m in final_meetings
    ]

    recs = client.chat.completions.parse(
        model="gpt-4o",
        response_format=RecommendationsEnvelope,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Ara's assistant, the product to help switch from synchronous meetings to async ones."
                    "Suggest how to reduce live meetings by converting suitable ones to async workflows. You should only suggest async video recordings with threads. Since its the Ara's current functionality.\n"
                    "Return `recommendations: Recommendation[]` where each item has {title, suggestion}.\n"
                    "Rules:\n"
                    "- Use the user's meetings to anchor every suggestion: mention at least one example meeting title that matches. If there are more use plural form and return one recommendation.\n"
                    "- Don't invent meetings not present in input."
                    "- Keep suggestions concise and actionable. AND ALWAYS SUGGEST ASYNC VIDEO RECORDINGS sometimes with threads."
                    "Example: Convert the 'Weekly Szymon-Magda' meeting to an async video update with threads."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Based on these meetings (with async flags), produce recommendations."
                    f"Meetings: {mini_view}"
                ),
            },
        ],
    ).choices[0].message.parsed

    return [r.dict() for r in recs.recommendations]

# ---------------- Public API: analyze_calendar_image --------------------

def analyze_calendar_image(image_path: str) -> BackendResponse:
    """
    Full pipeline:
      1) Extract meetings (o4-mini)
      2) Grade async suitability (gpt-4o-mini)
      3) Create recommendations (gpt-4o)
      4) Compute totals and savings
      5) Return BackendResponse
    """
    extraction = _extract_meetings(image_path)
    extracted = _normalize_extracted(extraction.meetings)
    if not extraction.calendar_detected:
        return BackendResponse(
            calendar_detected=False,
            total_meetings_detected=0,
            total_meetings_to_be_done_asynchronously=0,
            meetings=[],
            recommendations=[],
        )

    # Stage 2: grade
    decisions = _grade_meetings_async(extracted) or []
    decisions += [False] * max(0, len(extraction.meetings) - len(decisions))
    # Align lengths defensively
    n = min(len(extraction.meetings), len(decisions))
    final_meetings = [
    Meeting(
        title=em.title,
        start_time=em.start_time,
        end_time=em.end_time,
        should_be_done_asynchronously=decisions[i],
    )
        for i, em in enumerate(extraction.meetings)
    ]

    # Totals
    total_hours, savings_hours = _aggregate_hours(final_meetings)
    total_detected = len(final_meetings)
    total_async = sum(1 for m in final_meetings if m.should_be_done_asynchronously)

    # Stage 3: recommendations
    recs = _make_recommendations(final_meetings)

    return BackendResponse(
        calendar_detected=True,
        total_meetings_detected=total_detected,
        total_meetings_to_be_done_asynchronously=total_async,
        meetings=final_meetings,
        total_meeting_hours=total_hours,
        potential_savings_hours=savings_hours,
        recommendations=recs,
    )

# ------------- CLI smoke test -------------
if __name__ == "__main__":
    img = "uploads/calendar.png"
    out = analyze_calendar_image(img)
    # Print nicely
    try:
        # Pydantic v2
        print(out.model_dump_json(indent=2, ensure_ascii=False))
    except Exception:
        # Fallback
        import json
        print(json.dumps(out.dict(), indent=2, ensure_ascii=False))
