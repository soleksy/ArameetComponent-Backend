import base64
from openai import OpenAI
from models.calendar import CalendarAnalysis
import os
import dotenv

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_calendar_image(image_path: str) -> CalendarAnalysis:
    base64_image = encode_image(image_path)

    completion = client.chat.completions.parse(
        model="gpt-o4-mini",
        response_format=CalendarAnalysis,
        messages=[
            {
                "role": "system",
                "content": (
                    '''You are a Calendar Analysis Assistant for Arameet, a platform that enables asynchronous meetings with video recordings and AI-powered highlights. The user will provide an image that *may* be a calendar.

                        Your tasks:
                        1. Detect if the image is a calendar:
                        - If not, respond only with: `calendar_detected: false`
                        - Do not include any meetings or analysis when false.

                        2. If calendar_detected is true:
                        - Extract ALL MEETINGS visible in the calendar.
                        - For each meeting, YOU MUST include:
                            - `title`
                            - `start_time` and `end_time` in ISO 8601 (e.g. `"2025-08-04T14:00:00"`)
                            - If times are missing (e.g. “bars” on a grid), *estimate* them based on relative bar height compared to events with explicit times—justify your estimate.
                            - `should_be_done_asynchronously`: Boolean indicating whether this meeting type is suitable for conversion to an asynchronous video thread on Arameet. REMEMBER to be REASONABLE here, most meetings must be done synchronously.
                            - `reason`: Brief explanation why it's considered to be possible to convert this meeting to async or not.

                        3. Time analysis:
                        - Summarize `total_meeting_hours`.
                        - Compute `potential_savings_hours` from meetings marked valuable.
                        - **Return `recommendations` as a list of objects**:

                        {
                            "title": "<concise meeting name capitalized first letter>",
                            "suggestion": "<short imperative sentence (≤16 words)>",
                            "format": "recording" | "thread" | "checklist" | null
                        }
                        IN SUGGESTION ALWAYS INCLUDE ONE EXAMPLE OF THE USERS MEETING THAT MATCHES THE RECOMMENDATION.
                        eg. "Convert daily standups like "HERE_MEETING_NAME" to async thread with video highlights."

                        4. Recommended categories suitable for async meetings include (BUT ARE NOT LIMITED TO):
                        - Daily stand‑ups / 1:1s
                        - Brainstorms
                        - Peer feedback sessions
                        - Interview debriefs
                        - Project status updates
                        - CEO or company Q&A
                        - Design reviews
                        - Complex code reviews
                        - Product demos and reviews'''
                )
            },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Please analyze this image." },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ]
    )

    return completion.choices[0].message.parsed


if __name__ == "__main__":
    image_path = "uploads/calendar.png"
    result = analyze_calendar_image(image_path)
    print(result.model_dump_json(indent=2))