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
        model="gpt-4.1",
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
                        - Extract **all meetings** visible in the calendar.
                        - For each meeting, **you must** include:
                            - `title`
                            - `start_time` and `end_time` in ISO 8601 (e.g. `"2025-08-04T14:00:00"`)
                            - If times are missing (e.g. “bars” on a grid), *estimate* them based on relative bar height compared to events with explicit times—justify your estimate.
                            - `is_valuable`: Boolean indicating whether this meeting type is suitable for conversion to an **asynchronous video thread** on Arameet.
                            - `reason`: Brief explanation why it's considered valuable (or not).

                        3. Time analysis:
                        - Summarize `total_meeting_hours`.
                        - Compute `potential_savings_hours` by summing durations of “valuable” meetings.
                        - Provide `recommendations`: Suggestions for which meetings could be converted to async format (e.g. “Convert daily stand‑ups and design reviews to asynchronous video threads”).

                        4. Recommended categories suitable for async meetings include (but are not limited to):
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