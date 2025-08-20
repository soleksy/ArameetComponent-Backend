from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pathlib import Path
import shutil, uuid, time, logging

from models.calendar import CalendarAnalysis
from agent.analyzer import analyze_calendar_image

# ── Logging Setup ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("arameet.api")

# ── FastAPI App ──────────────────────────────────────────
app = FastAPI(
    title="Arameet Calendar Agent",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"https:\/\/.*\.framercanvas\.com$|https:\/\/.*\.framer\.app$|https:\/\/arameetcomponent-backend\.onrender\.com$",
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── /analyze Route ───────────────────────────────────────
@app.post("/analyze", response_model=CalendarAnalysis)
async def analyze(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    ext = Path(file.filename).suffix or ".png"
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
    
    try:
        with tmp_path.open("wb") as buf:
            shutil.copyfileobj(file.file, buf)
        log.info("Received %s (%s)", file.filename, file.content_type)

        t0 = time.perf_counter()
        result: CalendarAnalysis = await run_in_threadpool(
            analyze_calendar_image, str(tmp_path)
        )
        dt = time.perf_counter() - t0
        log.info("Analysis completed in %.2fs | Calendar detected: %s", dt, result.calendar_detected)

        return JSONResponse(content=result.model_dump())
    except Exception as exc:
        log.exception("Agent failure")
        raise HTTPException(500, f"Agent failure: {exc}") from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
            log.info("Deleted temporary file: %s", tmp_path)
        except Exception as e:
            log.warning("Failed to delete temp file %s: %s", tmp_path, e)
