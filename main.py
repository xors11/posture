

import base64
import json
import logging
import time

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import posture_api as api

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Posture Corrector API",
    description="Real-time posture analysis via WebSocket.",
    version="1.0.0",
)

# Allow the React dev server (and any other origin in development) to connect.
# Tighten this list when deploying to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST helpers (optional, useful for health-checks from the React app)
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Simple health-check endpoint."""
    return {"status": "ok"}


@app.get("/exercises")
def list_exercises():
    """Return the list of supported exercise names."""
    return {"exercises": api.SUPPORTED_EXERCISES}


@app.get("/recommend")
def recommend(bmi: float):
    """Return BMI-based exercise recommendations."""
    return {"exercises": api.recommend_exercise(bmi)}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/pose")
async def pose_websocket(websocket: WebSocket):
    """
    Real-time posture analysis WebSocket.

    Client → Server message (JSON):
    --------------------------------
    {
        "exercise": "Bicep Curl",
        "frame":    "<base64-encoded JPEG string>"
    }

    Server → Client response (JSON):
    ---------------------------------
    {
        "correct_posture": true,
        "progress":        72.5,
        "posture_score":   91.0,
        "rep_count":       3,
        "exercise":        "Bicep Curl",
        "feedback":        "Excellent form! Keep it up."
    }

    Error response (JSON):
    ----------------------
    {
        "error": "<description>"
    }
    """
    await websocket.accept()
    log.info("WebSocket connected: %s", websocket.client)

    # Each connection owns its own isolated session (Pose, smoother, state).
    session = api.create_session()

    try:
        while True:
            # ----------------------------------------------------------------
            # 1. Receive message
            # ----------------------------------------------------------------
            raw = await websocket.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            exercise_name = msg.get("exercise", "").strip()
            frame_b64 = msg.get("frame", "").strip()

            # ----------------------------------------------------------------
            # 2. Validate
            # ----------------------------------------------------------------
            if not exercise_name:
                await websocket.send_text(
                    json.dumps({"error": "Missing 'exercise' field"})
                )
                continue

            if exercise_name not in api.SUPPORTED_EXERCISES:
                await websocket.send_text(
                    json.dumps({
                        "error": f"Unknown exercise '{exercise_name}'. "
                                 f"Supported: {api.SUPPORTED_EXERCISES}"
                    })
                )
                continue

            if not frame_b64:
                await websocket.send_text(
                    json.dumps({"error": "Missing 'frame' field"})
                )
                continue

            # ----------------------------------------------------------------
            # 3. Decode base64 → JPEG bytes → BGR numpy array
            # ----------------------------------------------------------------
            try:
                # Strip the optional data-URL prefix if the browser sends it:
                # "data:image/jpeg;base64,/9j/4AAQ..."
                if "," in frame_b64:
                    frame_b64 = frame_b64.split(",", 1)[1]

                jpeg_bytes = base64.b64decode(frame_b64)
                np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame_bgr is None:
                    raise ValueError("cv2.imdecode returned None")

            except Exception as decode_err:
                log.warning("Frame decode error: %s", decode_err)
                await websocket.send_text(
                    json.dumps({"error": f"Frame decode failed: {decode_err}"})
                )
                continue

            # ----------------------------------------------------------------
            # 4. Run the MediaPipe + exercise_logic pipeline
            # ----------------------------------------------------------------
            t_start = time.perf_counter()
            try:
                result = api.process_frame(frame_bgr, exercise_name, session)
            except Exception as proc_err:
                log.exception("process_frame error: %s", proc_err)
                await websocket.send_text(
                    json.dumps({"error": f"Processing error: {proc_err}"})
                )
                continue
            proc_ms = round((time.perf_counter() - t_start) * 1000.0, 1)

            # Pass-through diagnostics for end-to-end frame synchronization & telemetry
            if "seq" in msg:
                result["seq"] = msg["seq"]
            if "timestamp" in msg:
                result["client_ts"] = msg["timestamp"]
            result["proc_ms"] = proc_ms
            result["input_dims"] = f"{frame_bgr.shape[1]}x{frame_bgr.shape[0]}"

            # ----------------------------------------------------------------
            # 5. Send JSON result back to the client
            # ----------------------------------------------------------------
            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        log.info("WebSocket disconnected: %s", websocket.client)

    except Exception as exc:
        log.exception("Unexpected WebSocket error: %s", exc)

    finally:
        # Always release the per-session MediaPipe Pose instance.
        api.close_session(session)
        log.info("Session cleaned up for: %s", websocket.client)
