#!/usr/bin/env python3
"""
FlowState WebServer
Serves the dashboard and streams real-time data from vision_tutor via SSE + MJPEG.
Auto-started by vision_tutor.py, or run standalone: python server.py
"""

import json
import os
import queue
import threading
import time
from pathlib import Path

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from flask import Flask, Response, jsonify, request, send_file
    FLASK_OK = True
except ImportError:
    FLASK_OK = False
    print("[Server] Flask not found — run: pip install flask")

FRONTEND_PATH = Path(__file__).parent / "frontend.html"

app = Flask(__name__) if FLASK_OK else None

# ── shared state ──────────────────────────────────────────────────────────────
_lock  = threading.Lock()
_state: dict = {
    "cam_ok":             False,
    "subject":            "general",
    "status":             "Standby",
    "status_state":       "standby",   # flow | drift | stuck | standby
    "focus_score":        0,
    "score":              None,
    "question":           "",
    "errors":             [],
    "repeated_errors":    [],
    "boxed_answer":       {},
    "analyzing":          False,
    "blank":              False,
    "stuck_time":         0.0,
    "popup_open":         False,
    "fidget":             0.0,
    "fidget_velocity":    0.0,
    "fidget_tapping":     0.0,
    "fidget_tension":     0.0,
    "fidget_restlessness":0.0,
    "session_active":     False,
    "working_seconds":    0,
    "distracted_seconds": 0,
    "session_seconds":    0,
    "error_history":      [],
    "ts":                 0.0,
}
_frame         = None
_subscribers   = []
_subs_lock     = threading.Lock()


def push_state(data: dict):
    """Update shared state and notify SSE clients. Called by vision_tutor."""
    with _lock:
        _state.update(data)
        _state["ts"] = time.time()
    _notify()


def push_frame(frame):
    """Push latest annotated BGR frame for MJPEG streaming."""
    global _frame
    with _lock:
        _frame = frame.copy() if frame is not None else None


def _notify():
    with _subs_lock:
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait(True)
            except Exception:
                dead.append(q)
        for q in dead:
            try:
                _subscribers.remove(q)
            except ValueError:
                pass


# ── routes ────────────────────────────────────────────────────────────────────
if FLASK_OK:

    @app.after_request
    def _cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    @app.route("/")
    def index():
        if FRONTEND_PATH.exists():
            return send_file(str(FRONTEND_PATH))
        return "<h2>frontend.html not found</h2>", 404

    @app.route("/api/state")
    def get_state():
        with _lock:
            return jsonify(dict(_state))

    @app.route("/api/teach", methods=["POST"])
    def teach():
        """Generate a step-by-step teaching guide for the current question (no answer given)."""
        try:
            import httpx
            body    = request.get_json(force=True) or {}
            question = body.get("question") or _state.get("question") or "the current problem"
            errors   = body.get("errors")   or _state.get("errors")   or []
            boxed    = body.get("boxed")    or _state.get("boxed_answer") or {}
            api_key  = os.environ.get("OPENROUTER_API_KEY","")
            if not api_key:
                return jsonify({"error":"No API key"}), 400

            err_text = ""
            if errors:
                err_text = "\nDetected mistakes:\n" + "\n".join(
                    f"- {e.get('step','?')}: wrote '{e.get('found','')}' — {e.get('hint','')}"
                    for e in errors[:4]
                )
            if boxed.get("detected") and (boxed.get("verdict") or "").lower() == "incorrect":
                err_text += f"\nTheir boxed answer was: {boxed.get('value','')} (marked incorrect)"

            prompt = f"""A student is working on: "{question}"
{err_text}

Create a TEACHING GUIDE that helps them understand HOW to solve this type of problem.

Rules:
- DO NOT give the final answer or any numeric result for this specific problem
- DO explain the METHOD and STEPS conceptually
- Use "you" language: "First you should...", "Next, you need to..."
- Reference what they wrote if there are errors, but guide them to self-discover the fix
- Keep each step short (1-2 sentences)
- End with a hint that nudges toward the answer without stating it

Return ONLY valid JSON:
{{
  "title": "How to solve: {question[:60]}",
  "concept": "one sentence explaining the key math concept",
  "steps": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ],
  "hint": "nudge toward the answer without giving it"
}}"""

            resp = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model":"anthropic/claude-3.5-sonnet","messages":[{"role":"user","content":prompt}],
                      "max_tokens":600,"temperature":0.3},
                timeout=20.0
            )
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"): raw = raw[4:]
            return jsonify(json.loads(raw.strip()))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/practice", methods=["POST"])
    def practice():
        """Generate targeted practice problems based on detected errors."""
        try:
            import httpx
            body     = request.get_json(force=True) or {}
            question = body.get("question") or _state.get("question") or ""
            errors   = body.get("errors")   or _state.get("errors")   or []
            subject  = body.get("subject")  or _state.get("subject")  or "math"
            api_key  = os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                return jsonify({"error": "No API key"}), 400

            err_text = ""
            if errors:
                err_text = "\nSpecific mistakes detected:\n" + "\n".join(
                    f"- {e.get('step','?')}: wrote '{e.get('found','')}' — {e.get('hint','')}"
                    for e in errors[:4]
                )

            prompt = f"""A student is practicing {subject}. They just worked on: "{question or 'an equation'}"
{err_text}

Generate 4 targeted practice problems that directly address their mistake pattern.
Each problem should practice the SAME skill but with different numbers.
Order them: easiest first, gradually harder.

Return ONLY valid JSON:
{{
  "topic": "one phrase describing the skill being practiced",
  "mistake_pattern": "one sentence: what conceptual error the student made",
  "problems": [
    {{
      "q": "the practice problem, e.g. 'Solve for x: x - 8 = 15'",
      "difficulty": "easy|medium|hard",
      "hint": "one-sentence hint for this specific problem (no answer)"
    }}
  ]
}}"""

            resp = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "anthropic/claude-3.5-sonnet",
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 600, "temperature": 0.4},
                timeout=20.0
            )
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"): raw = raw[4:]
            return jsonify(json.loads(raw.strip()))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/stream")
    def sse_stream():
        """Server-Sent Events — pushes state to browser on every change."""
        q = queue.Queue(maxsize=30)
        with _subs_lock:
            _subscribers.append(q)

        def generate():
            with _lock:
                yield f"data: {json.dumps(dict(_state))}\n\n"
            try:
                while True:
                    try:
                        q.get(timeout=5.0)
                    except queue.Empty:
                        yield ": ping\n\n"
                        continue
                    with _lock:
                        yield f"data: {json.dumps(dict(_state))}\n\n"
            except GeneratorExit:
                pass
            finally:
                with _subs_lock:
                    try:
                        _subscribers.remove(q)
                    except ValueError:
                        pass

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/api/video")
    def video_feed():
        """MJPEG stream of the live annotated OpenCV frame."""
        def generate():
            while True:
                with _lock:
                    f = _frame
                if f is not None and CV2_OK:
                    try:
                        ok, buf = cv2.imencode(
                            ".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 72]
                        )
                        if ok:
                            yield (
                                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                + buf.tobytes()
                                + b"\r\n"
                            )
                    except Exception:
                        pass
                time.sleep(0.04)   # ~25 fps

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )


def start_server(port: int = 5001):
    """Start Flask in a daemon background thread."""
    if not FLASK_OK:
        return
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    t = threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0", port=port,
            debug=False, use_reloader=False, threaded=True,
        ),
        daemon=True,
        name="FlowState-Server",
    )
    t.start()
    print(f"[FlowState] Dashboard → http://localhost:{port}")


if __name__ == "__main__":
    print("FlowState server — http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=True)
