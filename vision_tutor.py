#!/usr/bin/env python3
"""
Vision Tutor — Work Error Checker + Pencil Stillness + Socratic Help
• Watches camera for written work, checks errors every 10s (GPT-4o)
• Tracks where hand/pencil is active on the page
• If stuck at same location AND an error was found → Socratic help popup
• Tutor guides with questions — never gives the answer directly
• Press S = force scan  |  Q = quit
API: OpenRouter (vision: gpt-4o, tutor: claude-3.5-sonnet)
"""

import base64, json, os, platform, sys, threading, time, tkinter as tk
from collections import deque
from datetime import datetime
from pathlib import Path

# ── required packages ─────────────────────────────────────────────────────────
try:
    import cv2, numpy as np
except ImportError:
    print("Run: pip install opencv-python numpy"); sys.exit(1)
try:
    import httpx
except ImportError:
    print("Run: pip install httpx"); sys.exit(1)

IS_MAC      = platform.system() == "Darwin"
CONFIG_FILE = Path.home() / ".panicpoint" / "config.json"
_API_KEY    = ""   # set once at startup, used everywhere

# ── OpenRouter ────────────────────────────────────────────────────────────────
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
VISION_MODEL   = "openai/gpt-4o"
TUTOR_MODEL    = "anthropic/claude-3.5-sonnet"

# ── Tuning ────────────────────────────────────────────────────────────────────
PERIODIC_INTERVAL = 10.0   # vision check every N seconds
STILL_WAIT        = 2.5    # check after pen lifts
SCAN_INTERVAL     = 0.4    # motion scan frequency (seconds)
DIFF_THRESHOLD    = 0.035  # fraction of pixels that must change = "motion"
BLANK_BRIGHTNESS  = 40     # below this = dark/covered lens
BLANK_STD_DEV     = 18     # below this = blank/uniform page
STUCK_SECONDS     = 18.0   # seconds stuck at same location before offering help
STUCK_RADIUS_FRAC = 0.12   # fraction of frame width — how close counts as "same spot"
MIN_MOTION_PX     = 15     # minimum diff pixels to count as real motion (filters noise)
LOCKOUT_SECONDS   = 90     # minimum gap between help popups

# ── Colors (dark theme) ───────────────────────────────────────────────────────
BG_DARK  = "#020617"
BG_CARD  = "#0f172a"
BG_PANEL = "#1e293b"
C_MUTED  = "#475569"
C_SOFT   = "#94a3b8"
C_TEXT   = "#e2e8f0"
C_AMBER  = "#f59e0b"
C_GREEN  = "#22c55e"
C_RED    = "#ef4444"
C_BLUE   = "#60a5fa"
C_INDIGO = "#6366f1"

# ── Vision prompt ─────────────────────────────────────────────────────────────
VISION_PROMPT = """
You are a real-time academic tutor watching a student solve a problem on paper via camera.

FIRST — check if there is actual written work visible:
- If the image shows an empty desk, blank paper, or nothing legible, set has_work = false
  and return immediately with default values. Do NOT invent errors or give a score.

If has_work = true, do ALL of the following:
1. Identify the question/problem text.
2. Identify each step the student has written.
3. For every CLEARLY FINISHED step, check for: wrong calculation, wrong sign,
   wrong formula, skipped step, unit error, logic gap.
   - A step is ONLY finished if it has a complete expression with an equals sign and result.
   - DO NOT flag any step that looks mid-sentence, has a trailing operator, or is missing a result.
   - When in doubt, assume the student is still writing — skip the step entirely.
   - It is far better to miss an error than to flag something the student hasn't finished yet.
4. For each error, write a short "hint" (nudge to help them find the mistake
   themselves, WITHOUT giving the answer).
5. Detect if a box/rectangle is drawn around a final answer and evaluate it.

Respond ONLY in valid JSON — no markdown, no text outside JSON:

{
  "has_work": true,
  "question_detected": "the question text",
  "subject": "math|algebra|geometry|science|logic|other",
  "confidence": 0.9,
  "errors": [
    {
      "step": "Step 2",
      "found": "what was written wrong",
      "correction": "what it should be",
      "hint": "one-sentence nudge without giving the answer",
      "explanation": "full one-sentence reason",
      "severity": "critical|major|minor"
    }
  ],
  "boxed_answer": {
    "detected": false,
    "value": "",
    "verdict": "",
    "reason": ""
  },
  "overall_score": 100,
  "all_good": true
}

Rules:
- has_work = false → overall_score = null, errors = [], all_good = false, confidence = 0.
- all_good = true ONLY if errors = [] AND (no boxed answer OR boxed answer is correct).
- If no box drawn, boxed_answer.detected = false.
- Keep hint ONE sentence. Keep explanation ONE sentence.
"""

# ── Socratic tutor prompt ─────────────────────────────────────────────────────
TUTOR_SYSTEM = """You are a sharp Socratic tutor. A student is stuck on a math or science problem.
You know what mistake they made (given in their first message), but you MUST NEVER give the answer directly.

Rules:
- Ask ONE guiding question per turn that nudges them toward discovering the mistake themselves
- Reference what they actually wrote: "In step 2, you wrote X — what rule applies there?"
- Keep every response to 2-3 sentences max
- If they're getting closer: "Good thinking — now what does that mean for the next step?"
- If they figure it out: say exactly "Got it." then one sentence connecting back. End with RESUME on its own line.
- If after 3 turns still stuck: give a more direct hint (still not the answer)
- Tone: direct and warm, like a tutor who respects their intelligence
- NEVER say "the answer is", "you should write", or directly state the correct value"""


# ── API helpers ───────────────────────────────────────────────────────────────

def get_api_key() -> str:
    # 1. env var
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    # 2. shared config with panicpoint
    if CONFIG_FILE.exists():
        try:
            key = json.loads(CONFIG_FILE.read_text()).get("api_key", "")
            if key:
                return key
        except Exception:
            pass
    # 3. interactive prompt
    print("\n" + "─" * 44)
    print("  Vision Tutor — OpenRouter API key needed")
    print("─" * 44)
    key = input("OpenRouter API key (sk-or-...): ").strip()
    if not key:
        print("API key required."); sys.exit(1)
    # save for next time
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    data["api_key"] = key
    CONFIG_FILE.write_text(json.dumps(data, indent=2))
    return key


def encode_frame(frame) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def call_vision_api(b64: str) -> dict:
    try:
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {_API_KEY}",
                    "HTTP-Referer":  "https://vision-tutor.local",
                    "X-Title":       "Vision Tutor",
                    "Content-Type":  "application/json",
                },
                json={
                    "model": VISION_MODEL,
                    "messages": [{"role": "user", "content": [
                        {"type": "text",      "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}", "detail": "high"
                        }},
                    ]}],
                    "max_tokens": 1400,
                    "temperature": 0.15,
                },
            )
        if resp.status_code != 200:
            return {"api_error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        return {"api_error": f"JSON parse: {e}"}
    except Exception as e:
        return {"api_error": str(e)}


def call_tutor_api(history: list, max_tokens: int = 400) -> str:
    """Multi-turn Socratic tutor call."""
    messages = [{"role": "system", "content": TUTOR_SYSTEM}] + history
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {_API_KEY}",
                    "HTTP-Referer":  "https://vision-tutor.local",
                    "X-Title":       "Vision Tutor",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       TUTOR_MODEL,
                    "messages":    messages,
                    "max_tokens":  max_tokens,
                    "temperature": 0.5,
                },
            )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Error contacting tutor: {e}]"


# ── Frame helpers ─────────────────────────────────────────────────────────────

def is_blank_frame(frame) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
    return gray.mean() < BLANK_BRIGHTNESS or gray.std() < BLANK_STD_DEV


def motion_centroid(a, b) -> tuple[float, float] | None:
    """
    Return (cx_frac, cy_frac) as fraction of frame [0,1] where motion is occurring.
    Returns None if no significant motion detected.
    """
    s_a = cv2.GaussianBlur(
        cv2.cvtColor(cv2.resize(a, (160, 120)), cv2.COLOR_BGR2GRAY), (7, 7), 0
    ).astype(float)
    s_b = cv2.GaussianBlur(
        cv2.cvtColor(cv2.resize(b, (160, 120)), cv2.COLOR_BGR2GRAY), (7, 7), 0
    ).astype(float)
    diff = np.abs(s_a - s_b) > 25
    if diff.sum() < MIN_MOTION_PX:
        return None
    ys, xs = np.where(diff)
    return float(xs.mean() / 160.0), float(ys.mean() / 120.0)   # normalized [0,1]


def frame_has_motion(a, b) -> bool:
    s_a = cv2.GaussianBlur(
        cv2.cvtColor(cv2.resize(a, (160, 120)), cv2.COLOR_BGR2GRAY), (7, 7), 0
    ).astype(float)
    s_b = cv2.GaussianBlur(
        cv2.cvtColor(cv2.resize(b, (160, 120)), cv2.COLOR_BGR2GRAY), (7, 7), 0
    ).astype(float)
    return float((np.abs(s_a - s_b) > 25).sum()) / s_a.size > DIFF_THRESHOLD


# ── Stuck tracker ─────────────────────────────────────────────────────────────

class StuckTracker:
    """
    Tracks where hand/pencil activity is occurring.
    Reports 'stuck' when activity centroid hasn't moved beyond STUCK_RADIUS_FRAC
    of frame width for STUCK_SECONDS seconds.
    """

    def __init__(self):
        self._zone: tuple[float, float] | None = None   # (cx_frac, cy_frac)
        self._zone_since: float | None          = None
        self._last_motion_ts: float             = time.time()

    def update(self, centroid: tuple[float, float] | None):
        """
        Call each scan cycle. centroid = (cx_frac, cy_frac) or None if no motion.
        """
        now = time.time()
        if centroid is None:
            return   # no new info — don't reset

        self._last_motion_ts = now
        cx, cy = centroid

        if self._zone is None:
            # First motion detected — establish zone
            self._zone       = (cx, cy)
            self._zone_since = now
            return

        zx, zy = self._zone
        dist = ((cx - zx) ** 2 + (cy - zy) ** 2) ** 0.5   # in fraction units
        if dist > STUCK_RADIUS_FRAC:
            # Moved to a new area — reset
            self._zone       = (cx, cy)
            self._zone_since = now

    def is_stuck(self) -> bool:
        """True if hand/pencil has been in same zone for STUCK_SECONDS."""
        if self._zone_since is None:
            return False
        return (time.time() - self._zone_since) >= STUCK_SECONDS

    def time_stuck(self) -> float:
        if self._zone_since is None:
            return 0.0
        return max(0.0, time.time() - self._zone_since)

    def reset(self):
        self._zone       = None
        self._zone_since = None


# ── Socratic popup ────────────────────────────────────────────────────────────

class SocraticPopup:
    """
    Tkinter overlay that opens a Socratic tutoring session.
    Knows about the specific error so it can guide without giving the answer.
    """

    def __init__(self, root: tk.Tk, error: dict, question: str, on_dismiss):
        self.error      = error
        self.question   = question
        self.on_dismiss = on_dismiss
        self._history   = []
        self._waiting   = False

        self.win = tk.Toplevel(root)
        self._setup()
        self._build()
        self._keep_on_top()
        self._start()

    # ── window setup ──

    def _setup(self):
        w = self.win
        sw, sh = w.winfo_screenwidth(), w.winfo_screenheight()
        dlg_w, dlg_h = 700, 580
        w.configure(bg=BG_CARD)
        w.attributes("-topmost", True)
        w.attributes("-alpha", 0.97)
        w.resizable(False, False)
        w.title("Vision Tutor")
        w.geometry(f"{dlg_w}x{dlg_h}+{(sw-dlg_w)//2}+{(sh-dlg_h)//2}")
        w.bind("<Escape>", lambda e: self._dismiss())
        w.after(200, lambda: self.win.focus_force())

    def _keep_on_top(self):
        try:
            if self.win.winfo_exists():
                self.win.attributes("-topmost", True)
                self.win.lift()
                self.win.after(500, self._keep_on_top)
        except Exception:
            pass

    def _build(self):
        BG = BG_CARD

        # ── header ──
        hdr = tk.Frame(self.win, bg=BG, pady=10, padx=16)
        hdr.pack(fill="x")

        tk.Label(hdr, text="📐  STUCK?  Let's figure it out.",
                 font=("Helvetica", 14, "bold"),
                 fg=C_AMBER, bg=BG).pack(side="left")

        tk.Button(hdr, text="✕ dismiss (ESC)",
                  font=("Helvetica", 9), fg=C_MUTED, bg=BG,
                  activeforeground=C_TEXT, activebackground=BG,
                  relief="flat", bd=0, cursor="hand2",
                  command=self._dismiss).pack(side="right")

        tk.Frame(self.win, bg=BG_PANEL, height=1).pack(fill="x")

        # ── error context strip ──
        ctx = tk.Frame(self.win, bg="#1a1a2e", padx=16, pady=8)
        ctx.pack(fill="x")

        step_txt = self.error.get("step", "")
        found_txt = self.error.get("found", "")
        sev = self.error.get("severity", "major")
        sev_col = C_RED if sev == "critical" else C_AMBER if sev == "major" else C_BLUE

        tk.Label(ctx,
                 text=f"[{sev.upper()}]  {step_txt}  —  you wrote: \"{found_txt}\"",
                 font=("Helvetica", 10, "bold"),
                 fg=sev_col, bg="#1a1a2e",
                 anchor="w").pack(fill="x")

        hint = self.error.get("hint", "")
        if hint:
            tk.Label(ctx, text=f"Hint: {hint}",
                     font=("Helvetica", 9, "italic"),
                     fg=C_SOFT, bg="#1a1a2e",
                     anchor="w", wraplength=650).pack(fill="x", pady=(2, 0))

        tk.Frame(self.win, bg=BG_PANEL, height=1).pack(fill="x")

        # ── chat area ──
        chat_frame = tk.Frame(self.win, bg=BG)
        chat_frame.pack(fill="both", expand=True)

        self._chat = tk.Text(
            chat_frame, bg=BG_DARK, fg=C_TEXT,
            font=("Helvetica", 12), relief="flat", bd=0,
            wrap="word", padx=14, pady=10,
            state="disabled", cursor="arrow",
            spacing1=3, spacing3=3,
        )
        sb = tk.Scrollbar(chat_frame, command=self._chat.yview,
                          bg=BG_PANEL, troughcolor=BG_DARK)
        self._chat.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._chat.pack(side="left", fill="both", expand=True)

        self._chat.tag_config("label_tutor",  foreground=C_AMBER, font=("Helvetica", 9, "bold"))
        self._chat.tag_config("label_you",    foreground=C_MUTED, font=("Helvetica", 9, "bold"))
        self._chat.tag_config("label_resume", foreground=C_GREEN, font=("Helvetica", 9, "bold"))
        self._chat.tag_config("body",         foreground=C_TEXT,  font=("Helvetica", 12))
        self._chat.tag_config("body_you",     foreground=C_SOFT,  font=("Helvetica", 12, "italic"))
        self._chat.tag_config("body_resume",  foreground=C_GREEN, font=("Helvetica", 12))
        self._chat.tag_config("muted",        foreground=C_MUTED, font=("Helvetica", 10))

        # ── input row ──
        tk.Frame(self.win, bg=BG_PANEL, height=1).pack(fill="x")

        input_row = tk.Frame(self.win, bg=BG, padx=10, pady=8)
        input_row.pack(fill="x")

        self._input = tk.Text(
            input_row, height=2,
            bg=BG_PANEL, fg=C_TEXT,
            font=("Helvetica", 12), relief="flat", bd=0,
            insertbackground="white", wrap="word", padx=8, pady=5,
        )
        self._input.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self._send_btn = tk.Button(
            input_row, text="Send  ↵",
            font=("Helvetica", 10, "bold"),
            fg="white", bg=C_INDIGO,
            activeforeground="white", activebackground="#4338ca",
            relief="flat", bd=0, padx=12, pady=9,
            cursor="hand2", command=self._send,
        )
        self._send_btn.pack(side="right")

        self._input.bind("<Return>",       self._on_enter)
        self._input.bind("<Shift-Return>", lambda e: None)
        self._input.focus_set()

    # ── chat helpers ──

    def _append(self, label, label_tag, body, body_tag):
        self._chat.config(state="normal")
        self._chat.insert("end", f"{label}\n", label_tag)
        self._chat.insert("end", f"{body}\n\n", body_tag)
        self._chat.config(state="disabled")
        self._chat.see("end")

    def _set_thinking(self, on: bool):
        self._chat.config(state="normal")
        if on:
            self._chat.insert("end", "⚡ thinking…\n", "muted")
            self._send_btn.config(state="disabled")
            self._input.config(state="disabled")
        else:
            content = self._chat.get("1.0", "end")
            if "⚡ thinking…" in content:
                idx = self._chat.search("⚡ thinking…", "1.0", "end")
                if idx:
                    self._chat.delete(idx, f"{idx} lineend +1c")
            self._send_btn.config(state="normal")
            self._input.config(state="normal")
            self._input.focus_set()
        self._chat.config(state="disabled")

    # ── conversation ──

    def _start(self):
        step  = self.error.get("step", "a step")
        found = self.error.get("found", "something")
        expl  = self.error.get("explanation", "")
        q     = self.question or "(problem not detected)"

        first_msg = (
            f"Problem: {q}\n"
            f"Mistake location: {step}\n"
            f"What they wrote: {found}\n"
            f"Why it's wrong: {expl}\n\n"
            "The student has been stuck here for a while. Start the Socratic dialogue."
        )
        self._history.append({"role": "user", "content": first_msg})
        self._set_thinking(True)

        def worker():
            reply = call_tutor_api(self._history)
            self._history.append({"role": "assistant", "content": reply})
            self.win.after(0, lambda: self._render_reply(reply))

        threading.Thread(target=worker, daemon=True).start()

    def _on_enter(self, event):
        self._send()
        return "break"

    def _send(self):
        if self._waiting:
            return
        text = self._input.get("1.0", "end").strip()
        if not text:
            return
        self._input.delete("1.0", "end")
        self._append("You", "label_you", text, "body_you")
        self._history.append({"role": "user", "content": text})
        self._waiting = True
        self._set_thinking(True)

        def worker():
            reply = call_tutor_api(self._history)
            self._history.append({"role": "assistant", "content": reply})
            self.win.after(0, lambda: self._render_reply(reply))

        threading.Thread(target=worker, daemon=True).start()

    def _render_reply(self, reply: str):
        self._set_thinking(False)
        self._waiting = False

        if "Got it." in reply or "RESUME" in reply:
            clean = reply.replace("RESUME", "").replace("Got it.", "").strip()
            self._append("✓  Got it!", "label_resume",
                         f"Got it. {clean}", "body_resume")
            self._input.config(state="disabled")
            self._send_btn.config(state="disabled")
            tk.Button(
                self.win, text="▶  Back to work",
                font=("Helvetica", 12, "bold"),
                fg="white", bg="#16a34a",
                activeforeground="white", activebackground="#15803d",
                relief="flat", bd=0, padx=20, pady=9,
                cursor="hand2", command=self._dismiss,
            ).pack(pady=(0, 10))
        else:
            self._append("Tutor", "label_tutor", reply, "body")

    def _dismiss(self):
        try:
            self.win.destroy()
        except Exception:
            pass
        self.on_dismiss()


# ── Terminal output ───────────────────────────────────────────────────────────

def print_result(result: dict, label: str = ""):
    if "api_error" in result:
        print(f"\n  [API ERROR] {result['api_error']}\n")
        return
    if not result.get("has_work"):
        print(f"\n  [{label}] No paper/work detected — skipped.\n")
        return
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*68}  [{label}  {ts}]")
    q = result.get("question_detected", "")
    if q:
        print(f"  QUESTION : {q}")
    score = result.get("overall_score")
    conf  = result.get("confidence", 1.0)
    print(f"  SUBJECT  : {result.get('subject','?').upper()}"
          f"    SCORE: {f'{score}/100' if score is not None else 'n/a'}"
          f"    CONFIDENCE: {int(conf*100)}%")
    errors = result.get("errors", [])
    if not errors:
        print("  No errors in completed steps.")
    else:
        for e in errors:
            print(f"\n  [{e.get('severity','?').upper()}] {e.get('step','?')}")
            print(f"    Found     : {e.get('found','')}")
            print(f"    Hint      : {e.get('hint','')}")
            print(f"    Correction: {e.get('correction','')}")
    boxed = result.get("boxed_answer", {})
    if boxed.get("detected"):
        v   = (boxed.get("verdict") or "").upper()
        sym = "CORRECT" if v == "CORRECT" else "WRONG"
        print(f"\n  BOXED ANSWER [{sym}]: {boxed.get('value','')} — {boxed.get('reason','')}")
    print("="*68)


# ── CV2 overlay ───────────────────────────────────────────────────────────────

def draw_overlay(frame, state: dict):
    h, w = frame.shape[:2]
    ov   = frame.copy()

    # top bar
    cv2.rectangle(ov, (0, 0), (w, 46), (20, 20, 20), -1)
    cv2.putText(ov, state.get("status", "Watching..."), (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                state.get("status_color", (200,200,200)), 2, cv2.LINE_AA)

    score = state.get("score")
    if score is not None:
        sc = (80,220,80) if score>=80 else (60,200,220) if score>=50 else (60,60,220)
        cv2.putText(ov, f"Score: {score}/100", (w-195, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.63, sc, 2, cv2.LINE_AA)

    q = state.get("question", "")
    if q:
        cv2.rectangle(ov, (0, 46), (w, 72), (15, 15, 50), -1)
        cv2.putText(ov, f"Q: {q[:88]}", (8, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160,205,255), 1, cv2.LINE_AA)

    nxt = state.get("next_in")
    if nxt is not None and not state.get("analyzing") and not state.get("blank"):
        stuck_t = state.get("stuck_time", 0.0)
        stuck_s = f"  |  stuck: {stuck_t:.0f}s" if stuck_t > 2 else ""
        cv2.putText(ov, f"next scan: {nxt:.0f}s  |  S = scan now{stuck_s}",
                    (8, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110,110,110), 1, cv2.LINE_AA)

    if state.get("blank"):
        cv2.putText(ov, "No paper detected — point camera at your work",
                    (w//2-240, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (100,100,220), 2, cv2.LINE_AA)

    # error banners
    errors  = state.get("errors", [])
    boxed   = state.get("boxed_answer", {})
    box_h   = 38 if boxed.get("detected") else 0
    for i, e in enumerate(errors[:4]):
        sev = e.get("severity","major")
        bg  = (0,0,165) if sev=="critical" else (0,65,185) if sev=="major" else (0,105,85)
        y   = h - box_h - 40 - i*38
        cv2.rectangle(ov, (0, y), (w, y+36), bg, -1)
        l1 = f"!  {e.get('step','?')}: {e.get('found','')}  →  {e.get('correction','')}"
        cv2.putText(ov, l1[:105], (8, y+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1, cv2.LINE_AA)
        hint = e.get("hint","")
        if hint:
            cv2.putText(ov, f"   Hint: {hint[:95]}", (8, y+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,230,255), 1, cv2.LINE_AA)

    if boxed.get("detected"):
        verdict = (boxed.get("verdict") or "").lower()
        bg  = (0,130,0) if verdict=="correct" else (0,0,185)
        sym = "CORRECT" if verdict=="correct" else "WRONG"
        txt = f"[{sym}] Boxed: {boxed.get('value','')} — {boxed.get('reason','')}"
        cv2.rectangle(ov, (0, h-38), (w, h), bg, -1)
        cv2.putText(ov, txt[:108], (8, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255,255,255), 2, cv2.LINE_AA)

    if state.get("analyzing"):
        dots = "." * (int(time.time()*2) % 4)
        cv2.putText(ov, f"Analyzing{dots}", (w//2-85, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0,215,255), 2, cv2.LINE_AA)

    # stuck indicator (progress arc-ish bar)
    stuck_t = state.get("stuck_time", 0.0)
    if stuck_t > 2 and not state.get("popup_open") and errors:
        frac = min(1.0, stuck_t / STUCK_SECONDS)
        bar_w = int(w * frac)
        col   = (0,200,255) if frac < 0.7 else (0,120,255) if frac < 1.0 else (0,60,200)
        cv2.rectangle(ov, (0, 93), (bar_w, 97), col, -1)
        if frac >= 1.0:
            cv2.putText(ov, "! Stuck — opening tutor...", (8, 112),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0,200,255), 1, cv2.LINE_AA)

    cv2.addWeighted(ov, 0.86, frame, 0.14, 0, frame)
    return frame


# ── Main loop ─────────────────────────────────────────────────────────────────

def live_watch(api_key: str):
    global _API_KEY
    _API_KEY = api_key
    # ── camera ──
    cap = None
    for idx in range(5):
        c = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if c.isOpened():
            for _ in range(8):
                c.read(); time.sleep(0.04)
            cap = c
            print(f"Camera ready (index {idx})")
            break
    if cap is None:
        print("ERROR: No camera found."); sys.exit(1)

    # ── tkinter root (hidden — just hosts popups) ──
    root = tk.Tk()
    root.withdraw()
    if IS_MAC:
        try:
            root.tk.call("::tk::unsupported::MacWindowStyle",
                         "style", root._w, "floating", "noActivates")
        except Exception:
            pass

    print("\nVision Tutor — Live")
    print("  S = force scan  |  Q = quit\n")

    # ── shared state ──
    lock         = threading.Lock()
    state        = {
        "status":       "Watching...",
        "status_color": (180,180,180),
        "score":        None,
        "question":     "",
        "errors":       [],
        "boxed_answer": {},
        "analyzing":    False,
        "next_in":      PERIODIC_INTERVAL,
        "blank":        False,
        "stuck_time":   0.0,
        "popup_open":   False,
    }
    error_history  = []
    active_errors  = []       # current error list from last vision check
    active_question = ""

    stuck_tracker   = StuckTracker()
    last_frame      = None
    last_scan_at    = 0.0
    last_periodic_at = time.time()
    still_since     = None
    still_triggered = False
    force_snap      = False
    popup_ref       = None
    last_popup_at   = 0.0

    def on_popup_dismiss():
        nonlocal popup_ref
        with lock:
            state["popup_open"] = False
        popup_ref = None
        stuck_tracker.reset()

    def open_socratic_popup(error: dict, question: str):
        nonlocal popup_ref, last_popup_at
        with lock:
            state["popup_open"] = True
        last_popup_at = time.time()
        popup_ref = SocraticPopup(root, error, question, on_popup_dismiss)

    def run_analysis(snap, label=""):
        nonlocal active_errors, active_question
        if is_blank_frame(snap):
            with lock:
                state.update(analyzing=False, blank=True, score=None,
                             status="No paper detected",
                             status_color=(120,120,120))
            return

        with lock:
            state.update(analyzing=True, blank=False,
                         status="Analyzing...", status_color=(0,215,255))

        result = call_vision_api(encode_frame(snap))
        print_result(result, label)

        errors  = result.get("errors", [])
        boxed   = result.get("boxed_answer", {})
        score   = result.get("overall_score")
        q       = result.get("question_detected", "")
        conf    = result.get("confidence", 1.0)
        hw      = result.get("has_work", True)

        active_errors   = errors if hw else []
        active_question = q or active_question

        ts = datetime.now().strftime("%H:%M:%S")
        for e in errors:
            if not any(h.get("found") == e.get("found") and
                       h.get("step") == e.get("step") for h in error_history):
                error_history.append({**e, "ts": ts})

        with lock:
            state["analyzing"] = False
            state["blank"]     = not hw
            if not hw:
                state.update(status="No paper detected", status_color=(120,120,120), score=None)
                return
            if q:
                state["question"] = q
            state["score"]        = score if conf >= 0.5 and score is not None else None
            state["errors"]       = errors
            state["boxed_answer"] = boxed if boxed.get("detected") else state["boxed_answer"]

            if "api_error" in result:
                state.update(status=f"API error: {result['api_error'][:55]}",
                             status_color=(0,0,200))
            elif conf < 0.5:
                state.update(status="Low confidence — move paper into frame",
                             status_color=(100,160,200))
            elif errors:
                state.update(status=f"{len(errors)} mistake(s) found!",
                             status_color=(0,60,235))
            elif boxed.get("detected") and (boxed.get("verdict") or "").lower() == "incorrect":
                state.update(status="Boxed answer is WRONG", status_color=(0,0,210))
            elif boxed.get("detected") and (boxed.get("verdict") or "").lower() == "correct":
                state.update(status="Boxed answer CORRECT!", status_color=(0,195,75))
            else:
                state.update(status="Looks good so far", status_color=(0,185,75))

    # ── main loop ──
    while True:
        ret, frame = cap.read()
        if not ret:
            root.update()
            continue

        now = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key in (ord('s'), ord('S')):
            force_snap = True

        # ── scan cycle ──
        if now - last_scan_at >= SCAN_INTERVAL:
            last_scan_at = now

            with lock:
                analyzing  = state["analyzing"]
                popup_open = state["popup_open"]

            # pencil/hand position via motion centroid from paper camera
            centroid = None
            if last_frame is not None:
                centroid = motion_centroid(last_frame, frame)

            stuck_tracker.update(centroid)
            stuck_t = stuck_tracker.time_stuck()
            with lock:
                state["stuck_time"] = stuck_t

            # ── trigger Socratic popup ──
            if (stuck_t >= STUCK_SECONDS
                    and active_errors
                    and not popup_open
                    and not analyzing
                    and now - last_popup_at >= LOCKOUT_SECONDS):
                worst = sorted(
                    active_errors,
                    key=lambda e: {"critical":0,"major":1,"minor":2}.get(e.get("severity","major"),1)
                )[0]
                root.after(0, lambda e=worst, q=active_question: open_socratic_popup(e, q))

            # ── force snap (S key) ──
            if force_snap and not analyzing:
                force_snap = False
                last_periodic_at = now
                still_triggered  = False
                snap = frame.copy()
                threading.Thread(target=run_analysis, args=(snap, "manual"), daemon=True).start()

            elif last_frame is not None and not analyzing:
                moving = frame_has_motion(last_frame, frame)
                if moving:
                    still_since     = None
                    still_triggered = False
                    with lock:
                        if not state["blank"]:
                            state.update(status="Watching...", status_color=(170,170,170))
                else:
                    if still_since is None:
                        still_since = now
                    if now - still_since >= STILL_WAIT and not still_triggered:
                        still_triggered  = True
                        last_periodic_at = now
                        snap = frame.copy()
                        threading.Thread(
                            target=run_analysis, args=(snap, "still"), daemon=True
                        ).start()

            last_frame = frame.copy()

            if not analyzing:
                remaining = PERIODIC_INTERVAL - (now - last_periodic_at)
                with lock:
                    state["next_in"] = max(0.0, remaining)
                if now - last_periodic_at >= PERIODIC_INTERVAL:
                    last_periodic_at = now
                    still_triggered  = False
                    snap = frame.copy()
                    threading.Thread(
                        target=run_analysis, args=(snap, "periodic"), daemon=True
                    ).start()

        # ── draw ──
        display = frame.copy()
        with lock:
            s = dict(state)
        display = draw_overlay(display, s)
        cv2.imshow("Vision Tutor  (S = scan  |  Q = quit)", display)

        # pump tkinter events
        try:
            root.update()
        except tk.TclError:
            break

    cap.release()
    cv2.destroyAllWindows()
    try:
        root.destroy()
    except Exception:
        pass

    # session summary
    print("\n" + "="*68)
    print("  SESSION SUMMARY")
    print("="*68)
    if not error_history:
        print("  No errors caught this session.")
    else:
        print(f"  {len(error_history)} error(s) found:\n")
        for i, e in enumerate(error_history, 1):
            print(f"  {i}. [{e.get('ts','')}] [{e.get('severity','?').upper()}] "
                  f"{e.get('step','?')} — {e.get('found','')}")
    print("="*68 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    live_watch(get_api_key())
