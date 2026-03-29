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

import base64, hashlib, json, os, platform, re, subprocess, sys, threading, time, tkinter as tk
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

try:
    from hand_tracker import HandTracker
    HAND_TRACKER_OK = True
except ImportError:
    HAND_TRACKER_OK = False

try:
    import server as _server
    SERVER_OK = True
except ImportError:
    SERVER_OK = False

# ── OpenRouter ────────────────────────────────────────────────────────────────
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
VISION_MODEL    = "openai/gpt-4o"                  # primary vision model
CONSENSUS_MODEL = "google/gemini-2.0-flash-001"    # secondary — must agree for result to count
FAST_MODEL      = "openai/gpt-4o-mini"             # periodic scans (single model, speed)
TUTOR_MODEL     = "anthropic/claude-3.5-sonnet"

# ── Tuning ────────────────────────────────────────────────────────────────────
PERIODIC_INTERVAL = 6.0    # vision check every N seconds
STILL_WAIT        = 2.5    # check after pen lifts
SCAN_INTERVAL     = 0.4    # motion scan frequency (seconds)
DIFF_THRESHOLD    = 0.035  # fraction of pixels that must change = "motion"
BLANK_BRIGHTNESS  = 40     # below this = dark/covered lens
BLANK_STD_DEV     = 18     # below this = blank/uniform page
STUCK_SECONDS     = 12.0   # seconds stuck at same location before offering help
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
You are a careful math/science tutor watching a student's handwritten work via camera.
Your #1 rule: NEVER flag a correct step. A false alarm is far worse than a missed error.

── STEP 0: ANSWER VERIFICATION (do this FIRST) ──────────────────────────────────
1. Find the student's final answer (last written value, or boxed value).
2. Substitute it back into the original equation/problem.
3. If it makes the equation TRUE → ALL steps are correct. Set errors=[], all_good=true. STOP.
4. Only if the substitution is FALSE → then examine individual steps for the mistake.

Concrete example:
  Q: x + 17 = 22
  Student writes: x = 22 − 17, then x = 5
  Verify: 5 + 17 = 22 ✓  →  errors=[], all_good=true.  DO NOT FLAG ANYTHING.

── STEP 1: Is there work to check? ──────────────────────────────────────────────
If the image is blank, empty desk, or has no legible writing: set has_work=false, return defaults.

── STEP 2: Identify and SKIP crossed-out work ───────────────────────────────────
BEFORE reading any step, look for crossed-out marks:
• Lines drawn through a step, X marks, heavy scribbles = CANCELLED WORK.
• COMPLETELY IGNORE any crossed-out step — do not flag, mention, or score it.
• Only evaluate clean, uncrossed steps.

── STEP 3: Read handwriting charitably ──────────────────────────────────────────
• A horizontal stroke between two numbers or variables IS a minus sign (−).
  Never read "a − b" as "a b" or as a positive number.
• Faint, short, or thin horizontal lines between terms = minus sign. Assume it.
• Sloppy equals signs, arrows, underlines = formatting, not errors.
• If a symbol is ambiguous, pick the interpretation that makes the step CORRECT.
• Never invent errors based on handwriting style or slant.

── STEP 4: Verify steps (only if Step 0 says answer is wrong) ────────────────────
For each completed step (has "=" and a value on both sides):
  a. Compute what the step SHOULD produce from the prior step.
  b. Compute what the student ACTUALLY wrote (read charitably).
  c. Flag ONLY if (b) is provably wrong AND you are ≥95% confident.
  d. If any doubt exists — skip. Do NOT flag.

CORRECT moves to NEVER flag:
• x + n = m  →  x = m − n          ✓ (subtract n from both sides)
• x − n = m  →  x = m + n          ✓ (add n to both sides)
• nx = m     →  x = m/n            ✓ (divide both sides by n)
• x/n = m    →  x = m·n            ✓ (multiply both sides by n)
• Distributing, factoring, combining like terms that are equivalent ✓

Only flag (≥95% confidence, verified answer is wrong):
• Arithmetic result is provably wrong (e.g. 22 − 17 = 6 instead of 5)
• Wrong algebraic operation applied (adding when should subtract, etc.)
• Sign error that causes a wrong final answer
• Wrong physics/geometry formula used

── STEP 5: Skip incomplete or ambiguous steps ────────────────────────────────────
Skip: trailing operators, no result yet, partially written, unclear handwriting.
When in doubt → skip. Silence is correct.

── OUTPUT FORMAT ────────────────────────────────────────────────────────────────
Respond ONLY in valid JSON — no markdown, no text outside JSON:

{
  "has_work": true,
  "question_detected": "the question text",
  "subject": "math|algebra|geometry|science|logic|other",
  "confidence": 0.95,
  "errors": [
    {
      "step": "Step 2",
      "found": "exactly what the student wrote",
      "correction": "what it should be",
      "hint": "one-sentence nudge without giving the answer",
      "explanation": "one sentence: what is mathematically wrong",
      "severity": "critical|major|minor"
    }
  ],
  "boxed_answer": {
    "detected": false,
    "value": "",
    "verdict": "correct|incorrect|",
    "reason": ""
  },
  "overall_score": 100,
  "all_good": true
}

Rules:
- has_work=false → overall_score=null, errors=[], all_good=false, confidence=0.
- all_good=true ONLY if errors=[] AND (no boxed answer OR boxed answer is correct).
- If no box drawn, boxed_answer.detected=false.
- boxed_answer.verdict: substitute the boxed value into the original equation to verify.
- overall_score: 100 if errors=[], subtract 15 per error, minimum 0.
- Confidence = how sure you are the handwriting reading is correct (not error existence).
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
    # 1. config file (preferred — avoids shell typos)
    if CONFIG_FILE.exists():
        try:
            key = json.loads(CONFIG_FILE.read_text()).get("api_key", "").strip()
            if key.startswith("sk-or-"):
                os.environ["OPENROUTER_API_KEY"] = key
                return key
        except Exception:
            pass
    # 2. env var (only if looks valid)
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key.startswith("sk-or-"):
        return key
    key = ""  # invalid / missing
    # 3. interactive prompt if still nothing
    if not key:
        print("\n" + "─" * 44)
        print("  Vision Tutor — OpenRouter API key needed")
        print("─" * 44)
        key = input("OpenRouter API key (sk-or-...): ").strip()
        if not key:
            print("API key required."); sys.exit(1)
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text())
            except Exception:
                pass
        data["api_key"] = key
        CONFIG_FILE.write_text(json.dumps(data, indent=2))
    # Cache in env so every thread can read it with no globals needed
    os.environ["OPENROUTER_API_KEY"] = key
    return key


def encode_frame(frame) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def _call_single(b64: str, model: str) -> dict:
    """Call one vision model and return parsed JSON result."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    try:
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "HTTP-Referer":  "https://vision-tutor.local",
                    "X-Title":       "Vision Tutor",
                    "Content-Type":  "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": [
                        {"type": "text",      "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}", "detail": "high"
                        }},
                    ]}],
                    "max_tokens": 1400,
                    "temperature": 0.1,
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
        return {"api_error": str(e)[:120]}


def _merge_consensus(results: list) -> dict:
    """
    Merge results from multiple models.
    Errors: only kept if EVERY model flags the same step.
    Boxed verdict: only fires if ALL models agree (correct vs incorrect).
    Score/question: averaged / longest.
    """
    valid = [r for r in results if r and "api_error" not in r]
    if not valid:
        return results[0] if results else {"api_error": "all models failed"}
    if len(valid) == 1:
        return valid[0]

    # has_work
    hw = any(r.get("has_work", True) for r in valid)

    # question: longest non-empty string across models
    questions = [r.get("question_detected", "") for r in valid if r.get("question_detected")]
    question  = max(questions, key=len) if questions else ""

    # confidence: minimum (most conservative)
    min_conf = min(r.get("confidence", 1.0) for r in valid)

    # boxed answer: need ALL models to agree on detected=True AND same verdict
    boxed_list = [r.get("boxed_answer") or {} for r in valid]
    all_detected = all(b.get("detected") for b in boxed_list)
    if all_detected:
        verdicts = [b.get("verdict", "").lower() for b in boxed_list]
        # unanimous verdict only
        unanimous = all(v == verdicts[0] for v in verdicts) and verdicts[0] in ("correct", "incorrect")
        consensus_boxed = {
            **boxed_list[0],
            "detected": True,
            "verdict": verdicts[0] if unanimous else "",   # empty = don't speak
        }
    else:
        consensus_boxed = {"detected": False, "value": "", "verdict": "", "reason": ""}

    boxed_wrong = consensus_boxed.get("verdict", "").lower() == "incorrect"

    # errors: strategy depends on whether boxed answer is confirmed wrong
    # If answer confirmed wrong → UNION (any model flagging = real error, we know mistake exists)
    # If answer not confirmed  → INTERSECTION (require both models to agree, conservative)
    def _step_key(e):
        # normalize "Step 1", "step1", "step 2:", "Step1" → "step1"
        s = e.get("step", "").lower().strip().rstrip(":")
        s = re.sub(r"\s+", "", s)  # remove spaces → "step1"
        return s

    error_maps = [{_step_key(e): e for e in r.get("errors", [])} for r in valid]

    if boxed_wrong:
        # UNION: any model that flags a step contributes it (answer confirmed wrong)
        merged = {}
        for em in error_maps:
            for k, e in em.items():
                if k not in merged:
                    merged[k] = e
        consensus_errors = [merged[k] for k in sorted(merged.keys())]
    elif error_maps:
        # INTERSECTION: only errors flagged by ALL models
        common_steps = set(error_maps[0].keys())
        for em in error_maps[1:]:
            common_steps &= set(em.keys())
        consensus_errors = [error_maps[0][k] for k in sorted(common_steps)]
    else:
        consensus_errors = []

    # score: average
    scores = [r.get("overall_score") for r in valid if r.get("overall_score") is not None]
    avg_score = round(sum(scores) / len(scores)) if scores else None

    return {
        "has_work":          hw,
        "question_detected": question,
        "confidence":        min_conf,
        "errors":            consensus_errors,
        "boxed_answer":      consensus_boxed,
        "overall_score":     avg_score,
        "all_good":          len(consensus_errors) == 0,
    }


def call_vision_api(b64: str, fast: bool = False) -> dict:
    """
    fast=True  → single fast model (periodic background scans, low latency).
    fast=False → two models in parallel, consensus merge (manual scan, accurate).
    """
    if fast:
        result = _call_single(b64, FAST_MODEL)
        print(f"  [fast/{FAST_MODEL.split('/')[-1]}]")
        return result

    # Parallel consensus: GPT-4o + Gemini Flash
    results = [None, None]
    def _t(idx, model):
        results[idx] = _call_single(b64, model)
        name = model.split("/")[-1]
        errs = results[idx].get("errors", []) if results[idx] else []
        boxed = (results[idx].get("boxed_answer") or {}) if results[idx] else {}
        print(f"  [{name}] errors={len(errs)} boxed={boxed.get('detected')} verdict={boxed.get('verdict','')}")

    threads = [
        threading.Thread(target=_t, args=(0, VISION_MODEL),    daemon=True),
        threading.Thread(target=_t, args=(1, CONSENSUS_MODEL), daemon=True),
    ]
    for t in threads: t.start()
    for t in threads: t.join(timeout=50)

    merged = _merge_consensus(results)
    print(f"  [consensus] errors={len(merged.get('errors',[]))} boxed_verdict={merged.get('boxed_answer',{}).get('verdict','')}")
    return merged


def call_tutor_api(history: list, max_tokens: int = 400) -> str:
    """Multi-turn Socratic tutor call."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    messages = [{"role": "system", "content": TUTOR_SYSTEM}] + history
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {key}",
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
        # Delay start so window fully renders before we insert text
        self.win.after(150, self._start)

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
        try:
            self.win.update_idletasks()
        except Exception:
            pass

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
        try:
            self.win.update_idletasks()
        except Exception:
            pass

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


# ── Audio alerts (macOS) ──────────────────────────────────────────────────────

_speak_lock = threading.Lock()

def play_error_sound():
    """Play system error sound (non-blocking)."""
    if not IS_MAC:
        return
    sounds = [
        "/System/Library/Sounds/Sosumi.aiff",
        "/System/Library/Sounds/Basso.aiff",
    ]
    for s in sounds:
        if os.path.exists(s):
            subprocess.Popen(["afplay", s],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return

def speak(text: str):
    """Speak text aloud via macOS `say` (non-blocking, queued)."""
    if not IS_MAC:
        return
    def _run():
        with _speak_lock:
            subprocess.run(["say", "-v", "Samantha", text],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    threading.Thread(target=_run, daemon=True).start()


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

    # errors are shown on the frontend dashboard — not overlaid on video
    errors = state.get("errors", [])

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

def live_watch(api_key: str, cam_index: int = -1):
    global _API_KEY
    _API_KEY = api_key
    # ── camera ──
    cap = None
    USB_KEYWORDS = ["emeet", "smartcam", "logitech", "elgato", "uvc", "external",
                    "c920", "c960", "c922", "brio", "facecam"]

    def _open_cam(idx: int):
        c = cv2.VideoCapture(idx)
        if not c.isOpened():
            return None
        # Give camera time to warm up — iPhone Continuity Camera needs ~2s
        deadline = time.time() + 3.0
        ret = False
        while time.time() < deadline:
            ret, _ = c.read()
            if ret:
                break
            time.sleep(0.1)
        if not ret:
            c.release()
            return None
        # Flush a few more frames
        for _ in range(10):
            c.read(); time.sleep(0.1)
        return c

    if cam_index >= 0:
        cap = _open_cam(cam_index)
        if cap is None:
            print(f"ERROR: Camera {cam_index} not found or unreadable.")
            sys.exit(1)
        print(f"Camera ready (index {cam_index})")
    else:
        # Prefer external cameras: iPhone Continuity (2), EMEET (1), built-in (0)
        for idx in [2, 1, 0, 3, 4]:
            c = _open_cam(idx)
            if c is not None:
                cap = c
                print(f"Camera ready (index {idx})")
                break
        if cap is None:
            print("ERROR: No camera found.")
            sys.exit(1)

    # ── tkinter root (hidden — just hosts popups) ──
    root = tk.Tk()
    root.withdraw()
    if IS_MAC:
        try:
            root.tk.call("::tk::unsupported::MacWindowStyle",
                         "style", root._w, "floating", "noActivates")
        except Exception:
            pass

    # ── hand tracker (shared camera) ──
    hand = None
    if HAND_TRACKER_OK:
        hand = HandTracker()
        hand.start_shared()
        print("[HandTracker] Initializing (model loading)...")
    else:
        print("[HandTracker] Not available (mediapipe/opencv missing).")

    # ── dashboard server ──
    if SERVER_OK:
        _server.start_server()

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
    error_history        = []
    error_repeat_counts  = {}   # (step, found) -> int
    error_consec_counts  = {}   # (step, found) -> consecutive scan count (debounce)
    last_spoken_hash     = ""
    last_all_good        = False
    last_speech_snap     = None  # frame at last TTS — suppresses same-paper speech
    last_boxed_spoken    = ""    # verdict hash to avoid repeating boxed verdict
    pending_boxed_verdict = ""   # verdict seen last scan — must match twice to fire
    pending_boxed_data    = {}
    active_errors        = []
    active_question      = ""
    repeated_errors      = []
    locked_question      = ""    # question locked once detected with high confidence
    locked_question_conf = 0.0   # confidence at lock time
    question_done        = False  # True after boxed-correct; resets for next question

    def _work_hash(question: str, errors: list) -> str:
        """Fingerprint of current detected work — changes when errors or question change."""
        parts = question + "|" + ",".join(
            sorted(f"{e.get('step','')}:{e.get('found','')}" for e in errors)
        )
        return hashlib.md5(parts.encode()).hexdigest()

    def _frame_changed(snap_new) -> bool:
        """True if the paper image has visibly changed since last speech (>3% pixels differ)."""
        if last_speech_snap is None:
            return True
        a = cv2.resize(last_speech_snap, (80, 45)).astype(float)
        b = cv2.resize(snap_new, (80, 45)).astype(float)
        diff = np.abs(a - b).max(axis=2)           # per-pixel max channel diff
        changed_frac = np.mean(diff > 25)          # fraction of pixels that changed a lot
        return changed_frac > 0.03                 # >3% pixels changed = new writing

    stuck_tracker    = StuckTracker()
    last_frame       = None
    last_scan_at     = 0.0
    last_periodic_at = time.time()
    still_since      = None
    still_triggered  = False
    force_snap       = False
    popup_ref        = None
    last_popup_at    = 0.0

    # session timing
    session_start      = time.time()
    session_working    = 0
    session_distracted = 0
    last_sec_at        = time.time()
    last_server_push   = 0.0

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

    SPEAK_COOLDOWN = 90.0   # seconds between speaking the same error key

    def run_analysis(snap, label="", fast=False):
        nonlocal active_errors, active_question, repeated_errors, last_spoken_hash, last_all_good, last_speech_snap, last_boxed_spoken, locked_question, locked_question_conf, question_done, pending_boxed_verdict, pending_boxed_data
        if is_blank_frame(snap):
            with lock:
                state.update(analyzing=False, blank=True, score=None,
                             status="No paper detected",
                             status_color=(120,120,120))
            return

        with lock:
            state.update(analyzing=True, blank=False,
                         status="Analyzing...", status_color=(0,215,255))

        result = call_vision_api(encode_frame(snap), fast=fast)
        print_result(result, label)

        errors  = result.get("errors", [])
        boxed   = result.get("boxed_answer", {})
        score   = result.get("overall_score")
        q       = result.get("question_detected", "")
        conf    = result.get("confidence", 1.0)
        hw      = result.get("has_work", True)

        # ── Question locking ─────────────────────────────────────────────────────
        # Lock the question text once detected with >= 0.75 confidence.
        # Never change it until a correct boxed answer resets for the next question.
        if q and conf >= 0.75 and not locked_question:
            locked_question      = q
            locked_question_conf = conf
            print(f"  [Question locked] '{locked_question}' (conf={conf:.2f})")
        elif q and conf >= locked_question_conf + 0.15 and not question_done:
            # Replace if significantly more confident (edge case)
            locked_question      = q
            locked_question_conf = conf

        display_question = locked_question or q

        active_errors   = errors if hw else []
        active_question = display_question

        ts    = datetime.now().strftime("%H:%M:%S")
        now_t = time.time()

        # ── Frame-change gate — suppress ALL TTS if paper hasn't visibly changed ──
        paper_changed = _frame_changed(snap)

        # ── Error debouncing: only surface an error after it appears in 2+ consecutive scans ──
        current_error_keys = {(e.get("step",""), e.get("found","")) for e in errors}
        # Increment consecutive count for errors seen this scan
        for key in current_error_keys:
            error_consec_counts[key] = error_consec_counts.get(key, 0) + 1
        # Reset count for errors NOT seen this scan (they disappeared)
        for key in list(error_consec_counts.keys()):
            if key not in current_error_keys:
                error_consec_counts[key] = 0

        # Only show errors that have appeared in >= 2 consecutive scans
        confirmed_errors = [e for e in errors
                            if error_consec_counts.get((e.get("step",""), e.get("found","")), 0) >= 2]

        # Track history for first confirmed appearance
        for e in confirmed_errors:
            key = (e.get("step",""), e.get("found",""))
            count = error_repeat_counts.get(key, 0) + 1
            error_repeat_counts[key] = count
            if count == 1:
                error_history.append({**e, "ts": ts})
            if count >= 2:
                repeated_errors = [r for r in repeated_errors
                                   if (r.get("step"), r.get("found")) != key]
                repeated_errors.append({**e, "repeat_count": count, "ts": ts})

        # Remove repeated errors that are no longer confirmed
        confirmed_keys = {(e.get("step",""), e.get("found","")) for e in confirmed_errors}
        repeated_errors = [r for r in repeated_errors
                           if (r.get("step",""), r.get("found","")) in confirmed_keys]

        # ── Boxed-answer: require SAME verdict on 2 consecutive scans before speaking ──
        if hw and boxed.get("detected"):
            verdict   = (boxed.get("verdict") or "").lower()
            boxed_key = f"{verdict}:{boxed.get('value','')}"
            if verdict and verdict == pending_boxed_verdict:
                # Confirmed — same verdict two scans in a row, speak it
                if boxed_key != last_boxed_spoken and paper_changed:
                    last_boxed_spoken = boxed_key
                    last_speech_snap  = snap.copy()
                    if verdict == "correct":
                        question_done        = True
                        locked_question      = ""
                        locked_question_conf = 0.0
                        speak("Answer correct! Well done. Move on to the next question.")
                    elif verdict == "incorrect":
                        if confirmed_errors:
                            bad_step = confirmed_errors[0].get("step", "your work")
                            speak(f"Boxed answer is not right. Please check {bad_step}.")
                        else:
                            speak("Boxed answer doesn't look right. Double-check your steps.")
            else:
                # First time seeing this verdict — wait for next scan to confirm
                pending_boxed_verdict = verdict
                pending_boxed_data    = boxed
        else:
            pending_boxed_verdict = ""
            pending_boxed_data    = {}

        last_all_good = (hw and not confirmed_errors)

        # Play chime (no speech) when new confirmed errors appear — visual + sound only
        work_hash    = _work_hash(display_question, confirmed_errors)
        work_changed = (work_hash != last_spoken_hash) and paper_changed
        if confirmed_errors and work_changed:
            last_spoken_hash = work_hash
            last_speech_snap = snap.copy()
            play_error_sound()   # chime only — no speech, user presses SPACE to hear it

        with lock:
            state["analyzing"] = False
            state["blank"]     = not hw
            if not hw:
                state.update(status="No paper detected", status_color=(120,120,120), score=None)
                return
            if display_question:
                state["question"] = display_question
            if result.get("subject"):
                state["subject"] = result["subject"].lower()
            state["score"]        = score if conf >= 0.5 and score is not None else None
            state["errors"]       = confirmed_errors
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

        # Feed to hand tracker (shares same frame, no extra camera needed)
        if hand is not None:
            hand.feed_frame(frame)

        now = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key in (ord('s'), ord('S')):
            force_snap = True
        if key == ord(' '):   # SPACE = speak current errors on demand
            errs = list(active_errors)
            q_now = active_question
            def _speak_on_demand(errs, q_now):
                if not errs:
                    speak("All steps look correct so far." if q_now else "No errors detected.")
                else:
                    for e in errs:
                        speak(f"Please check {e.get('step','that step')}. {e.get('hint','')}")
                        time.sleep(0.6)
            threading.Thread(target=_speak_on_demand, args=(errs, q_now), daemon=True).start()

        # ── scan cycle ──
        if now - last_scan_at >= SCAN_INTERVAL:
            last_scan_at = now

            with lock:
                analyzing  = state["analyzing"]
                popup_open = state["popup_open"]

            # pencil/hand position — prefer MediaPipe landmarks (precise),
            # fall back to frame-diff centroid when no hand is detected
            centroid = None
            if hand is not None:
                centroid = hand.get_hand_position()   # (cx,cy) normalized or None
            if centroid is None and last_frame is not None:
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
                            target=run_analysis, args=(snap, "still"),
                            kwargs={"fast": False}, daemon=True
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
                        target=run_analysis, args=(snap, "periodic"),
                        kwargs={"fast": False}, daemon=True
                    ).start()

        # ── draw ──
        display = frame.copy()
        with lock:
            s = dict(state)
        display = draw_overlay(display, s)

        # fidget HUD (top-right corner) — composite + sub-scores
        if hand is not None:
            fs = hand.get_fidget_scores()
            if fs.get("cam_ok"):
                fidget = fs["fidget"]
                fh, fw = display.shape[:2]
                # background panel
                panel_w, panel_h = 175, 130
                px = fw - panel_w - 4
                py = 4
                cv2.rectangle(display, (px, py), (fw - 4, py + panel_h), (20,20,20), -1)
                cv2.rectangle(display, (px, py), (fw - 4, py + panel_h), (60,60,60), 1)

                # composite bar
                col = (80,220,80) if fidget < 40 else (0,200,220) if fidget < 70 else (60,60,220)
                cv2.putText(display, f"Fidget: {fidget:.0f}", (px+6, py+18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2, cv2.LINE_AA)
                bar_w = int((panel_w - 12) * fidget / 100)
                cv2.rectangle(display, (px+6, py+22), (px + panel_w - 6, py+27), (50,50,50), -1)
                cv2.rectangle(display, (px+6, py+22), (px+6+bar_w, py+27), col, -1)

                # sub-scores
                subs = [
                    ("Tap",  fs["tapping"]),
                    ("Vel",  fs["velocity"]),
                    ("Tens", fs["tension"]),
                    ("Rest", fs["restlessness"]),
                ]
                for i, (lbl, val) in enumerate(subs):
                    row_y = py + 44 + i * 22
                    sub_col = (80,220,80) if val < 35 else (0,200,220) if val < 65 else (60,80,220)
                    cv2.putText(display, f"{lbl}:{val:4.0f}", (px+6, row_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, sub_col, 1, cv2.LINE_AA)
                    sb_w = int(60 * val / 100)
                    cv2.rectangle(display, (px+68, row_y-8), (px+128, row_y-4), (45,45,45), -1)
                    cv2.rectangle(display, (px+68, row_y-8), (px+68+sb_w, row_y-4), sub_col, -1)

                # hand detected indicator
                hand_pos = hand.get_hand_position()
                dot_col = (0,220,80) if hand_pos else (80,80,80)
                cv2.circle(display, (fw - 12, py + 8), 5, dot_col, -1)

        # ── session timing (1 Hz) ──
        if now - last_sec_at >= 1.0:
            last_sec_at = now
            hp = hand.get_hand_position() if hand else None
            is_active = hp is not None or centroid is not None
            if is_active:
                session_working += 1
            else:
                session_distracted += 1

        # ── push to dashboard (2 Hz) ──
        if SERVER_OK and now - last_server_push >= 0.5:
            last_server_push = now
            with lock:
                _s = dict(state)
            _fs = hand.get_fidget_scores() if hand else {}
            _fid    = _fs.get("fidget", 0.0)
            _errs   = _s.get("errors", [])
            _stuck  = _s.get("stuck_time", 0.0)
            _cam_ok = _fs.get("cam_ok", False)

            def _focus(errs, stk, fid):
                sc = 100 - min(32, len(errs)*8)
                if stk > 5: sc -= min(25, int(stk - 5))
                if fid > 50: sc -= min(15, int((fid - 50)*0.3))
                return max(0, sc)

            def _flow_state(errs, stk, fid, cam):
                if not cam: return "standby"
                if errs and stk > 8: return "stuck"
                if fid > 55 or (errs and stk > 3) or stk > 15: return "drift"
                return "flow"

            _server.push_state({
                "cam_ok":              _cam_ok,
                "status":              _s.get("status", ""),
                "status_state":        _flow_state(_errs, _stuck, _fid, _cam_ok),
                "focus_score":         _focus(_errs, _stuck, _fid),
                "score":               _s.get("score"),
                "question":            _s.get("question", ""),
                "errors":              _errs,
                "repeated_errors":     repeated_errors,
                "boxed_answer":        _s.get("boxed_answer", {}),
                "analyzing":           _s.get("analyzing", False),
                "blank":               _s.get("blank", False),
                "stuck_time":          _stuck,
                "popup_open":          _s.get("popup_open", False),
                "fidget":              float(_fid),
                "fidget_velocity":     float(_fs.get("velocity", 0.0)),
                "fidget_tapping":      float(_fs.get("tapping", 0.0)),
                "fidget_tension":      float(_fs.get("tension", 0.0)),
                "fidget_restlessness": float(_fs.get("restlessness", 0.0)),
                "next_in":             _s.get("next_in", 0.0),
                "session_active":      True,
                "working_seconds":     session_working,
                "distracted_seconds":  session_distracted,
                "session_seconds":     session_working + session_distracted,
                "error_history":       error_history[-20:],
            })
            _server.push_frame(display)

        cv2.imshow("Vision Tutor  (S = scan  |  Q = quit)", display)

        # pump tkinter events
        try:
            root.update()
        except tk.TclError:
            break

    cap.release()
    if hand is not None:
        hand.stop()
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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=-1,
                   help="Camera index to use (default: auto-detect)")
    args = p.parse_args()
    live_watch(get_api_key(), cam_index=args.camera)
