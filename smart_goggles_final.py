#!/usr/bin/env python3
"""
Smart Goggles for Visually Impaired — Unified AI Navigation System
==================================================================
Final-year project: Raspberry Pi 4 (4 GB RAM)

Architecture
------------
┌─────────────────────────────────────────────────────────┐
│                    MainController                       │
│  ┌──────────────┐  ┌───────────┐  ┌──────────────────┐ │
│  │ YOLODetector │  │ SharedCam │  │  VisionAssistant │ │
│  │  (Thread 1)  │◄─┤  (Lock)   ├─►│   (Thread 2)     │ │
│  └──────┬───────┘  └───────────┘  └────────┬─────────┘ │
│         │                                   │           │
│         ▼                                   ▼           │
│  ┌──────────────┐               ┌──────────────────┐   │
│  │ AudioManager │◄──────────────┤  SpeechListener  │   │
│  │  (TTS queue) │               │  (Vosk / wake)   │   │
│  └──────────────┘               └──────────────────┘   │
└─────────────────────────────────────────────────────────┘

Threads
-------
1. **YOLO thread** — runs non-stop, grabs frames, detects obstacles,
   runs pothole detection, pushes alerts into the speech queue.
2. **VQA thread** — listens for wake-word "hello", records a question,
   grabs one frame, calls vision LLM, speaks the answer.
3. **TTS thread** — drains the speech queue; YOLO alerts are
   interruptible, LLM answers are not.
4. **Main thread** — starts everything, waits for Ctrl-C.

Camera sharing
--------------
A single `SharedCamera` instance wraps an ESP32-CAM MJPEG stream via
OpenCV VideoCapture.  A background thread keeps the buffer drained so
YOLO and VQA always get the freshest frame.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import textwrap
import threading
import time
import traceback
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ──────────────────────────────────────────────
# Third-party imports with graceful fallback
# ──────────────────────────────────────────────
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None  # type: ignore

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore

try:
    import requests
except ImportError:
    requests = None  # type: ignore

try:
    import pyaudio
except ImportError:
    pyaudio = None  # type: ignore

try:
    from vosk import Model as VoskModel, KaldiRecognizer
except ImportError:
    VoskModel = None  # type: ignore
    KaldiRecognizer = None  # type: ignore

# ╔══════════════════════════════════════════════════════════╗
# ║                   CONFIGURATION                         ║
# ╠══════════════════════════════════════════════════════════╣

# ── Camera ────────────────────────────────────────────────
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480

# ── YOLO Obstacle Detection ──────────────────────────────
YOLO_MODEL_NAME: str = "yolov8n.pt"
YOLO_PROCESS_WIDTH: int = 320
YOLO_PROCESS_HEIGHT: int = 240
YOLO_CONFIDENCE: float = 0.40
YOLO_IOU_THRESHOLD: float = 0.45
YOLO_FRAME_INTERVAL: float = 0.25
YOLO_ALERT_COOLDOWN: float = 5.0
YOLO_CLOSE_DISTANCE_M: float = 1.5
YOLO_MEDIUM_DISTANCE_M: float = 3.0

CRITICAL_CLASSES: dict[int, str] = {
    0: "person",   1: "bicycle",   2: "car",       3: "motorcycle",
    5: "bus",      7: "truck",     9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 13: "bench",  15: "cat",      16: "dog",
    56: "chair",   57: "couch",    58: "potted plant", 59: "bed",
    60: "dining table", 62: "tv",  63: "laptop",   67: "cell phone",
    73: "book",    74: "clock",    24: "backpack",  25: "umbrella",
    28: "suitcase",
}

OBJECT_REAL_WIDTHS_M: dict[str, float] = {
    "person": 0.5,    "bicycle": 0.6,  "car": 1.8,     "motorcycle": 0.8,
    "bus": 2.5,       "truck": 2.5,    "traffic light": 0.3, "fire hydrant": 0.3,
    "stop sign": 0.6, "bench": 1.5,    "cat": 0.25,    "dog": 0.4,
    "chair": 0.5,     "couch": 1.8,    "potted plant": 0.3, "bed": 1.5,
    "dining table": 1.2, "tv": 0.8,   "laptop": 0.35,  "cell phone": 0.08,
    "book": 0.2,      "clock": 0.3,    "backpack": 0.4, "umbrella": 1.0,
    "suitcase": 0.5,
}

ESTIMATED_FOCAL_PX: float = YOLO_PROCESS_WIDTH / (2.0 * np.tan(np.radians(30)))

# ── Pothole Detection ─────────────────────────────────────
# Fraction of frame height from the top to start the ground-plane crop.
# 0.6 = analyse the bottom 40% of the frame (the ground ahead).
POTHOLE_GROUND_CROP_RATIO: float = 0.60

# Laplacian energy ratio (lower_strip / upper_strip) above which a
# surface hazard is announced.  Tune on your camera / surface.
POTHOLE_RATIO_THRESHOLD: float = 1.8   # conservative — raise to reduce false positives
POTHOLE_HIGH_RATIO: float = 2.8        # above this → "Warning!" instead of "Caution."

# Seconds before the same column (left/centre/right) can be announced again.
POTHOLE_ALERT_COOLDOWN: float = 6.0

# ── Vision LLM (VQA) ─────────────────────────────────────
LLM_API_URL: str = "https://openrouter.ai/api/v1/chat/completions"
LLM_API_KEY_ENV: str = "OPENROUTER_API_KEY"
LLM_MODEL: str = "openai/gpt-4o-mini"
LLM_MAX_TOKENS: int = 200
LLM_TEMPERATURE: float = 0.3
LLM_TIMEOUT_S: int = 20
LLM_IMAGE_WIDTH: int = 384
LLM_IMAGE_HEIGHT: int = 288
LLM_JPEG_QUALITY: int = 30

# ── Wake-word & Speech Recognition ────────────────────────
VOSK_MODEL_PATH: str = os.path.expanduser("~/vosk-model-small-en-us-0.15")
WAKE_WORD: str = "hello"
LISTEN_TIMEOUT_S: float = 7.0
SILENCE_TIMEOUT_S: float = 2.0

# ── Text-to-Speech ────────────────────────────────────────
TTS_RATE: int = 160
TTS_VOLUME: float = 1.0

# ── Misc / Logging ────────────────────────────────────────
LOG_LEVEL: str = "INFO"
LOG_FILE: Optional[str] = "smart_goggles.log"

# ╚══════════════════════════════════════════════════════════╝

# ── Logger setup ──────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if LOG_FILE:
        handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                        format=fmt, handlers=handlers)
    return logging.getLogger("SmartGoggles")

log = _setup_logging()


# ════════════════════════════════════════════════════════════
#  DATA TYPES
# ════════════════════════════════════════════════════════════
@dataclass
class Detection:
    """One detected object with spatial info."""
    label: str
    confidence: float
    distance_m: float
    clock_position: int
    bbox: Tuple[int, int, int, int]

@dataclass(order=True)
class SpeechItem:
    """Item queued for TTS.  Lower priority number = spoken first."""
    priority: int
    text: str = field(compare=False)
    interruptible: bool = field(default=True, compare=False)


# ════════════════════════════════════════════════════════════
#  SHARED CAMERA  (ESP32-CAM over Wi-Fi MJPEG stream)
# ════════════════════════════════════════════════════════════
import requests as _requests_module  # ensure available for SharedCamera

class SharedCamera:
    """Thread-safe camera wrapper for Raspberry Pi 4 to stream from ESP32-CAM."""

    def __init__(self) -> None:
        # --- UPDATE THIS IP TO YOUR ESP32 IP FROM SERIAL MONITOR ---
        self.esp32_ip = "10.162.224.83"
        self.stream_url = f"http://{self.esp32_ip}:81/stream"
        # -----------------------------------------------------------

        self._cap = None
        self._lock = threading.Lock()
        self._alive = True
        self.latest_frame = None

        # Background thread that constantly drains the Wi-Fi buffer
        self._t = threading.Thread(target=self._update_loop, daemon=True,
                                   name="CamStream")
        self._t.start()
        log.info("Connecting to ESP32-CAM stream at %s", self.stream_url)

    def _update_loop(self) -> None:
        """Internal loop to keep the buffer empty and capture the newest frame."""
        while self._alive:
            if self._cap is None or not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.stream_url)
                time.sleep(2)
                continue

            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self.latest_frame = frame
            else:
                log.warning("Stream disconnected. Retrying...")
                self._cap.release()
                self._cap = None
                time.sleep(1)

    def read(self) -> Optional[np.ndarray]:
        """Returns the most recent frame captured by the background thread."""
        with self._lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def grab_for_llm(self) -> Optional[np.ndarray]:
        """Captures a frame and resizes it for the Vision LLM."""
        frame = self.read()
        if frame is None:
            return None
        return cv2.resize(frame, (LLM_IMAGE_WIDTH, LLM_IMAGE_HEIGHT),
                          interpolation=cv2.INTER_AREA)

    def release(self) -> None:
        """Stops the thread and releases the stream."""
        self._alive = False
        if self._cap:
            self._cap.release()


# ════════════════════════════════════════════════════════════
#  TEXT-TO-SPEECH MANAGER (thread-safe queue)
# ════════════════════════════════════════════════════════════
class AudioManager:
    """Serialises all speech output through a single background thread.

    * YOLO / pothole alerts are *interruptible* — a newer alert can
      replace a queued-but-not-yet-spoken one.
    * LLM answers and system messages are *not* interruptible.
    """

    PRIORITY_SYSTEM = 0
    PRIORITY_LLM = 1
    PRIORITY_YOLO = 2

    def __init__(self) -> None:
        self._queue: queue.PriorityQueue[SpeechItem] = queue.PriorityQueue()
        self._alive = True
        self._engine_lock = threading.Lock()
        self._engine: Any = None
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="TTS")
        self._init_engine()
        self._thread.start()

    def say_system(self, text: str) -> None:
        self._enqueue(text, self.PRIORITY_SYSTEM, interruptible=False)

    def say_llm(self, text: str) -> None:
        self._enqueue(text, self.PRIORITY_LLM, interruptible=False)

    def say_yolo(self, text: str) -> None:
        self._enqueue(text, self.PRIORITY_YOLO, interruptible=True)

    def shutdown(self) -> None:
        self._alive = False
        self._queue.put(SpeechItem(priority=-1, text="", interruptible=True))
        self._thread.join(timeout=5)
        with self._engine_lock:
            if self._engine is not None:
                try:
                    self._engine.stop()
                except Exception:
                    pass

    def _init_engine(self) -> None:
        if pyttsx3 is None:
            log.warning("pyttsx3 not installed — will use espeak CLI fallback")
            return
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", TTS_RATE)
            engine.setProperty("volume", TTS_VOLUME)
            voices = engine.getProperty("voices")
            for v in voices:
                if "english" in v.name.lower():
                    engine.setProperty("voice", v.id)
                    break
            self._engine = engine
            log.info("pyttsx3 TTS engine initialised (rate=%d)", TTS_RATE)
        except Exception as exc:
            log.warning("pyttsx3 init failed (%s) — using espeak fallback", exc)

    def _enqueue(self, text: str, priority: int, interruptible: bool) -> None:
        self._queue.put(SpeechItem(priority=priority, text=text,
                                   interruptible=interruptible))

    def _speak(self, text: str) -> None:
        if not text:
            return
        log.debug("TTS ▶ %s", text[:80])
        with self._engine_lock:
            if self._engine is not None:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                    return
                except Exception as exc:
                    log.warning("pyttsx3 speak failed (%s), trying espeak", exc)
        try:
            subprocess.run(
                ["espeak", "-s", str(TTS_RATE), "--", text],
                timeout=30, check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            log.error("Neither pyttsx3 nor espeak available — cannot speak")
        except Exception as exc:
            log.error("espeak failed: %s", exc)

    def _run(self) -> None:
        while self._alive:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item.priority == -1:
                break
            if item.interruptible and not self._queue.empty():
                log.debug("Skipping stale alert: %s", item.text[:60])
                continue
            self._speak(item.text)


# ════════════════════════════════════════════════════════════
#  POTHOLE DETECTOR
# ════════════════════════════════════════════════════════════
class PotholeDetector:
    """Ground-plane Laplacian surface-hazard detector.

    Called from ``YOLODetector._detect_once()`` on the same resized frame
    used for YOLO inference — no extra resize overhead.

    Algorithm
    ---------
    1. Crop the bottom (100 - POTHOLE_GROUND_CROP_RATIO*100) % of the
       frame to isolate the ground plane directly ahead.
    2. Convert to greyscale and compute the absolute Laplacian (edge
       energy proxy).
    3. Divide the crop into a 3-column × 2-row grid (left/centre/right
       × upper-strip/lower-strip).
    4. If lower_energy / upper_energy exceeds POTHOLE_RATIO_THRESHOLD in
       any column, a surface hazard is announced.  The intuition is that
       a pothole or raised kerb creates a sudden spike in edge energy
       close to the camera compared with the smooth road further ahead.
    """

    def __init__(self, audio: AudioManager) -> None:
        self._audio = audio
        # Cooldown tracker: column_key → last_announcement_time
        self._cooldowns: dict[str, float] = {}
        log.info("PotholeDetector initialised (ground-plane Laplacian method)")

    def analyze(self, frame: np.ndarray) -> None:
        """Run pothole analysis on *frame* and announce if a hazard is detected.

        Parameters
        ----------
        frame:
            BGR frame at YOLO_PROCESS_WIDTH × YOLO_PROCESS_HEIGHT
            (already resized by the YOLO thread — no extra resize needed).
        """
        try:
            self._analyze_impl(frame)
        except Exception:
            log.debug("PotholeDetector error:\n%s", traceback.format_exc())

    def _analyze_impl(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]

        # ── 1. Crop lower portion (ground plane) ──────────
        crop_top = int(h * POTHOLE_GROUND_CROP_RATIO)
        ground = frame[crop_top:h, 0:w]
        if ground.shape[0] < 10:
            return

        # ── 2. Greyscale + Laplacian ───────────────────────
        gray = cv2.cvtColor(ground, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_abs = np.abs(lap)

        # ── 3. Three-column grid ───────────────────────────
        cols = [
            ("left",   0,           w // 3),
            ("centre", w // 3,      2 * w // 3),
            ("right",  2 * w // 3,  w),
        ]
        gh = lap_abs.shape[0]
        mid = gh // 2

        for col_name, x0, x1 in cols:
            strip = lap_abs[:, x0:x1]
            upper_energy = float(strip[:mid, :].mean())
            lower_energy = float(strip[mid:, :].mean())

            if upper_energy < 1e-3:
                continue

            ratio = lower_energy / upper_energy

            if ratio > POTHOLE_RATIO_THRESHOLD:
                clock = {"left": 10, "centre": 12, "right": 2}[col_name]
                self._maybe_announce(col_name, clock, ratio)

        # Prune stale cooldowns periodically
        if len(self._cooldowns) > 50:
            now = time.monotonic()
            self._cooldowns = {
                k: v for k, v in self._cooldowns.items()
                if now - v < POTHOLE_ALERT_COOLDOWN * 2
            }

    def _maybe_announce(self, col_name: str, clock: int,
                        ratio: float) -> None:
        """Announce hazard if cooldown for this column has expired."""
        now = time.monotonic()
        key = f"pothole_{col_name}"
        if now - self._cooldowns.get(key, 0.0) < POTHOLE_ALERT_COOLDOWN:
            return
        self._cooldowns[key] = now

        urgency = "Warning!" if ratio > POTHOLE_HIGH_RATIO else "Caution."
        direction = f"{clock} o'clock"
        msg = (f"{urgency} Surface hazard ahead at {direction}. "
               "Watch your step.")
        log.info("Pothole alert: %s (ratio=%.2f)", msg, ratio)
        self._audio.say_yolo(msg)


# ════════════════════════════════════════════════════════════
#  YOLO OBSTACLE DETECTOR (Thread 1 — never stops)
# ════════════════════════════════════════════════════════════
class YOLODetector:
    """Continuous YOLO-based obstacle detection with monocular distance
    estimation and clock-position spatial audio alerts.

    Also runs ``PotholeDetector`` on every frame at zero extra cost
    (reuses the resized frame already produced for YOLO inference).

    Runs in its own thread.
    """

    def __init__(self, camera: SharedCamera, audio: AudioManager) -> None:
        self._camera = camera
        self._audio = audio
        self._alive = True
        self._model: Any = None
        self._cooldowns: dict[str, float] = {}
        # Pause gate: when *set* YOLO runs; when *cleared* YOLO waits.
        self._run_gate = threading.Event()
        self._run_gate.set()
        # Pothole detector shares the same audio manager
        self._pothole = PotholeDetector(audio)
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="YOLO")
        self._load_model()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._alive = False
        self._run_gate.set()
        self._thread.join(timeout=10)

    def pause(self) -> None:
        """Pause detection (called by VQA when handling a query)."""
        log.info("YOLO paused (VQA active)")
        self._run_gate.clear()

    def resume(self) -> None:
        """Resume detection (called by VQA after query completes)."""
        log.info("YOLO resumed")
        self._run_gate.set()

    def _load_model(self) -> None:
        if YOLO is None:
            log.error("ultralytics not installed — YOLO disabled")
            return
        try:
            log.info("Loading YOLO model '%s' …", YOLO_MODEL_NAME)
            self._model = YOLO(YOLO_MODEL_NAME)
            dummy = np.zeros((YOLO_PROCESS_HEIGHT, YOLO_PROCESS_WIDTH, 3),
                             dtype=np.uint8)
            self._model.predict(dummy, verbose=False,
                                conf=YOLO_CONFIDENCE,
                                iou=YOLO_IOU_THRESHOLD,
                                imgsz=YOLO_PROCESS_WIDTH)
            log.info("YOLO model loaded and warmed up")
        except Exception as exc:
            log.error("Failed to load YOLO model: %s", exc)
            self._model = None

    def _run(self) -> None:
        log.info("YOLO detection thread started")
        while self._alive:
            self._run_gate.wait()
            if not self._alive:
                break
            loop_start = time.monotonic()
            try:
                self._detect_once()
            except Exception:
                log.error("YOLO detection error:\n%s", traceback.format_exc())
            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.01, YOLO_FRAME_INTERVAL - elapsed))
        log.info("YOLO detection thread stopped")

    def _detect_once(self) -> None:
        """Grab one frame, run YOLO + pothole detection on it."""
        frame = self._camera.read()
        if frame is None:
            return

        small = cv2.resize(frame,
                           (YOLO_PROCESS_WIDTH, YOLO_PROCESS_HEIGHT),
                           interpolation=cv2.INTER_AREA)

        # ── YOLO obstacle detection ────────────────────────
        if self._model is not None:
            results = self._model.predict(
                small, verbose=False,
                conf=YOLO_CONFIDENCE,
                iou=YOLO_IOU_THRESHOLD,
                imgsz=YOLO_PROCESS_WIDTH,
            )
            detections = self._parse_results(results, small.shape[1])
            if detections:
                self._announce(detections)

        # ── Pothole detection (reuses same resized frame) ──
        self._pothole.analyze(small)

    def _parse_results(self, results: Any,
                       frame_width: int) -> list[Detection]:
        dets: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in CRITICAL_CLASSES:
                    continue
                label = CRITICAL_CLASSES[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                dist = self._estimate_distance(label, x2 - x1)
                clock = self._pixel_to_clock((x1 + x2) / 2.0, frame_width)
                dets.append(Detection(
                    label=label, confidence=conf,
                    distance_m=round(dist, 1),
                    clock_position=clock,
                    bbox=(x1, y1, x2, y2),
                ))
        dets.sort(key=lambda d: d.distance_m)
        return dets

    @staticmethod
    def _estimate_distance(label: str, pixel_width: int) -> float:
        real_w = OBJECT_REAL_WIDTHS_M.get(label, 0.5)
        if pixel_width < 1:
            pixel_width = 1
        return (real_w * ESTIMATED_FOCAL_PX) / pixel_width

    @staticmethod
    def _pixel_to_clock(center_x: float, frame_width: int) -> int:
        ratio = center_x / max(frame_width, 1)
        clock_raw = 9 + ratio * 6
        clock = int(round(clock_raw))
        if clock > 12:
            clock -= 12
        return max(1, min(12, clock))

    def _announce(self, detections: list[Detection]) -> None:
        now = time.monotonic()
        parts: list[str] = []
        for d in detections[:3]:
            key = f"{d.label}_{d.clock_position}"
            if now - self._cooldowns.get(key, 0.0) < YOLO_ALERT_COOLDOWN:
                continue
            self._cooldowns[key] = now

            if d.distance_m <= YOLO_CLOSE_DISTANCE_M:
                urgency = "Warning!"
                dist_word = f"{d.distance_m:.1f} meters"
            elif d.distance_m <= YOLO_MEDIUM_DISTANCE_M:
                urgency = "Caution."
                dist_word = f"{d.distance_m:.1f} meters"
            else:
                urgency = ""
                dist_word = f"about {d.distance_m:.0f} meters"

            direction = f"{d.clock_position} o'clock"
            msg = f"{urgency} {d.label} at {direction}, {dist_word}."
            parts.append(msg.strip())

        if parts:
            full = " ".join(parts)
            log.info("YOLO alert: %s", full)
            self._audio.say_yolo(full)

        if len(self._cooldowns) > 200:
            cutoff = now - YOLO_ALERT_COOLDOWN * 2
            self._cooldowns = {k: v for k, v in self._cooldowns.items()
                               if v > cutoff}


# ════════════════════════════════════════════════════════════
#  PROMPT ENGINEERING FOR BLIND USERS
# ════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are the AI assistant of a pair of smart goggles worn by a blind person.
    You see exactly what the camera on the goggles shows.
""")

_PROMPTS: dict[str, str] = {
    "safety": (
        "The user is worried about obstacles or safety. "
        "Describe any hazards, their clock position, and approximate distance. "
        "If nothing dangerous is visible, say it looks clear."
    ),
    "text": (
        "The user wants to read text in the image. "
        "Read any visible text — signs, labels, screens — verbatim. "
        "Mention where the text is (clock position). If no text, say so."
    ),
    "object": (
        "The user asks about a specific object. "
        "Identify it, give its clock position and distance. "
        "If the object is not visible, say you cannot see it."
    ),
    "people": (
        "The user asks about people in the scene. "
        "Mention how many people, their approximate positions (clock), "
        "and what they appear to be doing. No detailed appearance unless asked."
    ),
    "scene": (
        "The user wants a general scene description. "
        "Briefly describe the environment, key landmarks, and anything "
        "noteworthy for navigation."
    ),
    "color": (
        "The user asks about colours or visual appearance. "
        "Describe the relevant colours clearly and concisely."
    ),
    "general": (
        "Answer the user's question about what the camera sees. "
        "Prioritise safety-relevant information."
    ),
}

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "safety": ["safe", "danger", "obstacle", "block", "hazard", "clear",
               "walk", "cross", "step", "trip", "fall", "careful",
               "warning", "ahead", "path", "way"],
    "text":   ["read", "text", "sign", "label", "write", "written",
               "say", "says", "letter", "word", "screen", "display",
               "menu", "price", "name"],
    "object": ["what is", "what's", "identify", "recognize", "find",
               "where is", "where's", "thing", "object", "item"],
    "people": ["person", "people", "someone", "anyone", "man", "woman",
               "child", "kid", "crowd", "group", "who"],
    "scene":  ["describe", "scene", "around", "surroundings", "environment",
               "look like", "overview", "where am i", "location"],
    "color":  ["color", "colour", "bright", "dark", "light", "shade",
               "red", "blue", "green", "yellow", "white", "black"],
}


def classify_query(question: str) -> str:
    q = question.lower()
    scores: dict[str, int] = {cat: 0 for cat in _CATEGORY_KEYWORDS}
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                scores[cat] += 1
    best = max(scores, key=lambda c: scores[c])
    return "general" if scores[best] == 0 else best


# ════════════════════════════════════════════════════════════
#  VISION LLM ASSISTANT  (Thread 2 — wake-word driven)
# ════════════════════════════════════════════════════════════
class VisionAssistant:
    """Listens for wake-word → records question → calls vision LLM → speaks
    the answer.  Pauses YOLO for the duration of each query.
    """

    def __init__(self, camera: SharedCamera, audio: AudioManager,
                 yolo: Optional["YOLODetector"] = None) -> None:
        self._camera = camera
        self._audio = audio
        self._yolo = yolo
        self._alive = True
        self._api_key: str = os.environ.get(LLM_API_KEY_ENV, "")
        self._vosk_model: Any = None
        self._pa: Any = None
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="VQA")
        if not self._api_key:
            log.warning("Environment variable %s not set — VQA will be "
                        "disabled", LLM_API_KEY_ENV)
        self._init_vosk()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._alive = False
        self._thread.join(timeout=10)
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass

    def _init_vosk(self) -> None:
        if VoskModel is None or KaldiRecognizer is None:
            log.warning("Vosk not installed — wake-word detection disabled")
            return
        if not os.path.isdir(VOSK_MODEL_PATH):
            log.warning("Vosk model not found at '%s' — wake-word disabled. "
                        "Download from https://alphacephei.com/vosk/models",
                        VOSK_MODEL_PATH)
            return
        try:
            self._vosk_model = VoskModel(VOSK_MODEL_PATH)
            log.info("Vosk speech model loaded from '%s'", VOSK_MODEL_PATH)
        except Exception as exc:
            log.error("Failed to load Vosk model: %s", exc)
            self._vosk_model = None

    def _get_pyaudio(self) -> Any:
        if self._pa is None:
            if pyaudio is None:
                log.error("PyAudio not installed — cannot use microphone")
                return None
            try:
                self._pa = pyaudio.PyAudio()
            except Exception as exc:
                log.error("PyAudio init failed: %s", exc)
                return None
        return self._pa

    def _run(self) -> None:
        log.info("VQA assistant thread started")
        if self._vosk_model is None:
            log.error("VQA thread exiting — no Vosk model")
            return

        pa = self._get_pyaudio()
        if pa is None:
            log.error("VQA thread exiting — no PyAudio")
            return

        mic_stream: Any = None
        try:
            mic_stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4096,
            )
            mic_stream.start_stream()
            log.info("Microphone stream opened for wake-word detection")
        except Exception as exc:
            log.error("Failed to open microphone: %s", exc)
            return

        recognizer = KaldiRecognizer(self._vosk_model, 16000)

        while self._alive:
            try:
                data = mic_stream.read(4096, exception_on_overflow=False)
            except Exception as exc:
                log.warning("Mic read error: %s — retrying", exc)
                time.sleep(0.5)
                continue

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower().strip()
                if WAKE_WORD in text:
                    log.info("Wake word detected!")
                    self._handle_query(mic_stream, recognizer)
            else:
                partial = json.loads(recognizer.PartialResult())
                partial_text = partial.get("partial", "").lower()
                if WAKE_WORD in partial_text:
                    self._drain_until_final(mic_stream, recognizer)
                    log.info("Wake word detected (partial match)!")
                    self._handle_query(mic_stream, recognizer)

        try:
            mic_stream.stop_stream()
            mic_stream.close()
        except Exception:
            pass
        log.info("VQA assistant thread stopped")

    def _drain_until_final(self, stream: Any, rec: Any,
                           timeout: float = 1.5) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                data = stream.read(4096, exception_on_overflow=False)
            except Exception:
                break
            if rec.AcceptWaveform(data):
                break

    def _handle_query(self, mic_stream: Any, recognizer: Any) -> None:
        """Record the user's question, query the LLM, speak the answer.
        YOLO (including pothole detection) is paused for the duration.
        """
        if self._yolo is not None:
            self._yolo.pause()

        try:
            self._audio.say_system("Ready for your question.")
            log.info("Listening for question …")

            recognizer.Reset()
            question = self._listen_for_question(mic_stream, recognizer)
            if not question:
                self._audio.say_system("Sorry, I didn't catch that. "
                                       "Say hello again to try.")
                return

            log.info("Question: '%s'", question)
            self._audio.say_system("One moment.")

            frame = self._camera.grab_for_llm()
            if frame is None:
                self._audio.say_system("Camera is not available right now. "
                                       "Please try again.")
                return

            answer = self._query_llm(question, frame)
            if answer:
                log.info("LLM answer: %s", answer)
                self._audio.say_llm(answer)
            else:
                self._audio.say_system("Sorry, I couldn't get an answer. "
                                       "Please try again.")
        finally:
            if self._yolo is not None:
                self._yolo.resume()

    def _listen_for_question(self, stream: Any, rec: Any) -> str:
        deadline = time.monotonic() + LISTEN_TIMEOUT_S
        heard_speech = False
        last_speech_time = time.monotonic()
        full_text_parts: list[str] = []

        while time.monotonic() < deadline:
            try:
                data = stream.read(4096, exception_on_overflow=False)
            except Exception:
                break

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()
                if text:
                    heard_speech = True
                    last_speech_time = time.monotonic()
                    full_text_parts.append(text)
            else:
                partial = json.loads(rec.PartialResult())
                if partial.get("partial", "").strip():
                    heard_speech = True
                    last_speech_time = time.monotonic()

            if heard_speech and (time.monotonic() - last_speech_time
                                 > SILENCE_TIMEOUT_S):
                rec.AcceptWaveform(b"\x00\x00" * 2048)
                final = json.loads(rec.FinalResult())
                t = final.get("text", "").strip()
                if t:
                    full_text_parts.append(t)
                break

        question = " ".join(full_text_parts).strip()
        question = re.sub(r"\bhello\b", "", question, flags=re.IGNORECASE).strip()
        return question

    def _query_llm(self, question: str, frame: np.ndarray) -> Optional[str]:
        if not self._api_key:
            log.error("No API key — cannot call LLM")
            return None
        if requests is None:
            log.error("requests library not installed")
            return None

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, LLM_JPEG_QUALITY]
        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            log.error("JPEG encoding failed")
            return None
        b64_image = base64.b64encode(buf.tobytes()).decode("utf-8")
        log.debug("Image encoded: %.1f KB (%dx%d @ Q%d)",
                  len(buf) / 1024, LLM_IMAGE_WIDTH, LLM_IMAGE_HEIGHT,
                  LLM_JPEG_QUALITY)

        category = classify_query(question)
        user_prompt = _PROMPTS[category]
        log.debug("Query category: %s", category)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": LLM_MODEL,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{user_prompt}\n\nUser's question: {question}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                            },
                        },
                    ],
                },
            ],
        }

        for attempt in range(1, 4):
            try:
                log.debug("LLM API call (attempt %d) …", attempt)
                resp = requests.post(
                    LLM_API_URL, headers=headers,
                    json=payload, timeout=LLM_TIMEOUT_S,
                )
                if resp.status_code == 200:
                    body = resp.json()
                    answer = (body.get("choices", [{}])[0]
                              .get("message", {})
                              .get("content", "").strip())
                    if answer:
                        return answer
                    log.warning("LLM returned empty answer")
                    return None
                elif resp.status_code == 429:
                    log.warning("LLM rate-limited — waiting 3s")
                    time.sleep(3)
                    continue
                else:
                    log.error("LLM API error %d: %s",
                              resp.status_code, resp.text[:300])
                    return None
            except requests.exceptions.Timeout:
                log.warning("LLM API timeout (attempt %d)", attempt)
                continue
            except requests.exceptions.ConnectionError as exc:
                log.error("LLM API connection error: %s", exc)
                return None
            except Exception:
                log.error("LLM API unexpected error:\n%s",
                          traceback.format_exc())
                return None
        log.error("LLM API failed after 3 attempts")
        return None


# ════════════════════════════════════════════════════════════
#  MAIN CONTROLLER — ties everything together
# ════════════════════════════════════════════════════════════
class SmartGogglesController:
    """Top-level controller: initialises subsystems, manages lifecycle."""

    def __init__(self) -> None:
        self._alive = True
        self._camera: Optional[SharedCamera] = None
        self._audio: Optional[AudioManager] = None
        self._yolo: Optional[YOLODetector] = None
        self._vqa: Optional[VisionAssistant] = None

    def run(self) -> None:
        log.info("=" * 55)
        log.info("  Smart Goggles System — starting up")
        log.info("=" * 55)

        # ── 1. Audio ──────────────────────────────────────
        try:
            self._audio = AudioManager()
        except Exception as exc:
            log.critical("Failed to initialise audio: %s", exc)
            return
        self._audio.say_system("Smart goggles activated. Initialising camera.")

        # ── 2. Camera ─────────────────────────────────────
        try:
            self._camera = SharedCamera()
        except Exception as exc:
            log.critical("Camera init failed: %s", exc)
            self._audio.say_system("Camera initialisation failed. "
                                   "Please check the camera connection "
                                   "and restart.")
            time.sleep(5)
            self._shutdown()
            return

        test_frame = self._camera.read()
        if test_frame is None:
            self._audio.say_system("Camera connected but not responding. "
                                   "Will keep trying.")
        else:
            self._audio.say_system("Camera ready.")

        # ── 3. YOLO + Pothole detector ────────────────────
        try:
            self._yolo = YOLODetector(self._camera, self._audio)
            self._yolo.start()
            self._audio.say_system(
                "Obstacle and surface hazard detection active.")
        except Exception as exc:
            log.error("YOLO init failed: %s — running without detection", exc)
            self._audio.say_system("Obstacle detection failed to start.")

        # ── 4. VQA assistant ──────────────────────────────
        try:
            self._vqa = VisionAssistant(self._camera, self._audio,
                                        yolo=self._yolo)
            self._vqa.start()
            self._audio.say_system("Voice assistant ready. Say hello to ask "
                                   "a question.")
        except Exception as exc:
            log.error("VQA init failed: %s — running without assistant", exc)
            self._audio.say_system("Voice assistant failed to start.")

        log.info("All subsystems initialised — entering main loop")

        self._install_signal_handlers()
        try:
            while self._alive:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("Ctrl-C received")
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        log.info("Shutting down …")
        if self._audio:
            self._audio.say_system("Shutting down. Goodbye.")
            time.sleep(2)

        self._alive = False
        if self._vqa:
            log.info("Stopping VQA …")
            self._vqa.stop()
        if self._yolo:
            log.info("Stopping YOLO …")
            self._yolo.stop()
        if self._camera:
            log.info("Releasing camera …")
            self._camera.release()
        if self._audio:
            log.info("Stopping audio …")
            self._audio.shutdown()
        log.info("Shutdown complete")

    def _install_signal_handlers(self) -> None:
        def _handler(signum: int, frame: Any) -> None:
            log.info("Signal %d received — shutting down", signum)
            self._alive = False

        try:
            signal.signal(signal.SIGTERM, _handler)
            signal.signal(signal.SIGINT, _handler)
        except (OSError, ValueError):
            pass


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════
def main() -> None:
    controller = SmartGogglesController()
    controller.run()


if __name__ == "__main__":
    main()
