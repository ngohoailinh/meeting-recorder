import sys
import json
import threading
import queue
import time
import traceback
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import mss
import cv2
import scipy.io.wavfile as wavfile
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QStatusBar, QFileDialog,
    QComboBox, QGroupBox, QCheckBox, QMessageBox, QTabWidget,
    QListWidget, QListWidgetItem, QLineEdit, QSplitter,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QObject, QThread
from PyQt5.QtGui import QFont, QTextCursor, QColor


SAMPLE_RATE = 16000
CONFIG_PATH = Path.home() / ".meeting_recorder.json"
MAX_TRANSCRIPT_CHARS = 60_000


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config() -> dict:
    try:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_config(cfg: dict):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[config] could not save: {e}", flush=True)


# ── Video / audio merge ───────────────────────────────────────────────────────

def merge_video_audio(video_path: str, audio_path: str, output_path: str):
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("imageio-ffmpeg not installed. Run: pip install imageio-ffmpeg")

    result = subprocess.run(
        [ffmpeg, "-y",
         "-i", video_path,
         "-i", audio_path,
         "-c:v", "copy", "-c:a", "aac", "-shortest",
         output_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{result.stderr[-600:]}")


# ── AI summariser ─────────────────────────────────────────────────────────────

SUMMARY_PROMPT = """\
You are a meeting assistant. Analyze the transcript below and respond with a \
structured summary using these sections:

## Overview
2-3 sentences describing the meeting.

## Key Topics
Bullet list of main subjects discussed.

## Decisions
Decisions or conclusions reached (write "None identified" if absent).

## Action Items
Tasks assigned or follow-ups needed (write "None identified" if absent).

## Highlights
Notable quotes or important moments.

---
Transcript:
{transcript}
"""


def summarize_with_ai(transcript: str, provider: str, api_key: str,
                      base_url: str = None, model: str = None) -> str:
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = transcript[:MAX_TRANSCRIPT_CHARS] + "\n[transcript truncated]"

    prompt = SUMMARY_PROMPT.format(transcript=transcript)

    if provider == "Anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model or "claude-opus-4-8",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    elif provider == "OpenAI":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model or "gpt-4o",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    elif provider == "Gemini":
        from google import genai
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model or "gemini-2.0-flash",
            contents=prompt,
        )
        return resp.text

    elif provider == "Custom":
        from openai import OpenAI
        if not base_url:
            raise ValueError("Base URL is required for a custom provider.")
        client = OpenAI(
            api_key=api_key or "none",   # local servers (Ollama) accept any value
            base_url=base_url,
        )
        resp = client.chat.completions.create(
            model=model or "gpt-4o",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Vosk model cache + audio file transcription ───────────────────────────────

_vosk_model_cache = None
_vosk_model_lock  = threading.Lock()


def _get_vosk_model():
    global _vosk_model_cache
    with _vosk_model_lock:
        if _vosk_model_cache is None:
            from vosk import Model, SetLogLevel
            SetLogLevel(-1)
            _vosk_model_cache = Model(model_name="vosk-model-small-en-us-0.15")
        return _vosk_model_cache


def extract_audio_to_wav(src_path: str, dst_wav: str):
    """Extract / convert any audio or video file to 16 kHz mono WAV."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("imageio-ffmpeg not installed. Run: pip install imageio-ffmpeg")

    result = subprocess.run(
        [ffmpeg, "-y", "-i", src_path,
         "-ar", "16000", "-ac", "1", "-f", "wav", dst_wav],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{result.stderr[-400:]}")


def transcribe_audio_file(wav_path: str, progress_cb=None) -> str:
    """Transcribe a WAV file (16 kHz mono) using the local vosk model."""
    import wave as _wave
    from vosk import KaldiRecognizer

    model = _get_vosk_model()

    wf           = _wave.open(wav_path, "rb")
    sample_rate  = wf.getframerate()
    total_frames = wf.getnframes()
    rec          = KaldiRecognizer(model, sample_rate)
    parts        = []
    read_frames  = 0

    while True:
        data = wf.readframes(8000)
        if not data:
            break
        if rec.AcceptWaveform(data):
            r = json.loads(rec.Result())
            if r.get("text"):
                parts.append(r["text"])
        read_frames += 8000
        if progress_cb and total_frames:
            progress_cb(min(99, int(read_frames / total_frames * 100)))

    final = json.loads(rec.FinalResult())
    if final.get("text"):
        parts.append(final["text"])

    wf.close()
    if progress_cb:
        progress_cb(100)
    return " ".join(parts)


def get_transcript_for_session(session_dir: Path, progress_cb=None) -> str:
    """
    Return transcript text for a session.
    Priority: transcript.txt → audio.wav → meeting_final.mp4 → screen.mp4
    Raises RuntimeError if nothing usable is found.
    """
    txt = session_dir / "transcript.txt"
    if txt.exists():
        content = txt.read_text(encoding="utf-8").strip()
        if content:
            return content

    # Try audio sources in priority order
    for candidate in ["audio.wav", "meeting_final.mp4", "screen.mp4"]:
        src = session_dir / candidate
        if not src.exists():
            continue

        if progress_cb:
            progress_cb(f"Extracting audio from {candidate}…")

        wav_path = str(session_dir / "_tmp_transcribe.wav")
        try:
            if candidate == "audio.wav":
                wav_path = str(src)          # already the right format
            else:
                extract_audio_to_wav(str(src), wav_path)

            if progress_cb:
                progress_cb(f"Transcribing {candidate}…")

            transcript = transcribe_audio_file(
                wav_path,
                progress_cb=lambda pct: progress_cb(f"Transcribing {candidate}… {pct}%"),
            )

            # Save so next run is instant
            if transcript:
                txt.write_text(transcript, encoding="utf-8")

            return transcript
        finally:
            tmp = session_dir / "_tmp_transcribe.wav"
            if tmp.exists() and candidate != "audio.wav":
                tmp.unlink(missing_ok=True)

    raise RuntimeError(
        "No usable audio source found.\n"
        "Expected one of: transcript.txt, audio.wav, meeting_final.mp4, screen.mp4"
    )


# ── Audio recorder ────────────────────────────────────────────────────────────

class AudioRecorder:
    def __init__(self, sample_rate=SAMPLE_RATE, channels=1):
        self.sample_rate = sample_rate
        self.channels    = channels
        self.frames      = []
        self._lock       = threading.Lock()
        self.is_recording = False
        self.stream       = None

    def get_devices(self):
        devices = [(-1, "Default Microphone")]
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                devices.append((i, dev["name"]))
        return devices

    def start(self, device_id=None, chunk_callback=None):
        with self._lock:
            self.frames = []
        self.is_recording = True

        def callback(indata, frames, time_info, status):
            if not self.is_recording:
                return
            with self._lock:
                self.frames.append(indata.copy())
            if chunk_callback:
                mono = indata.flatten()
                pcm  = np.clip(mono * 32767, -32768, 32767).astype(np.int16).tobytes()
                chunk_callback(pcm)

        kwargs = dict(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=callback,
            blocksize=4000,
        )
        if device_id is not None and device_id >= 0:
            kwargs["device"] = device_id

        self.stream = sd.InputStream(**kwargs)
        self.stream.start()

    def stop(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        with self._lock:
            return np.concatenate(self.frames, axis=0) if self.frames else None

    def save(self, audio_data, filepath):
        audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        wavfile.write(filepath, self.sample_rate, audio_int16)


# ── Screen recorder ───────────────────────────────────────────────────────────

class ScreenRecorder:
    def __init__(self, fps=10, output_path=None):
        self.fps         = fps
        self.output_path = output_path
        self.is_recording = False
        self._thread      = None
        self._writer      = None

    def start(self):
        self.is_recording = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            w, h    = monitor["width"], monitor["height"]
            fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
            interval = 1.0 / self.fps
            while self.is_recording:
                t0    = time.monotonic()
                img   = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                self._writer.write(frame)
                sleep = interval - (time.monotonic() - t0)
                if sleep > 0:
                    time.sleep(sleep)

    def stop(self):
        self.is_recording = False
        if self._thread:
            self._thread.join(timeout=8)
        if self._writer:
            self._writer.release()
            self._writer = None


# ── Transcription worker (vosk) ───────────────────────────────────────────────

class TranscriptionWorker(QObject):
    caption_ready = pyqtSignal(str)
    partial_ready = pyqtSignal(str)
    model_loaded  = pyqtSignal()
    error         = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._model   = None
        self._rec     = None
        self._queue   = queue.Queue()
        self._running = False

    @pyqtSlot()
    def run(self):
        try:
            from vosk import KaldiRecognizer
            self._model = _get_vosk_model()
            self._rec   = KaldiRecognizer(self._model, SAMPLE_RATE)
            self.model_loaded.emit()
        except Exception as e:
            self.error.emit(f"Failed to load speech model:\n{e}\n\n{traceback.format_exc()}")
            return

        self._running = True
        while self._running:
            try:
                pcm = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                if self._rec.AcceptWaveform(pcm):
                    result = json.loads(self._rec.Result())
                    text   = result.get("text", "").strip()
                    if text:
                        self.caption_ready.emit(text)
                else:
                    partial = json.loads(self._rec.PartialResult())
                    self.partial_ready.emit(partial.get("partial", "").strip())
            except Exception as e:
                print(f"[transcription] {e}", flush=True)

    def enqueue(self, pcm_bytes: bytes):
        self._queue.put(pcm_bytes)

    def stop(self):
        self._running = False


# ── Summary worker ────────────────────────────────────────────────────────────

class SummaryWorker(QObject):
    status_update = pyqtSignal(str)
    session_done  = pyqtSignal(str, str)   # name, summary
    session_error = pyqtSignal(str, str)   # name, error
    all_done      = pyqtSignal()

    def __init__(self, sessions: list, provider: str, api_key: str,
                 transcribe_audio: bool = False,
                 base_url: str = None, model: str = None):
        super().__init__()
        self.sessions         = sessions
        self.provider         = provider
        self.api_key          = api_key
        self.transcribe_audio = transcribe_audio
        self.base_url         = base_url
        self.model            = model
        self._running         = True

    @pyqtSlot()
    def run(self):
        for i, session_dir in enumerate(self.sessions):
            if not self._running:
                break

            base_status = f"[{i + 1}/{len(self.sessions)}] {session_dir.name}"
            self.status_update.emit(f"{base_status} — reading transcript…")

            try:
                if self.transcribe_audio:
                    transcript = get_transcript_for_session(
                        session_dir,
                        progress_cb=lambda msg: self.status_update.emit(
                            f"{base_status} — {msg}"
                        ),
                    )
                else:
                    txt = session_dir / "transcript.txt"
                    if not txt.exists():
                        self.session_error.emit(
                            session_dir.name,
                            "No transcript.txt — enable 'Transcribe audio if missing' to use audio files.",
                        )
                        continue
                    transcript = txt.read_text(encoding="utf-8").strip()
                    if not transcript:
                        self.session_error.emit(session_dir.name, "transcript.txt is empty")
                        continue

                self.status_update.emit(f"{base_status} — sending to {self.provider}…")
                summary = summarize_with_ai(
                    transcript, self.provider, self.api_key,
                    base_url=self.base_url, model=self.model,
                )
                (session_dir / "summary.txt").write_text(summary, encoding="utf-8")
                self.session_done.emit(session_dir.name, summary)

            except Exception as e:
                self.session_error.emit(session_dir.name, str(e))

        self.all_done.emit()

    def stop(self):
        self._running = False


# ── Stylesheet ────────────────────────────────────────────────────────────────

APP_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', Tahoma, sans-serif;
    font-size: 13px;
}
QTabWidget::pane { border: none; background: #1e1e2e; }
QTabBar::tab {
    background: #181825; color: #6c7086;
    padding: 9px 22px; border: none;
    border-bottom: 2px solid transparent;
    font-size: 13px;
}
QTabBar::tab:selected { color: #cdd6f4; border-bottom: 2px solid #cba6f7; background: #1e1e2e; }
QTabBar::tab:hover    { color: #cdd6f4; background: #313244; }
QPushButton#startBtn {
    background-color: #a6e3a1; color: #1e1e2e;
    border: none; border-radius: 10px;
    padding: 14px 40px; font-size: 15px; font-weight: bold; min-width: 200px;
}
QPushButton#startBtn:hover { background-color: #94d480; }
QPushButton#startBtn[recording="true"] { background-color: #f38ba8; }
QPushButton#startBtn[recording="true"]:hover { background-color: #e0789a; }
QPushButton#smallBtn {
    background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 6px; padding: 5px 14px;
}
QPushButton#smallBtn:hover    { background-color: #45475a; }
QPushButton#smallBtn:disabled { color: #6c7086; }
QPushButton#accentBtn {
    background-color: #89b4fa; color: #1e1e2e;
    border: none; border-radius: 6px; padding: 6px 18px; font-weight: bold;
}
QPushButton#accentBtn:hover    { background-color: #74a8f5; }
QPushButton#accentBtn:disabled { background-color: #313244; color: #6c7086; }
QTextEdit {
    background-color: #181825; color: #cdd6f4;
    border: 1px solid #313244; border-radius: 8px;
    padding: 10px; font-size: 13px;
    selection-background-color: #45475a;
}
QLineEdit {
    background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 6px; padding: 5px 10px;
}
QLineEdit:focus { border-color: #89b4fa; }
QListWidget {
    background-color: #181825; color: #cdd6f4;
    border: 1px solid #313244; border-radius: 6px; padding: 4px;
}
QListWidget::item { padding: 6px 8px; border-radius: 4px; }
QListWidget::item:selected { background-color: #45475a; }
QListWidget::item:hover    { background-color: #313244; }
QLabel#timerLabel {
    font-size: 26px; font-weight: bold; color: #6c7086;
    font-family: 'Courier New', monospace;
}
QLabel#timerLabel[recording="true"] { color: #f38ba8; }
QLabel#recDot { font-size: 20px; color: #313244; }
QLabel#recDot[recording="true"] { color: #f38ba8; }
QLabel#partialLabel { color: #6c7086; font-style: italic; font-size: 12px; padding: 2px 10px 4px; }
QComboBox {
    background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 6px; padding: 5px 10px;
}
QComboBox QAbstractItemView {
    background-color: #313244; color: #cdd6f4;
    selection-background-color: #45475a; border: 1px solid #45475a;
}
QGroupBox {
    border: 1px solid #313244; border-radius: 8px;
    margin-top: 14px; padding-top: 6px;
    color: #89b4fa; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px; }
QCheckBox { color: #cdd6f4; spacing: 8px; }
QCheckBox::indicator {
    width: 15px; height: 15px;
    border: 2px solid #45475a; border-radius: 3px; background: #313244;
}
QCheckBox::indicator:checked { background: #89b4fa; border-color: #89b4fa; }
QSplitter::handle { background: #313244; width: 1px; }
QStatusBar {
    background: #181825; color: #6c7086;
    border-top: 1px solid #313244; padding: 2px 8px;
}
"""


def _repoll(widget):
    widget.style().unpolish(widget)
    widget.style().polish(widget)


# ── Summary tab ───────────────────────────────────────────────────────────────

class SummaryTab(QWidget):
    def __init__(self, get_output_dir, parent=None):
        super().__init__(parent)
        self._get_output_dir = get_output_dir
        self._config         = load_config()
        self._sw_thread      = None
        self._sw_worker      = None
        self._build_ui()
        self.refresh_sessions()

    def _build_ui(self):
        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(18, 14, 18, 12)
        vbox.setSpacing(12)

        # Provider settings
        prov_group = QGroupBox("AI Provider")
        pg = QVBoxLayout(prov_group)

        prov_row = QHBoxLayout()
        prov_row.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Anthropic", "OpenAI", "Gemini", "Custom"])
        saved = self._config.get("provider", "Anthropic")
        idx = self.provider_combo.findText(saved)
        if idx >= 0:
            self.provider_combo.setCurrentIndex(idx)
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        prov_row.addWidget(self.provider_combo)
        prov_row.addStretch()
        pg.addLayout(prov_row)

        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Paste your API key here…")
        self.api_key_input.setText(
            self._config.get(f"key_{self.provider_combo.currentText()}", "")
        )
        self.api_key_input.textChanged.connect(self._save_key)
        key_row.addWidget(self.api_key_input, 1)
        show_btn = QPushButton("Show")
        show_btn.setObjectName("smallBtn")
        show_btn.setFixedWidth(50)
        show_btn.setCheckable(True)
        show_btn.toggled.connect(
            lambda on: self.api_key_input.setEchoMode(
                QLineEdit.Normal if on else QLineEdit.Password
            )
        )
        key_row.addWidget(show_btn)
        pg.addLayout(key_row)

        # Custom provider extra fields (shown only when "Custom" is selected)
        self.custom_widget = QWidget()
        cw = QVBoxLayout(self.custom_widget)
        cw.setContentsMargins(0, 4, 0, 0)
        cw.setSpacing(6)

        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("Base URL:"))
        self.base_url_input = QLineEdit()
        self.base_url_input.setPlaceholderText(
            "e.g. http://localhost:11434/v1  or  https://api.groq.com/openai/v1"
        )
        self.base_url_input.setText(self._config.get("custom_base_url", ""))
        self.base_url_input.textChanged.connect(
            lambda t: (self._config.update({"custom_base_url": t}), save_config(self._config))
        )
        url_row.addWidget(self.base_url_input, 1)
        cw.addLayout(url_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText(
            "e.g. llama3.2  /  mistral  /  gpt-4o  /  deepseek-r1"
        )
        self.model_input.setText(self._config.get("custom_model", ""))
        self.model_input.textChanged.connect(
            lambda t: (self._config.update({"custom_model": t}), save_config(self._config))
        )
        model_row.addWidget(self.model_input, 1)
        cw.addLayout(model_row)

        pg.addWidget(self.custom_widget)
        self.custom_widget.setVisible(self.provider_combo.currentText() == "Custom")

        self.cb_transcribe = QCheckBox(
            "Transcribe audio if no transcript.txt  "
            "(reads audio.wav / meeting_final.mp4)"
        )
        self.cb_transcribe.setChecked(self._config.get("transcribe_audio", True))
        self.cb_transcribe.toggled.connect(
            lambda on: (self._config.update({"transcribe_audio": on}), save_config(self._config))
        )
        pg.addWidget(self.cb_transcribe)
        vbox.addWidget(prov_group)

        # Splitter: session list | summary output
        splitter = QSplitter(Qt.Horizontal)

        # Left panel — session list
        left = QWidget()
        lv   = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.addWidget(QLabel("Recordings:"))
        self.session_list = QListWidget()
        self.session_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.session_list.itemDoubleClicked.connect(self._load_existing_summary)
        lv.addWidget(self.session_list)

        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("smallBtn")
        refresh_btn.clicked.connect(self.refresh_sessions)
        btn_row.addWidget(refresh_btn)
        btn_row.addStretch()
        lv.addLayout(btn_row)

        action_row = QHBoxLayout()
        self.summarize_btn = QPushButton("Summarize Selected")
        self.summarize_btn.setObjectName("accentBtn")
        self.summarize_btn.clicked.connect(self._summarize_selected)
        action_row.addWidget(self.summarize_btn)

        self.summarize_all_btn = QPushButton("Summarize All")
        self.summarize_all_btn.setObjectName("smallBtn")
        self.summarize_all_btn.clicked.connect(self._summarize_all)
        action_row.addWidget(self.summarize_all_btn)
        lv.addLayout(action_row)

        splitter.addWidget(left)

        # Right panel — summary output
        right = QWidget()
        rv    = QVBoxLayout(right)
        rv.setContentsMargins(6, 0, 0, 0)
        rv.addWidget(QLabel("Summary:"))
        self.summary_output = QTextEdit()
        self.summary_output.setReadOnly(True)
        self.summary_output.setPlaceholderText(
            "Select a session and click Summarize Selected.\n"
            "Double-click a session that already has a summary (✓) to view it."
        )
        rv.addWidget(self.summary_output)

        foot_row = QHBoxLayout()
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color: #6c7086; font-size: 12px;")
        foot_row.addWidget(self.status_lbl, 1)
        copy_btn = QPushButton("Copy")
        copy_btn.setObjectName("smallBtn")
        copy_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(self.summary_output.toPlainText())
        )
        foot_row.addWidget(copy_btn)
        rv.addLayout(foot_row)

        splitter.addWidget(right)
        splitter.setSizes([280, 540])
        vbox.addWidget(splitter, 1)

    def _on_provider_changed(self, provider):
        self._config["provider"] = provider
        save_config(self._config)
        self.api_key_input.blockSignals(True)
        self.api_key_input.setText(self._config.get(f"key_{provider}", ""))
        placeholder = "Optional for local servers (Ollama, LM Studio…)" \
            if provider == "Custom" else "Paste your API key here…"
        self.api_key_input.setPlaceholderText(placeholder)
        self.api_key_input.blockSignals(False)
        self.custom_widget.setVisible(provider == "Custom")

    def _save_key(self, text):
        provider = self.provider_combo.currentText()
        self._config[f"key_{provider}"] = text
        save_config(self._config)

    def refresh_sessions(self):
        self.session_list.clear()
        output_dir = self._get_output_dir()
        if not output_dir.exists():
            return
        for d in sorted(output_dir.iterdir(), reverse=True):
            if not (d.is_dir() and d.name.startswith("meeting_")):
                continue
            has_transcript = (d / "transcript.txt").exists()
            has_summary    = (d / "summary.txt").exists()
            has_audio      = (d / "audio.wav").exists()
            has_final      = (d / "meeting_final.mp4").exists()

            icon   = "✓ " if has_summary else "  "
            tags   = []
            if has_final:   tags.append("video")
            if has_audio and not has_transcript: tags.append("audio only")
            label  = icon + d.name + (f"  [{', '.join(tags)}]" if tags else "")

            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, d)
            if has_summary:
                item.setForeground(QColor("#a6e3a1"))
                item.setToolTip("Summary exists — double-click to view")
            elif has_transcript:
                item.setToolTip("Has transcript — ready to summarize")
            elif has_audio or has_final:
                item.setForeground(QColor("#fab387"))
                item.setToolTip(
                    "No transcript — enable 'Transcribe audio if missing' to summarize from audio"
                )
            else:
                item.setForeground(QColor("#6c7086"))
                item.setToolTip("No transcript or audio found")
            self.session_list.addItem(item)

    def _load_existing_summary(self, item):
        session_dir  = item.data(Qt.UserRole)
        summary_path = session_dir / "summary.txt"
        if summary_path.exists():
            self.summary_output.setPlainText(summary_path.read_text(encoding="utf-8"))
            self.status_lbl.setText(f"Loaded: {session_dir.name}")

    def _summarize_selected(self):
        items = self.session_list.selectedItems()
        if not items:
            QMessageBox.information(self, "Nothing Selected",
                                    "Select one or more sessions from the list.")
            return
        sessions = [item.data(Qt.UserRole) for item in items
                    if (item.data(Qt.UserRole) / "transcript.txt").exists()]
        if not sessions:
            QMessageBox.warning(self, "No Transcripts",
                                "Selected sessions have no transcripts to summarize.")
            return
        self._run_summarize(sessions)

    def _summarize_all(self):
        output_dir        = self._get_output_dir()
        use_audio         = self.cb_transcribe.isChecked()
        audio_candidates  = {"audio.wav", "meeting_final.mp4", "screen.mp4"}

        def is_eligible(d: Path) -> bool:
            if (d / "transcript.txt").exists():
                return True
            return use_audio and any((d / c).exists() for c in audio_candidates)

        sessions = sorted([
            d for d in output_dir.iterdir()
            if d.is_dir() and d.name.startswith("meeting_") and is_eligible(d)
        ])
        if not sessions:
            QMessageBox.information(
                self, "Nothing to Summarize",
                "No sessions with transcripts or audio files found.\n"
                "Enable 'Transcribe audio if missing' to use raw audio.",
            )
            return
        self._run_summarize(sessions)

    def _run_summarize(self, sessions):
        provider = self.provider_combo.currentText()
        api_key  = self.api_key_input.text().strip()
        base_url = self.base_url_input.text().strip() if provider == "Custom" else None
        model    = self.model_input.text().strip() if provider == "Custom" else None

        if provider == "Custom" and not base_url:
            QMessageBox.warning(self, "Base URL Required",
                                "Enter the Base URL for your custom provider.")
            return
        if provider != "Custom" and not api_key:
            QMessageBox.warning(self, "API Key Required",
                                f"Enter your {provider} API key above.")
            return

        self.summarize_btn.setEnabled(False)
        self.summarize_all_btn.setEnabled(False)
        self.summary_output.clear()
        self.status_lbl.setText("Starting…")

        worker = SummaryWorker(
            sessions, provider, api_key,
            transcribe_audio=self.cb_transcribe.isChecked(),
            base_url=base_url,
            model=model,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.status_update.connect(self.status_lbl.setText)
        worker.session_done.connect(self._on_session_done)
        worker.session_error.connect(self._on_session_error)
        worker.all_done.connect(lambda: self._on_all_done(thread))
        thread.start()
        self._sw_thread = thread
        self._sw_worker = worker

    @pyqtSlot(str, str)
    def _on_session_done(self, name, summary):
        cur = self.summary_output.textCursor()
        cur.movePosition(QTextCursor.End)
        if self.summary_output.toPlainText():
            cur.insertText("\n\n")
        cur.insertText(f"{'='*60}\n{name}\n{'='*60}\n{summary}")
        self.summary_output.setTextCursor(cur)
        self.summary_output.ensureCursorVisible()

    @pyqtSlot(str, str)
    def _on_session_error(self, name, error):
        cur = self.summary_output.textCursor()
        cur.movePosition(QTextCursor.End)
        if self.summary_output.toPlainText():
            cur.insertText("\n\n")
        cur.insertText(f"{'='*60}\n{name}\n{'='*60}\n[Error: {error}]")
        self.summary_output.setTextCursor(cur)

    def _on_all_done(self, thread):
        thread.quit()
        thread.wait(3000)
        self.summarize_btn.setEnabled(True)
        self.summarize_all_btn.setEnabled(True)
        self.status_lbl.setText("Done")
        self.refresh_sessions()


# ── Main window ───────────────────────────────────────────────────────────────

class MeetingRecorder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.is_recording     = False
        self.start_time       = None
        self.output_dir       = Path.home() / "MeetingRecordings"
        self.output_dir.mkdir(exist_ok=True)
        self.session_dir      = None
        self.transcript_lines = []

        self.audio      = AudioRecorder()
        self.screen_rec = None
        self._worker    = None
        self._t_thread  = None

        self._build_ui()
        QTimer.singleShot(500, self._init_transcription)

    # ── UI ────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("Meeting Recorder")
        self.setMinimumSize(820, 700)
        self.resize(960, 760)
        self.setStyleSheet(APP_STYLE)

        root = QWidget()
        self.setCentralWidget(root)
        root_vbox = QVBoxLayout(root)
        root_vbox.setContentsMargins(0, 0, 0, 0)
        root_vbox.setSpacing(0)

        self.tabs = QTabWidget()
        root_vbox.addWidget(self.tabs)

        # ── Tab 1: Record ─────────────────────────────────────────
        record_widget = QWidget()
        vbox = QVBoxLayout(record_widget)
        vbox.setContentsMargins(22, 18, 22, 10)
        vbox.setSpacing(14)

        title_row = QHBoxLayout()
        title = QLabel("Meeting Recorder")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setStyleSheet("color: #cba6f7;")
        title_row.addWidget(title)
        title_row.addStretch()
        self.rec_dot = QLabel("●")
        self.rec_dot.setObjectName("recDot")
        self.rec_dot.setProperty("recording", "false")
        title_row.addWidget(self.rec_dot)
        self.timer_label = QLabel("00:00:00")
        self.timer_label.setObjectName("timerLabel")
        self.timer_label.setProperty("recording", "false")
        title_row.addWidget(self.timer_label)
        vbox.addLayout(title_row)

        settings = QGroupBox("Settings")
        sg = QVBoxLayout(settings)

        mic_row = QHBoxLayout()
        mic_row.addWidget(QLabel("Microphone / Input:"))
        self.mic_combo = QComboBox()
        self._populate_devices()
        mic_row.addWidget(self.mic_combo, 1)
        sg.addLayout(mic_row)

        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Save to:"))
        self.folder_lbl = QLabel(str(self.output_dir))
        self.folder_lbl.setStyleSheet("color: #89b4fa; font-size: 12px;")
        self.folder_lbl.setWordWrap(True)
        folder_row.addWidget(self.folder_lbl, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.setObjectName("smallBtn")
        browse_btn.clicked.connect(self._browse)
        folder_row.addWidget(browse_btn)
        sg.addLayout(folder_row)

        opts_row = QHBoxLayout()
        self.cb_screen     = QCheckBox("Record screen")
        self.cb_screen.setChecked(True)
        self.cb_transcript = QCheckBox("Save transcript")
        self.cb_transcript.setChecked(True)
        self.cb_merge      = QCheckBox("Merge to final video after recording")
        self.cb_merge.setChecked(True)
        opts_row.addWidget(self.cb_screen)
        opts_row.addWidget(self.cb_transcript)
        opts_row.addWidget(self.cb_merge)
        opts_row.addStretch()
        sg.addLayout(opts_row)
        vbox.addWidget(settings)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.start_btn = QPushButton("▶  Start Recording")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setProperty("recording", "false")
        self.start_btn.clicked.connect(self._toggle)
        btn_row.addWidget(self.start_btn)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        caps = QGroupBox("Live Captions")
        cl   = QVBoxLayout(caps)
        self.caption_box = QTextEdit()
        self.caption_box.setReadOnly(True)
        self.caption_box.setMinimumHeight(160)
        self.caption_box.setPlaceholderText("Live captions will appear here once recording starts…")
        cl.addWidget(self.caption_box)
        self.partial_label = QLabel("")
        self.partial_label.setObjectName("partialLabel")
        self.partial_label.setWordWrap(True)
        cl.addWidget(self.partial_label)
        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("smallBtn")
        clear_btn.setFixedWidth(70)
        clear_btn.clicked.connect(self._clear_captions)
        cl.addWidget(clear_btn, alignment=Qt.AlignRight)
        vbox.addWidget(caps, 1)

        self.tabs.addTab(record_widget, "⏺  Record")

        # ── Tab 2: Summaries ──────────────────────────────────────
        self.summary_tab = SummaryTab(lambda: self.output_dir)
        self.tabs.addTab(self.summary_tab, "📝  Summaries")

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Loading speech recognition model…")

        self._timer       = QTimer()
        self._timer.timeout.connect(self._tick)
        self._blink_state = False

    def _clear_captions(self):
        self.caption_box.clear()
        self.partial_label.clear()

    def _populate_devices(self):
        self.mic_combo.clear()
        for dev_id, name in self.audio.get_devices():
            self.mic_combo.addItem(name, dev_id)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", str(self.output_dir),
            QFileDialog.DontUseNativeDialog,
        )
        if folder:
            self.output_dir = Path(folder)
            self.folder_lbl.setText(folder)
            self.summary_tab.refresh_sessions()

    # ── Transcription ─────────────────────────────────────────────

    def _init_transcription(self):
        try:
            self._worker   = TranscriptionWorker()
            self._t_thread = QThread()
            self._worker.moveToThread(self._t_thread)
            self._t_thread.started.connect(self._worker.run)
            self._worker.model_loaded.connect(self._on_model_loaded)
            self._worker.caption_ready.connect(self._on_caption)
            self._worker.partial_ready.connect(self._on_partial)
            self._worker.error.connect(self._on_transcription_error)
            self._t_thread.start()
        except Exception as e:
            self._on_transcription_error(
                f"Could not start transcription:\n{e}\n\n{traceback.format_exc()}"
            )

    @pyqtSlot()
    def _on_model_loaded(self):
        self.status.showMessage("Ready — press Start Recording to begin")

    @pyqtSlot(str)
    def _on_partial(self, text):
        self.partial_label.setText(text + "…" if text else "")

    @pyqtSlot(str)
    def _on_caption(self, text):
        self.partial_label.clear()
        self.transcript_lines.append(text)
        cur = self.caption_box.textCursor()
        cur.movePosition(QTextCursor.End)
        prefix = "\n" if self.caption_box.toPlainText() else ""
        cur.insertText(prefix + text)
        self.caption_box.setTextCursor(cur)
        self.caption_box.ensureCursorVisible()

    @pyqtSlot(str)
    def _on_transcription_error(self, msg):
        print(f"[transcription error]\n{msg}", flush=True)
        self.status.showMessage("Speech model failed — captions unavailable (recording still works)")
        QMessageBox.warning(self, "Speech Model Error", msg)

    # ── Recording ─────────────────────────────────────────────────

    def _toggle(self):
        if self.is_recording:
            self._stop()
        else:
            self._start()

    def _start(self):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = self.output_dir / f"meeting_{ts}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        device_id = self.mic_combo.currentData()
        chunk_cb  = self._worker.enqueue if self._worker else None
        self.audio.start(device_id=device_id, chunk_callback=chunk_cb)

        if self.cb_screen.isChecked():
            self.screen_rec = ScreenRecorder(
                fps=10,
                output_path=str(self.session_dir / "screen.mp4"),
            )
            self.screen_rec.start()

        self.transcript_lines = []
        self.is_recording     = True
        self.start_time       = time.monotonic()

        self.start_btn.setText("■  Stop Recording")
        self.start_btn.setProperty("recording", "true")
        _repoll(self.start_btn)
        self.timer_label.setProperty("recording", "true")
        _repoll(self.timer_label)
        self.rec_dot.setProperty("recording", "true")
        _repoll(self.rec_dot)

        self._timer.start(500)
        self._clear_captions()
        self.status.showMessage(f"Recording…  →  {self.session_dir}")

    def _stop(self):
        self.is_recording = False
        self._timer.stop()
        self.partial_label.clear()

        self.start_btn.setText("▶  Start Recording")
        self.start_btn.setProperty("recording", "false")
        _repoll(self.start_btn)
        self.timer_label.setProperty("recording", "false")
        _repoll(self.timer_label)
        self.rec_dot.setProperty("recording", "false")
        _repoll(self.rec_dot)

        self.status.showMessage("Saving files…")
        threading.Thread(target=self._save, daemon=True).start()

    def _save(self):
        # Save audio
        audio_data  = self.audio.stop()
        audio_path  = None
        if audio_data is not None:
            audio_path = self.session_dir / "audio.wav"
            self.audio.save(audio_data, str(audio_path))

        # Stop screen recorder
        video_path = None
        if self.screen_rec:
            self.screen_rec.stop()
            self.screen_rec = None
            video_path = self.session_dir / "screen.mp4"

        # Merge video + audio into final video
        if (self.cb_merge.isChecked()
                and video_path and video_path.exists()
                and audio_path and audio_path.exists()):
            try:
                final_path = self.session_dir / "meeting_final.mp4"
                self._update_status("Merging video + audio…")
                merge_video_audio(str(video_path), str(audio_path), str(final_path))
            except Exception as e:
                print(f"[merge] {e}", flush=True)

        # Save transcript
        if self.cb_transcript.isChecked() and self.transcript_lines:
            (self.session_dir / "transcript.txt").write_text(
                "\n".join(self.transcript_lines), encoding="utf-8"
            )

        from PyQt5.QtCore import QMetaObject
        QMetaObject.invokeMethod(self, "_on_saved", Qt.QueuedConnection)

    @pyqtSlot()
    def _on_saved(self):
        self.status.showMessage(f"Saved  →  {self.session_dir}")
        self.summary_tab.refresh_sessions()

    def _tick(self):
        if self.start_time is not None and self.is_recording:
            elapsed = int(time.monotonic() - self.start_time)
            h, r    = divmod(elapsed, 3600)
            m, s    = divmod(r, 60)
            self.timer_label.setText(f"{h:02d}:{m:02d}:{s:02d}")
            self._blink_state = not self._blink_state
            self.rec_dot.setStyleSheet(
                "color: #f38ba8;" if self._blink_state else "color: #7d3045;"
            )

    def closeEvent(self, event):
        if self.is_recording:
            self._stop()
        if self._worker:
            self._worker.stop()
        if self._t_thread:
            self._t_thread.quit()
            self._t_thread.wait(3000)
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def _exception_hook(exctype, value, tb):
    msg = "".join(traceback.format_exception(exctype, value, tb))
    print(msg, flush=True)
    QMessageBox.critical(None, "Unhandled Error", msg)
    sys.__excepthook__(exctype, value, tb)


def main():
    sys.excepthook = _exception_hook
    app = QApplication(sys.argv)
    app.setApplicationName("Meeting Recorder")
    win = MeetingRecorder()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
