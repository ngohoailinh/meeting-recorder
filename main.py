import sys
import os
import threading
import queue
import time
import traceback
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
    QComboBox, QGroupBox, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QObject, QThread
from PyQt5.QtGui import QFont, QTextCursor
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
CAPTION_CHUNK_SECONDS = 4


class AudioRecorder:
    def __init__(self, sample_rate=SAMPLE_RATE, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames = []
        self._lock = threading.Lock()
        self.is_recording = False
        self.stream = None

    def get_devices(self):
        devices = [(-1, "Default Microphone")]
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                devices.append((i, dev['name']))
        return devices

    def start(self, device_id=None):
        with self._lock:
            self.frames = []
        self.is_recording = True

        def callback(indata, frames, time_info, status):
            if self.is_recording:
                with self._lock:
                    self.frames.append(indata.copy())

        kwargs = dict(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            callback=callback,
        )
        if device_id is not None and device_id >= 0:
            kwargs['device'] = device_id

        self.stream = sd.InputStream(**kwargs)
        self.stream.start()

    def get_frames_snapshot(self):
        with self._lock:
            return list(self.frames)

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


class ScreenRecorder:
    def __init__(self, fps=10, output_path=None):
        self.fps = fps
        self.output_path = output_path
        self.is_recording = False
        self._thread = None
        self._writer = None

    def start(self):
        self.is_recording = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            w, h = monitor['width'], monitor['height']
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
            interval = 1.0 / self.fps
            while self.is_recording:
                t0 = time.monotonic()
                img = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                self._writer.write(frame)
                elapsed = time.monotonic() - t0
                sleep = interval - elapsed
                if sleep > 0:
                    time.sleep(sleep)

    def stop(self):
        self.is_recording = False
        if self._thread:
            self._thread.join(timeout=8)
        if self._writer:
            self._writer.release()
            self._writer = None


class TranscriptionWorker(QObject):
    caption_ready = pyqtSignal(str)
    model_loaded = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model_size="base"):
        super().__init__()
        self.model_size = model_size
        self._model = None
        self._queue = queue.Queue()
        self._running = False

    @pyqtSlot()
    def run(self):
        try:
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="float32")
            self.model_loaded.emit()
        except Exception as e:
            self.error.emit(f"Failed to load speech model:\n{e}\n\n{traceback.format_exc()}")
            return

        self._running = True
        while self._running:
            try:
                chunk = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                if chunk.ndim > 1:
                    chunk = chunk.mean(axis=1)
                segments, _ = self._model.transcribe(
                    chunk,
                    language="en",
                    beam_size=1,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=300),
                )
                text = " ".join(seg.text for seg in segments).strip()
                if text:
                    self.caption_ready.emit(text)
            except Exception as e:
                print(f"[transcription error] {e}", flush=True)

    def enqueue(self, chunk: np.ndarray):
        self._queue.put(chunk)

    def stop(self):
        self._running = False


APP_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', Tahoma, sans-serif;
    font-size: 13px;
}
QPushButton#startBtn {
    background-color: #a6e3a1;
    color: #1e1e2e;
    border: none;
    border-radius: 10px;
    padding: 14px 40px;
    font-size: 15px;
    font-weight: bold;
    min-width: 200px;
}
QPushButton#startBtn:hover { background-color: #94d480; }
QPushButton#startBtn[recording="true"] { background-color: #f38ba8; }
QPushButton#startBtn[recording="true"]:hover { background-color: #e0789a; }
QPushButton#smallBtn {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 5px 14px;
}
QPushButton#smallBtn:hover { background-color: #45475a; }
QTextEdit {
    background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #313244;
    border-radius: 8px;
    padding: 10px;
    font-size: 13px;
    line-height: 1.6;
    selection-background-color: #45475a;
}
QLabel#timerLabel {
    font-size: 26px;
    font-weight: bold;
    color: #6c7086;
    font-family: 'Courier New', monospace;
}
QLabel#timerLabel[recording="true"] { color: #f38ba8; }
QLabel#recDot {
    font-size: 20px;
    color: #313244;
}
QLabel#recDot[recording="true"] { color: #f38ba8; }
QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 5px 10px;
    min-width: 200px;
}
QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    selection-background-color: #45475a;
    border: 1px solid #45475a;
}
QGroupBox {
    border: 1px solid #313244;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 6px;
    color: #89b4fa;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
}
QCheckBox { color: #cdd6f4; spacing: 8px; }
QCheckBox::indicator {
    width: 15px; height: 15px;
    border: 2px solid #45475a;
    border-radius: 3px;
    background: #313244;
}
QCheckBox::indicator:checked { background: #89b4fa; border-color: #89b4fa; }
QStatusBar {
    background: #181825;
    color: #6c7086;
    border-top: 1px solid #313244;
    padding: 2px 8px;
}
"""


def _repoll(widget):
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class MeetingRecorder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.start_time = None
        self.output_dir = Path.home() / "MeetingRecordings"
        self.output_dir.mkdir(exist_ok=True)
        self.session_dir = None
        self.transcript_lines = []

        self.audio = AudioRecorder()
        self.screen_rec = None
        self._chunk_thread = None

        self._worker = None
        self._t_thread = None
        self._transcription_ready = False

        self._build_ui()
        # Defer heavy model loading — starts only when recording begins
        self._schedule_transcription_init()

    def _build_ui(self):
        self.setWindowTitle("Meeting Recorder")
        self.setMinimumSize(720, 620)
        self.resize(820, 680)
        self.setStyleSheet(APP_STYLE)

        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(22, 18, 22, 10)
        vbox.setSpacing(14)

        # ── Title row ────────────────────────────────────────────
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

        # ── Settings ─────────────────────────────────────────────
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
        self.cb_screen = QCheckBox("Record screen")
        self.cb_screen.setChecked(True)
        opts_row.addWidget(self.cb_screen)
        self.cb_transcript = QCheckBox("Save transcript")
        self.cb_transcript.setChecked(True)
        opts_row.addWidget(self.cb_transcript)
        opts_row.addStretch()
        sg.addLayout(opts_row)
        vbox.addWidget(settings)

        # ── Start / Stop button ──────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.start_btn = QPushButton("▶  Start Recording")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setProperty("recording", "false")
        self.start_btn.clicked.connect(self._toggle)
        btn_row.addWidget(self.start_btn)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        # ── Live captions ────────────────────────────────────────
        caps = QGroupBox("Live Captions")
        cl = QVBoxLayout(caps)
        self.caption_box = QTextEdit()
        self.caption_box.setReadOnly(True)
        self.caption_box.setMinimumHeight(180)
        self.caption_box.setPlaceholderText(
            "Live captions will appear here once recording starts…"
        )
        cl.addWidget(self.caption_box)

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("smallBtn")
        clear_btn.setFixedWidth(70)
        clear_btn.clicked.connect(self.caption_box.clear)
        cl.addWidget(clear_btn, alignment=Qt.AlignRight)
        vbox.addWidget(caps, 1)

        # ── Status bar ───────────────────────────────────────────
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — loading speech recognition model…")

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._blink_state = False

    def _populate_devices(self):
        self.mic_combo.clear()
        for dev_id, name in self.audio.get_devices():
            self.mic_combo.addItem(name, dev_id)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            str(self.output_dir),
            QFileDialog.DontUseNativeDialog,
        )
        if folder:
            self.output_dir = Path(folder)
            self.folder_lbl.setText(folder)

    # ── Transcription ─────────────────────────────────────────────

    def _schedule_transcription_init(self):
        # Start model loading 500ms after the window is shown so the UI appears first
        QTimer.singleShot(500, self._init_transcription)

    def _init_transcription(self):
        try:
            self._worker = TranscriptionWorker(model_size="base")
            self._t_thread = QThread()
            self._worker.moveToThread(self._t_thread)
            self._t_thread.started.connect(self._worker.run)
            self._worker.caption_ready.connect(self._on_caption)
            self._worker.model_loaded.connect(self._on_model_loaded)
            self._worker.error.connect(self._on_transcription_error)
            self._t_thread.start()
        except Exception as e:
            self._on_transcription_error(f"Could not start transcription thread:\n{e}\n\n{traceback.format_exc()}")

    @pyqtSlot()
    def _on_model_loaded(self):
        self.status.showMessage("Ready — press Start Recording to begin")

    @pyqtSlot(str)
    def _on_transcription_error(self, msg):
        from PyQt5.QtWidgets import QMessageBox
        print(f"[transcription error]\n{msg}", flush=True)
        self.status.showMessage("Speech model failed to load — captions unavailable")
        QMessageBox.warning(self, "Speech Model Error", msg)

    @pyqtSlot(str)
    def _on_caption(self, text):
        self.transcript_lines.append(text)
        cur = self.caption_box.textCursor()
        cur.movePosition(QTextCursor.End)
        prefix = "\n" if self.caption_box.toPlainText() else ""
        cur.insertText(prefix + text)
        self.caption_box.setTextCursor(cur)
        self.caption_box.ensureCursorVisible()

    # ── Recording control ─────────────────────────────────────────

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
        self.audio.start(device_id=device_id)

        if self.cb_screen.isChecked():
            self.screen_rec = ScreenRecorder(
                fps=10,
                output_path=str(self.session_dir / "screen.mp4"),
            )
            self.screen_rec.start()

        self.transcript_lines = []
        self.is_recording = True
        self.start_time = time.monotonic()

        if self._worker is not None:
            self._chunk_thread = threading.Thread(target=self._feed_chunks, daemon=True)
            self._chunk_thread.start()

        self.start_btn.setText("■  Stop Recording")
        self.start_btn.setProperty("recording", "true")
        _repoll(self.start_btn)
        self.timer_label.setProperty("recording", "true")
        _repoll(self.timer_label)
        self.rec_dot.setProperty("recording", "true")
        _repoll(self.rec_dot)

        self._timer.start(500)
        self.caption_box.clear()
        self.status.showMessage(f"Recording…  →  {self.session_dir}")

    def _feed_chunks(self):
        last_len = 0
        while self.is_recording:
            time.sleep(CAPTION_CHUNK_SECONDS)
            snap = self.audio.get_frames_snapshot()
            if len(snap) > last_len:
                chunk = np.concatenate(snap[last_len:], axis=0)
                last_len = len(snap)
                self._worker.enqueue(chunk)

    def _stop(self):
        self.is_recording = False
        self._timer.stop()

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
        audio_data = self.audio.stop()
        if audio_data is not None:
            self.audio.save(audio_data, str(self.session_dir / "audio.wav"))

        if self.screen_rec:
            self.screen_rec.stop()
            self.screen_rec = None

        if self.cb_transcript.isChecked() and self.transcript_lines:
            (self.session_dir / "transcript.txt").write_text(
                "\n".join(self.transcript_lines), encoding="utf-8"
            )

        # fire status update back on main thread
        from PyQt5.QtCore import QMetaObject
        QMetaObject.invokeMethod(self, "_on_saved", Qt.QueuedConnection)

    @pyqtSlot()
    def _on_saved(self):
        self.status.showMessage(f"Saved  →  {self.session_dir}")

    def _tick(self):
        if self.start_time is not None and self.is_recording:
            elapsed = int(time.monotonic() - self.start_time)
            h, r = divmod(elapsed, 3600)
            m, s = divmod(r, 60)
            self.timer_label.setText(f"{h:02d}:{m:02d}:{s:02d}")
            # blink the dot
            self._blink_state = not self._blink_state
            self.rec_dot.setStyleSheet(
                "color: #f38ba8;" if self._blink_state else "color: #7d3045;"
            )

    def closeEvent(self, event):
        if self.is_recording:
            self._stop()
        self._worker.stop()
        self._t_thread.quit()
        self._t_thread.wait(3000)
        event.accept()


def _exception_hook(exctype, value, tb):
    msg = "".join(traceback.format_exception(exctype, value, tb))
    print(msg, flush=True)
    from PyQt5.QtWidgets import QMessageBox
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
