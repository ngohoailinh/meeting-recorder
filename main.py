import sys
import json
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
    QComboBox, QGroupBox, QCheckBox, QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QObject, QThread
from PyQt5.QtGui import QFont, QTextCursor


SAMPLE_RATE = 16000


# ── Audio recorder ────────────────────────────────────────────────────────────

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
                pcm = np.clip(mono * 32767, -32768, 32767).astype(np.int16).tobytes()
                chunk_callback(pcm)

        kwargs = dict(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            callback=callback,
            blocksize=4000,  # ~250ms per callback
        )
        if device_id is not None and device_id >= 0:
            kwargs['device'] = device_id

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


# ── Transcription worker (vosk) ───────────────────────────────────────────────

class TranscriptionWorker(QObject):
    caption_ready  = pyqtSignal(str)   # final sentence
    partial_ready  = pyqtSignal(str)   # in-progress text
    model_loaded   = pyqtSignal()
    error          = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._model = None
        self._rec   = None
        self._queue = queue.Queue()
        self._running = False

    @pyqtSlot()
    def run(self):
        try:
            from vosk import Model, KaldiRecognizer, SetLogLevel
            SetLogLevel(-1)
            self._model = Model(model_name="vosk-model-small-en-us-0.15")
            self._rec   = KaldiRecognizer(self._model, SAMPLE_RATE)
            self.model_loaded.emit()
        except Exception as e:
            self.error.emit(f"Failed to load speech model:\n{e}\n\n{traceback.format_exc()}")
            return

        self._running = True
        while self._running:
            try:
                pcm_bytes = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                if self._rec.AcceptWaveform(pcm_bytes):
                    result = json.loads(self._rec.Result())
                    text = result.get('text', '').strip()
                    if text:
                        self.caption_ready.emit(text)
                else:
                    partial = json.loads(self._rec.PartialResult())
                    text = partial.get('partial', '').strip()
                    self.partial_ready.emit(text)
            except Exception as e:
                print(f"[transcription error] {e}", flush=True)

    def enqueue(self, pcm_bytes: bytes):
        self._queue.put(pcm_bytes)

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
    selection-background-color: #45475a;
}
QLabel#timerLabel {
    font-size: 26px;
    font-weight: bold;
    color: #6c7086;
    font-family: 'Courier New', monospace;
}
QLabel#timerLabel[recording="true"] { color: #f38ba8; }
QLabel#recDot { font-size: 20px; color: #313244; }
QLabel#recDot[recording="true"] { color: #f38ba8; }
QLabel#partialLabel {
    color: #6c7086;
    font-style: italic;
    font-size: 12px;
    padding: 2px 10px 4px 10px;
    min-height: 18px;
}
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


# ── Main window ───────────────────────────────────────────────────────────────

class MeetingRecorder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.is_recording  = False
        self.start_time    = None
        self.output_dir    = Path.home() / "MeetingRecordings"
        self.output_dir.mkdir(exist_ok=True)
        self.session_dir   = None
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
        self.setMinimumSize(720, 640)
        self.resize(820, 700)
        self.setStyleSheet(APP_STYLE)

        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(22, 18, 22, 10)
        vbox.setSpacing(14)

        # Title row
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

        # Settings
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
        opts_row.addWidget(self.cb_screen)
        opts_row.addWidget(self.cb_transcript)
        opts_row.addStretch()
        sg.addLayout(opts_row)
        vbox.addWidget(settings)

        # Start / Stop
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.start_btn = QPushButton("▶  Start Recording")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setProperty("recording", "false")
        self.start_btn.clicked.connect(self._toggle)
        btn_row.addWidget(self.start_btn)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        # Live captions
        caps = QGroupBox("Live Captions")
        cl = QVBoxLayout(caps)
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

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Loading speech recognition model…")

        self._timer = QTimer()
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
                f"Could not start transcription thread:\n{e}\n\n{traceback.format_exc()}"
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
        self.is_recording = True
        self.start_time   = time.monotonic()

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
