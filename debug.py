import sys

def ok(label):
    print(f"  OK  {label}", flush=True)

def fail(label, e):
    print(f"  FAIL  {label}: {e}", flush=True)
    sys.exit(1)

print("Running diagnostics — each step will print OK or FAIL\n")

print("[1] numpy")
try:
    import numpy as np
    ok(f"numpy {np.__version__}")
except Exception as e:
    fail("numpy", e)

print("[2] scipy")
try:
    import scipy.io.wavfile
    import scipy
    ok(f"scipy {scipy.__version__}")
except Exception as e:
    fail("scipy", e)

print("[3] mss (screen capture)")
try:
    import mss
    ok("mss")
except Exception as e:
    fail("mss", e)

print("[4] opencv")
try:
    import cv2
    ok(f"cv2 {cv2.__version__}")
except Exception as e:
    fail("cv2", e)

print("[5] PyQt5")
try:
    from PyQt5.QtWidgets import QApplication
    ok("PyQt5")
except Exception as e:
    fail("PyQt5", e)

print("[6] sounddevice (audio)")
try:
    import sounddevice as sd
    ok(f"sounddevice {sd.__version__}")
except Exception as e:
    fail("sounddevice", e)

print("[7] sounddevice — list devices")
try:
    devs = sd.query_devices()
    ok(f"{len(devs)} audio devices found")
except Exception as e:
    fail("sd.query_devices()", e)

print("[8] faster-whisper import")
try:
    from faster_whisper import WhisperModel
    ok("faster_whisper imported")
except Exception as e:
    fail("faster_whisper import", e)

print("[9] loading Whisper base model with float32 (int8 crashes on some CPUs)...")
try:
    model = WhisperModel("base", device="cpu", compute_type="float32")
    ok("WhisperModel loaded")
except Exception as e:
    fail("WhisperModel()", e)

print("[10] short transcription test")
try:
    audio = np.zeros(16000, dtype=np.float32)
    segs, _ = model.transcribe(audio, language="en")
    list(segs)
    ok("transcription works")
except Exception as e:
    fail("transcribe()", e)

print("\nAll diagnostics passed — the crash is likely in the Qt event loop or threading.")
print("Check the updated main.py for the fix.")
