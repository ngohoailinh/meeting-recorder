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

print("[8] vosk import")
try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
    ok("vosk imported")
except Exception as e:
    fail("vosk import", e)

print("[9] loading vosk small English model (downloads ~40MB on first run)...")
try:
    SetLogLevel(-1)
    model = Model(model_name="vosk-model-small-en-us-0.15")
    ok("vosk model loaded")
except Exception as e:
    fail("vosk Model()", e)

print("[10] short transcription test")
try:
    import json
    rec = KaldiRecognizer(model, 16000)
    silence = np.zeros(16000, dtype=np.int16).tobytes()
    rec.AcceptWaveform(silence)
    result = json.loads(rec.FinalResult())
    ok(f"transcription works — result: {result}")
except Exception as e:
    fail("KaldiRecognizer transcribe", e)

print("\nAll diagnostics passed — run: python main.py")
