# Meeting Recorder

A Windows 11 desktop app that records meetings, shows live captions, and saves everything to disk.

## Features
- **One-click** Start / Stop recording
- **Live English captions** powered by Whisper (base model, runs offline)
- **Audio recording** from microphone (saved as `audio.wav`)
- **Screen recording** at 10 fps (saved as `screen.mp4`)
- **Transcript** saved as `transcript.txt`
- Dark UI, blinking REC indicator, elapsed timer

Each session is saved in a timestamped folder:
```
~/MeetingRecordings/
└── meeting_2025-01-15_09-30-00/
    ├── audio.wav
    ├── screen.mp4   (if screen recording enabled)
    └── transcript.txt
```

---

## Setup (Windows 11)

### 1. Install Python 3.10+
Download from https://www.python.org — check **"Add Python to PATH"** during install.

### 2. Install dependencies
Open a terminal in this folder and run:
```cmd
pip install -r requirements.txt
```
> First run will also download the Whisper base model (~74 MB). Subsequent runs are instant.

### 3. Run the app
```cmd
python main.py
```

---

## Recording Zoom / Teams / Google Meet audio

By default the app records from your **microphone only**. To also capture the call audio
(the other participants' voices), you have two options:

### Option A — Enable "Stereo Mix" (free, built-in)
1. Right-click the speaker icon in the taskbar → **Sound settings**
2. Scroll to **More sound settings** → **Recording** tab
3. Right-click in the empty area → **Show Disabled Devices**
4. If **Stereo Mix** appears, right-click it → **Enable**
5. In the Meeting Recorder app, select **Stereo Mix** from the microphone dropdown

### Option B — VB-CABLE virtual audio device (recommended, works with headphones)
1. Download VB-CABLE from https://vb-audio.com/Cable/
2. Install it (free)
3. In Zoom/Teams/Meet, set the **speaker output** to "CABLE Input (VB-Audio)"
4. Route audio back to your headphones so you can still hear the meeting:
   - Right-click the speaker icon → **Sound settings** → **More sound settings**
   - Go to the **Recording** tab → right-click **CABLE Output (VB-Audio)** → **Properties**
   - Click the **Listen** tab → check **"Listen to this device"**
   - Set **"Playback through this device"** to your headphones → click OK
5. In the Meeting Recorder app, select **CABLE Output (VB-Audio)** as the microphone

The audio chain: **Zoom/Teams → CABLE → headphones** (you hear it) **+ Meeting Recorder** (app records it)

---

## Tips
- The Whisper **base** model gives good accuracy with ~1–2 second caption delay.
  For faster captions at lower accuracy, change `model_size="base"` to `"tiny"` in `main.py`.
- Screen recording adds CPU load. Uncheck **Record screen** if you only need audio + captions.
- All processing is **offline** — no data leaves your machine.
