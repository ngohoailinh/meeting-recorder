# Meeting Recorder

A Windows 11 desktop app that records meetings, shows live captions, and generates AI summaries — all offline-first, no cloud required for core features.

---

## Features at a glance

| Feature | Detail |
|---|---|
| One-click record | Start / Stop button with live timer |
| Live captions | Real-time English captions via local Whisper (vosk) |
| Audio recording | Saves `audio.wav` from microphone or system audio |
| Screen recording | Saves `screen.mp4` at 10 fps |
| Final video | Auto-merges screen + audio into `meeting_final.mp4` |
| Transcript | Saves `transcript.txt` from live captions |
| AI summary | Summarize any session with Anthropic / OpenAI / Gemini / any custom OpenAI-compatible API |
| Audio fallback | Can transcribe `audio.wav` or `meeting_final.mp4` to generate summaries even without a live transcript |

---

## Quick start

### 1. Install Python 3.10 or newer

Download from https://www.python.org — check **"Add Python to PATH"** during install.

> Python 3.14 is supported. Python 3.11 / 3.12 also work.

### 2. Install dependencies

Open a terminal in this folder:

```cmd
pip install -r requirements.txt
```

For AI summaries, install the provider(s) you plan to use:

```cmd
pip install anthropic          # Anthropic Claude
pip install openai             # OpenAI GPT, or any Custom OpenAI-compatible server
pip install google-genai       # Google Gemini
```

> The `openai` package is also required for Custom providers (Ollama, Groq, Azure, etc.).

### 3. Run the app

```cmd
python main.py
```

On first launch the app downloads the vosk speech model (~40 MB). Subsequent launches are instant.

---

## Recording a meeting

1. Select your **microphone** from the dropdown (see [Capturing system audio](#capturing-system-audio-zoomteamsmeet) for Zoom/Teams/Meet).
2. Choose where to **save** recordings (defaults to `~/MeetingRecordings`).
3. Check the options you want:
   - **Record screen** — captures your screen at 10 fps
   - **Save transcript** — writes captions to `transcript.txt`
   - **Merge to final video** — combines screen + audio into `meeting_final.mp4` after stopping
4. Click **▶ Start Recording**.
5. Click **■ Stop Recording** when done.

While recording:
- The timer counts up and the **●** dot blinks red.
- **Live captions** stream in real time — grey italic text shows words being recognized; confirmed sentences appear in white.
- Click **Clear** to reset the caption display without affecting the saved transcript.

Each session is saved in a timestamped folder:

```
~/MeetingRecordings/
└── meeting_2025-06-27_09-30-00/
    ├── audio.wav           ← raw audio from your mic
    ├── screen.mp4          ← raw screen recording (no audio)
    ├── meeting_final.mp4   ← merged screen + audio (if enabled)
    └── transcript.txt      ← live caption text
```

---

## Capturing system audio (Zoom / Teams / Meet)

By default the app records only your **microphone**. To also capture the other participants' voices:

### Option A — Stereo Mix (free, built-in, no headphone issues)

1. Right-click the speaker icon → **Sound settings** → **More sound settings**
2. **Recording** tab → right-click empty area → **Show Disabled Devices**
3. If **Stereo Mix** appears, right-click → **Enable**
4. In the app's microphone dropdown, select **Stereo Mix**

> Stereo Mix captures all system audio. Your microphone will not be recorded separately.

### Option B — VB-CABLE (recommended when using headphones)

VB-CABLE creates a virtual audio cable so you can capture call audio *and* keep hearing through your headphones.

1. Download VB-CABLE from https://vb-audio.com/Cable/ and install it (free, restart required)
2. In **Zoom / Teams / Meet**, set the **speaker output** to `CABLE Input (VB-Audio Virtual Cable)`
3. Route audio back to your headphones:
   - Right-click speaker icon → **Sound settings** → **More sound settings**
   - **Recording** tab → right-click **CABLE Output (VB-Audio Virtual Cable)** → **Properties**
   - **Listen** tab → check **"Listen to this device"** → set playback device to your headphones → OK
4. In the app's microphone dropdown, select **CABLE Output (VB-Audio Virtual Cable)**

Audio chain:
```
Zoom/Teams/Meet
      ↓  (speaker = CABLE Input)
  VB-CABLE
      ├──→ your headphones   (via "Listen to this device")
      └──→ Meeting Recorder  (mic input = CABLE Output)
```

---

## AI Summaries

Open the **📝 Summaries** tab.

### Setup

1. **Choose a provider** — Anthropic, OpenAI, Gemini, or Custom.
2. **Paste your API key** — stored locally in `~/.meeting_recorder.json`, never sent anywhere except the selected API.
3. Click **Show** to reveal / hide the key.

### Generating summaries

- **Summarize Selected** — select one or more sessions from the list, then click the button.
- **Summarize All** — processes every eligible session in the recordings folder.
- Double-click a session marked **✓** to reload its saved summary instantly.
- **Copy** button copies the summary text to clipboard.

Summaries are saved as `summary.txt` inside each session folder and loaded instantly on future runs.

### Session list colors

| Color | Meaning |
|---|---|
| Green ✓ | Summary already generated |
| White | Has `transcript.txt`, ready to summarize |
| Orange | Audio file only — needs transcription first |
| Grey | No transcript or audio found |

### Summarizing sessions without a transcript

Enable **"Transcribe audio if no transcript.txt"** in the provider settings.

When this is on, the app will automatically:
1. Read `audio.wav` if present
2. Or extract audio from `meeting_final.mp4` / `screen.mp4` using ffmpeg
3. Transcribe the audio locally with vosk
4. Save the result as `transcript.txt` for future use
5. Then send the transcript to the AI for summarization

This works for recordings made with captions off, or older recordings from other tools.

### Supported providers

#### Anthropic

- Model used: `claude-opus-4-8`
- Get API key: https://console.anthropic.com

#### OpenAI

- Model used: `gpt-4o`
- Get API key: https://platform.openai.com

#### Gemini

- Model used: `gemini-2.0-flash`
- Get API key: https://aistudio.google.com

#### Custom (OpenAI-compatible)

Works with any server that speaks the OpenAI chat completions API:

| Service | Base URL | Notes |
|---|---|---|
| **Ollama** (local) | `http://localhost:11434/v1` | API key = anything, e.g. `ollama` |
| **LM Studio** (local) | `http://localhost:1234/v1` | API key = anything |
| **Groq** | `https://api.groq.com/openai/v1` | Fast inference, free tier |
| **Together.ai** | `https://api.together.xyz/v1` | Many open-source models |
| **Azure OpenAI** | `https://YOUR.openai.azure.com/openai/deployments/DEPLOY` | Use Azure API key |
| **Mistral** | `https://api.mistral.ai/v1` | Mistral API key |

Set **Base URL** and **Model** in the UI. API key is optional for local servers.

---

## Config file

Settings are auto-saved to `~/.meeting_recorder.json`:

```json
{
  "provider": "Anthropic",
  "key_Anthropic": "sk-ant-api03-...",
  "key_OpenAI": "sk-proj-...",
  "key_Gemini": "AIzaSy...",
  "key_Custom": "ollama",
  "custom_base_url": "http://localhost:11434/v1",
  "custom_model": "llama3.2",
  "transcribe_audio": true
}
```

Keys for providers you haven't used yet will simply be absent. The file is plain JSON — you can edit it directly.

> Keep this file private. Do not commit it to version control.

---

## Tips

- **Caption accuracy** — vosk `small` model trades some accuracy for speed. For better accuracy at the cost of a larger download, change `model_name="vosk-model-small-en-us-0.15"` to `"vosk-model-en-us-0.22"` (~1.8 GB) in `main.py`.
- **Screen recording CPU** — uncheck **Record screen** if you only need audio + captions; screen capture at 10 fps adds noticeable CPU load on older machines.
- **Privacy** — all recording and transcription happens locally. Audio is only sent to an external API when you click Summarize.
- **Large transcript** — transcripts longer than 60 000 characters are automatically truncated before being sent to the AI. For very long meetings, consider splitting the session.
- **Re-summarize** — delete `summary.txt` from a session folder and click Summarize again to regenerate with a different provider or prompt.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Window closes immediately on launch | Run `python debug.py` to identify which library crashes |
| "Speech model failed" warning | Vosk model download may have been interrupted — delete `~/.cache/vosk` and restart |
| No audio devices shown | Check Windows sound settings; try running as administrator |
| `screen.mp4` has no audio | Expected — audio is in `audio.wav`; enable **Merge to final video** to combine them |
| Summarize returns an error | Check API key is correct and the selected model is available on your account |
| Custom provider times out | Verify the Base URL is reachable and the local server (Ollama/LM Studio) is running |
