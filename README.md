# murmo

**Just talk — murmo turns your speech into text, right in your clipboard.**

A small command-line tool that records audio from your microphone and transcribes it locally with [OpenAI Whisper](https://github.com/openai/whisper). Nothing is sent to the cloud: the model runs on your machine.

Works on **macOS**, **Linux**, and **Windows**.

## Features

- **Auto-stop on silence** — recording starts when you speak; 3 seconds of silence after speaking stops it automatically
- **Append mode** — chain multiple recordings into a single transcript file (perfect for meetings)
- **Punctuation enhancement** — nudge Whisper towards proper punctuation and capitalization with a punctuated initial prompt
- **Clipboard copy** — the transcript is copied to the clipboard automatically, no manual Cmd+C needed (`--no-clipboard` to disable)
- **Plain terminal output** — the transcript is printed as-is, easy to copy

## Requirements

- Python 3.10 or newer (check with `python3 --version`)

## Installation

### 1. Install system dependencies

**macOS** (requires Homebrew):

```bash
brew install ffmpeg portaudio
```

**Ubuntu / Debian** (Linux):

```bash
sudo apt install ffmpeg libportaudio-dev
```

**Other Linux** (Fedora, Arch, …):

```bash
# Fedora
sudo dnf install ffmpeg portaudio-devel

# Arch
sudo pacman -S ffmpeg portaudio
```

**Windows**:

```bash
# Install FFmpeg via winget:
winget install Gyan.FFmpeg

# Then add it to PATH (it installs to C:\Program Files\FFmpeg\bin\).
# Add that path to your environment:
#   Settings → System → About → Advanced system settings → Environment Variables
#   Add C:\Program Files\FFmpeg\bin to your Path variable.
```

> **Windows:** No extra PortAudio install is needed — the `sounddevice` wheels for Windows ship with PortAudio included.

### 2. Install murmo

Pick one of the following.

**From PyPI as a global command (recommended)** — installs murmo into its own isolated environment and puts `murmo` on your PATH. Requires [pipx](https://pipx.pypa.io/) or [uv](https://docs.astral.sh/uv/):

```bash
pipx install murmo
# or
uv tool install murmo
```

To uninstall later: `pipx uninstall murmo` or `uv tool uninstall murmo`.

**With pip** — into the currently active (virtual) environment:

```bash
pip install murmo
```

**Directly from GitHub** — e.g. to get the latest unreleased version:

```bash
pipx install git+https://github.com/AICDEV/murmo.git
# or
uv tool install git+https://github.com/AICDEV/murmo.git
# or, into the active environment
pip install git+https://github.com/AICDEV/murmo.git
```

Append `@v0.1.0` (or any tag, branch, or commit) to the URL to install a specific version.

> **Note:** pip installs only the Python dependencies. FFmpeg and PortAudio (step 1) must be installed separately. Whisper pulls in PyTorch, so the first install is a few hundred MB.

**From a local clone (for development)** — useful if you want to hack on the code:

```bash
git clone https://github.com/AICDEV/murmo.git
cd murmo
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

With `-e` (editable), changes to `murmo.py` take effect immediately. The `murmo` command is available whenever the virtual environment is active.

## Usage

Run:

```bash
murmo
```

Then:

1. Just start talking — recording begins when speech is detected (silence before that is ignored).
2. **Just stop talking** — 3 seconds of silence stops it automatically. Or press **Enter** to stop manually.
3. Whisper transcribes the audio and prints the text in the terminal.

The Whisper model is downloaded on first use, which can take a moment.

> **macOS:** On the first run, the system prompts whether Terminal may access the microphone. Click **Allow**.
>
> **Windows:** If you get a `PortAudioError` / `SoundDeviceError`, check that a microphone is connected and your terminal has microphone access (Settings → Privacy → Microphone). For `--file`, make sure FFmpeg is on your PATH.

### Options

| Option | Description | Default |
|---|---|---|
| `--model` | Model size: `tiny`, `base`, `small`, `medium`, `large`, `turbo` | `small` |
| `--language` | Language code, e.g. `en` or `de` | auto-detect |
| `--out` | Save audio and transcript to a folder | prints to stdout only |
| `--no-clipboard` | Don't copy the transcript to the clipboard | clipboard on |
| `--append` | Append to the latest transcript in `--out` instead of creating a new file | off |
| `--punctuate` | Enhance output with proper punctuation and capitalization | off |
| `--file` | Transcribe an existing audio file instead of recording (requires ffmpeg) | — |

### Recording options

| Option | Description | Default |
|---|---|---|
| `--silence-threshold` | dB level below which audio counts as silence | -30 |
| `--silence-duration` | Seconds of silence before auto-stop | 3 |
| `--max-duration` | Maximum recording time in seconds | 300 (5 min) |

### Examples

```bash
# Quick dictation: record, auto-stop on silence, copy to clipboard
murmo

# Punctuation-enhanced recording, saved to a folder
murmo --out ~/Documents/notes --punctuate

# Meeting mode: each run appends its transcript to the latest file in the folder
murmo --out ~/Documents/meeting --append

# Transcribe an existing file with punctuation, English only
murmo --file interview.m4a --language en --punctuate

# Custom silence settings (picks up quieter speech, stops faster)
murmo --silence-threshold -35 --silence-duration 1.5
```

## Choosing a model

Bigger models are more accurate but slower. Whisper runs on the CPU on a Mac, so transcription can take longer than the recording itself.

| Model | Speed | Accuracy |
|---|---|---|
| `tiny` / `base` | fastest | basic |
| `small` | good balance (default) | good |
| `medium` | slow | very good |
| `large` / `turbo` | slowest / fast for its size | best |

Setting `--language` explicitly usually improves results compared to auto-detection.

## Troubleshooting

- **No sound recorded / permission error**
  - **macOS:** System Settings → Privacy & Security → Microphone → enable your terminal app (Terminal, iTerm, VS Code, …) → restart terminal.
  - **Windows:** Ensure your terminal has microphone access (Settings → Privacy → Microphone).
  - **Linux:** Add your user to the `audio` group: `sudo usermod -a -G audio $USER`, then log out and back in.
- **PortAudio library not found**
  - **macOS:** `brew install portaudio`, then `pip install --force-reinstall sounddevice`.
  - **Linux:** `sudo apt install libportaudio-dev` (Debian/Ubuntu) or equivalent, then reinstall `sounddevice`.
  - **Windows:** PortAudio is bundled with `sounddevice`; run `pip install --force-reinstall sounddevice`.
- **ffmpeg not found** (only when using `--file`)
  - **macOS:** `brew install ffmpeg`.
  - **Linux:** `sudo apt install ffmpeg`.
  - **Windows:** Ensure FFmpeg is on your PATH.
- **Wrong microphone is used** — The script uses the system default input. Change it in your OS sound settings.
- **Very slow transcription** — Use a smaller model, e.g. `--model base`.
- **Recording stops too early** — Lower `--silence-threshold` (e.g. `-40`) so quieter speech still counts, or use `--silence-duration 5`.
- **Recording doesn't stop** — Background noise is above the threshold: raise `--silence-threshold` (e.g. `-25`) or decrease `--silence-duration`.
- **FP16 is not supported on CPU** (macOS) — Harmless. The script already disables `fp16` for the Mac CPU.

## Testing

Run the test suite with [pytest](https://docs.pytest.org/):

```bash
pip install -e ".[dev]"
python3 -m pytest -v
```

34 tests covering WAV writing, argument parsing, output directory creation, both record and file modes, stdout formatting, clipboard integration, silence detection, append mode, and punctuation enhancement.
