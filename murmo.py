#!/usr/bin/env python3
"""
murmo - just talk: record audio and transcribe it with OpenAI Whisper (local, offline).

How it works:
  1. Recording starts as soon as you speak
  2. Stop talking (or press Enter) -> recording stops, transcription runs
  3. The transcript is printed and copied to the clipboard (--out also saves .wav/.txt)

Examples:
  murmo
  murmo --model medium --language en
  murmo --file my_recording.m4a   # transcribe an existing file
"""

import argparse
import math
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

SAMPLE_RATE = 16000  # Whisper works internally with 16 kHz
SILENCE_THRESHOLD_DB = -30  # below this dB, consider it silence
PRE_ROLL_SAMPLES = SAMPLE_RATE // 2  # keep 0.5 s of audio before speech starts
PUNCTUATION_PROMPTS = {
    "en": "Hello, welcome. Today, we'll talk about the plan: what's next, and why it matters.",
    "de": "Hallo, willkommen. Heute sprechen wir über den Plan: was als Nächstes kommt, und warum.",
}


class _NoDefault:
    """Sentinel for distinguishing 'no args passed' from 'explicit None'."""
    pass


_NO_DEFAULT = _NoDefault()


def copy_to_clipboard(text: str) -> bool:
    """Copy text to the system clipboard via platform-native tools.

    Returns True if the text was copied, False if no clipboard tool
    was found or the command failed.
    """
    platform = sys.platform
    if platform == "darwin":
        cmd = ["pbcopy"]
    elif platform == "win32":
        cmd = ["clip"]
    else:
        # Try xclip first (X11), fall back to xsel
        try:
            subprocess.run(["which", "xclip"], capture_output=True, check=True)
            cmd = ["xclip", "-selection", "clipboard"]
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(["which", "xsel"], capture_output=True, check=True)
                cmd = ["xsel", "--clipboard", "--input"]
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False

    try:
        subprocess.run(cmd, input=text.encode("utf-8"), check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _rms_db(chunk: np.ndarray) -> float:
    """Return the RMS energy in dB of an audio chunk."""
    rms = math.sqrt(np.mean(np.square(chunk)))
    if rms == 0:
        return float("-inf")
    return 20 * math.log10(rms)


def record(
    *,
    silence_threshold: float = SILENCE_THRESHOLD_DB,
    silence_duration: float = 3.0,
    max_duration: float = 300.0,
) -> np.ndarray:
    """Records from the default microphone until Enter is pressed, silence is detected, or the
    maximum duration is reached.

    The microphone opens immediately, but recording only counts once speech is detected —
    leading silence is discarded (except a short pre-roll) and never triggers auto-stop.
    Press Enter at any time to stop early.

    Args:
        silence_threshold: dB level below which audio is considered silence.
        silence_duration: seconds of silence after speech before auto-stop triggers.
        max_duration: maximum recording time in seconds after speech starts (default 5 min).
    """
    chunks = []
    running = threading.Event()
    running.set()
    silence_start: float | None = None
    speech_start: float | None = None

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        chunks.append(indata.copy())

    def wait_for_enter() -> None:
        """Listen for Enter on a background thread and signal stop."""
        try:
            input()
        except EOFError:
            pass
        running.clear()

    # Print the instructions on their own line so the status line below can be
    # redrawn with \r without clobbering them.
    print("Listening - start speaking. Press Enter to STOP recording.")
    enter_thread = threading.Thread(target=wait_for_enter, daemon=True)
    enter_thread.start()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        while running.is_set():
            if chunks:
                current_rms = _rms_db(chunks[-1].flatten())
                is_silent = current_rms < silence_threshold

                if speech_start is None:
                    if is_silent:
                        # Drop leading silence, keeping a short pre-roll so the first
                        # syllable isn't cut off.
                        while len(chunks) > 1 and sum(len(c) for c in chunks[1:]) >= PRE_ROLL_SAMPLES:
                            chunks.pop(0)
                    else:
                        speech_start = time.time()
                elif is_silent:
                    if silence_start is None:
                        silence_start = time.time()
                    elif (time.time() - silence_start) >= silence_duration:
                        print("\nSilence detected — recording stopped.")
                        running.clear()
                        continue
                else:
                    silence_start = None

            if speech_start is None:
                print("\r   waiting for speech ...\033[K", end="", flush=True)
            else:
                elapsed = time.time() - speech_start
                if elapsed > max_duration:
                    print("\nMax recording duration reached.")
                    running.clear()
                    continue
                elapsed_sec = int(elapsed)
                print(f"\r   recording {elapsed_sec // 60:02d}:{elapsed_sec % 60:02d}\033[K", end="", flush=True)
            time.sleep(0.15)

    if speech_start is None:
        sys.exit("\nNo speech detected — nothing was recorded.")
    print("\nRecording finished.")
    return np.concatenate(chunks, axis=0).flatten()


def save_wav(path: Path, audio: np.ndarray) -> None:
    wavfile.write(path, SAMPLE_RATE, (np.clip(audio, -1, 1) * 32767).astype(np.int16))


def transcribe(
    audio_or_path,
    model_name: str,
    language: str | None,
    punctuate: bool = False,
) -> str:
    import whisper  # imported here so the script starts faster

    print(f"Loading Whisper model '{model_name}' (it is downloaded on first use) ...")
    model = whisper.load_model(model_name)
    print(f"Loaded model: {model_name} "
          f"(device: {model.device}, dims: {model.dims})")

    print("Transcribing ...")
    options = {"language": language, "fp16": False}  # fp16 is not supported on the Mac CPU
    if punctuate:
        # Whisper mimics the style of the initial prompt, so a well-punctuated
        # sentence nudges it towards proper punctuation and capitalization.
        options["initial_prompt"] = PUNCTUATION_PROMPTS.get(language, PUNCTUATION_PROMPTS["en"])
    result = model.transcribe(audio_or_path, **options)
    return result["text"].strip()


def _append_transcript(output_path: Path, text: str) -> None:
    """Append a transcript to an existing file, or create it."""
    if output_path.exists() and output_path.stat().st_size > 0:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write("\n\n" + text + "\n")
    else:
        output_path.write_text(text + "\n", encoding="utf-8")


def _find_latest_transcript(folder: Path) -> Path:
    """Return the most recently modified .txt file in a folder."""
    txts = sorted(folder.glob("*.txt"), key=lambda p: p.stat().st_mtime)
    return txts[-1] if txts else folder / "transcript.txt"


def main(args: list[str] | _NoDefault = _NO_DEFAULT) -> None:
    # args=_NO_DEFAULT (default) → read from sys.argv[1:] (normal CLI invocation)
    # args=[] (explicit empty list) → use no arguments (tests)
    # args=["--model", "turbo"] → use those arguments (tests / API)
    if args is _NO_DEFAULT:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="murmo",
        description="murmo - just talk: record audio and transcribe it with Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="small",
                        help="tiny, base, small, medium, large, turbo (default: small)")
    parser.add_argument("--language", default=None,
                        help="Language code, e.g. de or en (default: auto-detect)")
    parser.add_argument("--out", default=None, help="Output folder (default: print to stdout only)")
    parser.add_argument("--clipboard", action=argparse.BooleanOptionalAction, default=True,
                        help="Copy the transcript to the system clipboard "
                             "(default: on; use --no-clipboard to disable)")
    parser.add_argument("--append", action="store_true",
                        help="Append transcript to the latest file in --out instead of creating a new one")
    parser.add_argument("--punctuate", action="store_true",
                        help="Nudge Whisper towards proper punctuation and capitalization "
                             "(via a punctuated initial prompt)")
    parser.add_argument("--file", default=None,
                        help="Transcribe an existing audio file instead of recording (requires ffmpeg)")

    # Recording options (only relevant for live recording)
    rec_group = parser.add_argument_group("Recording options")
    rec_group.add_argument("--silence-threshold", type=float, default=SILENCE_THRESHOLD_DB,
                           help="dB below which audio is silence (default: -30)")
    rec_group.add_argument("--silence-duration", type=float, default=3.0,
                           help="Seconds of silence before auto-stop (default: 3)")
    rec_group.add_argument("--max-duration", type=float, default=300.0,
                           help="Max recording duration in seconds (default: 300 / 5 min)")

    parsed = parser.parse_args(args if args is not None else sys.argv[1:])

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if parsed.punctuate:
        print("Enhancing punctuation and capitalization ...")

    if parsed.file:
        text = transcribe(parsed.file, parsed.model, parsed.language, parsed.punctuate)
    else:
        audio = record(
            silence_threshold=parsed.silence_threshold,
            silence_duration=parsed.silence_duration,
            max_duration=parsed.max_duration,
        )
        text = transcribe(audio, parsed.model, parsed.language, parsed.punctuate)

    if parsed.out:
        out = Path(parsed.out)
        out.mkdir(parents=True, exist_ok=True)

        if parsed.file:
            base = out / f"{Path(parsed.file).stem}_{timestamp}"
        else:
            base = out / f"recording_{timestamp}"
            save_wav(base.with_suffix(".wav"), audio)

        txt_path = base.with_suffix(".txt")

        if parsed.append:
            txt_path = _find_latest_transcript(out)
            _append_transcript(txt_path, text)
        else:
            txt_path.write_text(text + "\n", encoding="utf-8")

        print("\n" + text + "\n")
        print(f"Transcript saved: {txt_path}")
    else:
        # Print the transcript unwrapped and unframed so it can be copied as-is.
        print("\n" + text + "\n")
        print("Hint: run with --out <folder> to save audio and transcript files.")

    if parsed.clipboard:
        if copy_to_clipboard(text):
            print("Transcript copied to clipboard.")
        else:
            print("Warning: could not copy to clipboard (no clipboard tool available).", file=sys.stderr)


if __name__ == "__main__":
    main()
