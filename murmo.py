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
from collections import deque
from functools import lru_cache
import math
import queue
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
    if chunk.size == 0:
        return float("-inf")
    power = float(np.dot(chunk, chunk)) / chunk.size
    if power <= 0:
        return float("-inf")
    return 10 * math.log10(power)


def record(
    *,
    silence_threshold: float = SILENCE_THRESHOLD_DB,
    silence_duration: float | None = 3.0,
    max_duration: float = 300.0,
) -> np.ndarray:
    """Record from the default microphone until Enter is pressed, silence is detected, or the
    maximum duration is reached.

    The microphone opens immediately, but recording only counts once speech is detected —
    leading silence is discarded (except a short pre-roll) and never triggers auto-stop.
    Press Enter at any time to stop early. Pass silence_duration=None to disable
    silence-based auto-stop entirely.

    Args:
        silence_threshold: dB level below which audio is considered silence.
        silence_duration: seconds of silence after speech before auto-stop triggers,
            or None to disable silence-based auto-stop.
        max_duration: maximum recording time in seconds after speech starts (default 5 min).
    """
    audio_queue: queue.Queue[tuple[np.ndarray, float]] = queue.Queue()
    pre_roll_chunks: deque[np.ndarray] = deque()
    recorded_chunks: list[np.ndarray] = []
    running = threading.Event()
    running.set()
    pre_roll_samples = 0
    recorded_samples = 0
    speech_samples = 0
    silence_samples = 0
    speech_started = False
    silence_limit_samples = None if silence_duration is None else max(0, int(silence_duration * SAMPLE_RATE))
    max_duration_samples = max(1, int(max_duration * SAMPLE_RATE))

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        chunk = indata[:, 0].copy()
        audio_queue.put((chunk, _rms_db(chunk)))

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
        last_status = 0.0
        while running.is_set():
            processed_audio = False

            while True:
                try:
                    chunk, current_rms = audio_queue.get_nowait()
                except queue.Empty:
                    break

                processed_audio = True
                is_silent = current_rms < silence_threshold

                if not speech_started:
                    if is_silent:
                        pre_roll_chunks.append(chunk)
                        pre_roll_samples += len(chunk)
                        while pre_roll_chunks and pre_roll_samples > PRE_ROLL_SAMPLES:
                            pre_roll_samples -= len(pre_roll_chunks.popleft())
                        continue

                    speech_started = True
                    if pre_roll_chunks:
                        recorded_chunks.extend(pre_roll_chunks)
                        recorded_samples += pre_roll_samples
                        pre_roll_chunks.clear()
                        pre_roll_samples = 0

                recorded_chunks.append(chunk)
                chunk_samples = len(chunk)
                recorded_samples += chunk_samples
                speech_samples += chunk_samples

                if speech_samples >= max_duration_samples:
                    print("\nMax recording duration reached.")
                    running.clear()
                    break

                if silence_limit_samples is None:
                    continue

                if is_silent:
                    silence_samples += chunk_samples
                    if silence_samples >= silence_limit_samples:
                        print("\nSilence detected — recording stopped.")
                        running.clear()
                        break
                else:
                    silence_samples = 0

            now = time.monotonic()
            if now - last_status >= 0.15 or processed_audio:
                if not speech_started:
                    print("\r   waiting for speech ...\033[K", end="", flush=True)
                else:
                    elapsed_sec = int(speech_samples / SAMPLE_RATE)
                    print(f"\r   recording {elapsed_sec // 60:02d}:{elapsed_sec % 60:02d}\033[K", end="", flush=True)
                last_status = now

            time.sleep(0.15)

    if not speech_started:
        sys.exit("\nNo speech detected — nothing was recorded.")
    print("\nRecording finished.")
    if recorded_samples == 0:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(recorded_chunks)


@lru_cache(maxsize=None)
def _load_whisper_model(model_name: str):
    import whisper

    return whisper.load_model(model_name)


def save_wav(path: Path, audio: np.ndarray) -> None:
    wavfile.write(path, SAMPLE_RATE, (np.clip(audio, -1, 1) * 32767).astype(np.int16))


def _transcribe_options(language: str | None, punctuate: bool) -> dict[str, object]:
    options: dict[str, object] = {"language": language, "fp16": False}  # fp16 is not supported on the Mac CPU
    if punctuate:
        # Whisper mimics the style of the initial prompt, so a well-punctuated
        # sentence nudges it towards proper punctuation and capitalization.
        options["initial_prompt"] = PUNCTUATION_PROMPTS.get(language, PUNCTUATION_PROMPTS["en"])
    return options


def _transcribe_with_model(model, audio_or_path, language: str | None, punctuate: bool) -> str:
    result = model.transcribe(audio_or_path, **_transcribe_options(language, punctuate))
    return result["text"].strip()


def transcribe(
    audio_or_path,
    model_name: str,
    language: str | None,
    punctuate: bool = False,
) -> str:
    print(f"Loading Whisper model '{model_name}' (it is downloaded on first use) ...")
    model = _load_whisper_model(model_name)
    print(f"Loaded model: {model_name} "
          f"(device: {model.device}, dims: {model.dims})")

    print("Transcribing ...")
    return _transcribe_with_model(model, audio_or_path, language, punctuate)


def benchmark_transcription(
    audio_or_path,
    model_name: str,
    language: str | None,
    punctuate: bool = False,
    runs: int = 3,
) -> tuple[str, dict[str, float | int]]:
    """Run repeated transcriptions and return the last transcript plus timing stats."""
    if runs < 1:
        raise ValueError("runs must be at least 1")

    print(f"Loading Whisper model '{model_name}' (it is downloaded on first use) ...")
    model = _load_whisper_model(model_name)
    print(f"Loaded model: {model_name} "
          f"(device: {model.device}, dims: {model.dims})")

    print(f"Benchmarking transcription ({runs} run{'s' if runs != 1 else ''}, model load excluded) ...")
    timings: list[float] = []
    text = ""
    for _ in range(runs):
        start = time.perf_counter()
        text = _transcribe_with_model(model, audio_or_path, language, punctuate)
        timings.append(time.perf_counter() - start)

    return text, {
        "runs": runs,
        "min": min(timings),
        "avg": sum(timings) / len(timings),
        "max": max(timings),
    }


def _print_benchmark_stats(stats: dict[str, float | int]) -> None:
    print("Benchmark: transcription only (model load excluded)")
    print(f"  runs: {stats['runs']}")
    print(f"  min : {stats['min']:.2f}s")
    print(f"  avg : {stats['avg']:.2f}s")
    print(f"  max : {stats['max']:.2f}s")


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
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark transcription time for the current input")
    parser.add_argument("--benchmark-runs", type=int, default=3,
                        help="How many transcription runs to measure with --benchmark (default: 3)")

    # Recording options (only relevant for live recording)
    rec_group = parser.add_argument_group("Recording options")
    rec_group.add_argument("--silence-threshold", type=float, default=SILENCE_THRESHOLD_DB,
                           help="dB below which audio is silence (default: -30)")
    rec_group.add_argument("--silence-duration", type=float, default=3.0,
                           help="Seconds of silence before auto-stop (default: 3)")
    rec_group.add_argument("--silence-stop", action=argparse.BooleanOptionalAction, default=True,
                           help="Enable automatic stop after silence "
                                "(default: on; use --no-silence-stop to require Enter)")
    rec_group.add_argument("--max-duration", type=float, default=300.0,
                           help="Max recording duration in seconds (default: 300 / 5 min)")

    parsed = parser.parse_args(args if args is not None else sys.argv[1:])
    if parsed.benchmark_runs < 1:
        parser.error("--benchmark-runs must be at least 1")
    if not parsed.benchmark and parsed.benchmark_runs != 3:
        parser.error("--benchmark-runs requires --benchmark")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    benchmark_stats: dict[str, float | int] | None = None

    if parsed.punctuate:
        print("Enhancing punctuation and capitalization ...")

    if parsed.file:
        source = parsed.file
    else:
        audio = record(
            silence_threshold=parsed.silence_threshold,
            silence_duration=parsed.silence_duration if parsed.silence_stop else None,
            max_duration=parsed.max_duration,
        )
        source = audio

    if parsed.benchmark:
        text, benchmark_stats = benchmark_transcription(
            source,
            parsed.model,
            parsed.language,
            parsed.punctuate,
            parsed.benchmark_runs,
        )
    else:
        text = transcribe(source, parsed.model, parsed.language, parsed.punctuate)

    if benchmark_stats is not None:
        _print_benchmark_stats(benchmark_stats)

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
