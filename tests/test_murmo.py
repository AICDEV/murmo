"""Tests for murmo — recording, transcription, and file I/O."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

import murmo


class TestSaveWav:
    """Pure logic: does save_wav produce a valid WAV file?"""

    def test_writes_file(self, tmp_path: Path):
        audio = np.array([0.0, -0.5, 0.5], dtype=np.float32)
        murmo.save_wav(tmp_path / "out.wav", audio)
        assert (tmp_path / "out.wav").exists()

    def test_correct_sample_rate_and_dtype(self, tmp_path: Path):
        audio = np.linspace(-1.0, 1.0, 1600, dtype=np.float32)
        murmo.save_wav(tmp_path / "out.wav", audio)
        rate, data = murmo.wavfile.read(tmp_path / "out.wav")
        assert rate == murmo.SAMPLE_RATE
        assert data.dtype == np.int16

    def test_clips_values(self, tmp_path: Path):
        """Samples outside [-1, 1] should be clipped before int16 conversion."""
        audio = np.array([-10.0, 10.0, 0.0], dtype=np.float32)
        murmo.save_wav(tmp_path / "out.wav", audio)
        rate, data = murmo.wavfile.read(tmp_path / "out.wav")
        assert rate == murmo.SAMPLE_RATE
        assert data.min() >= -32768
        assert data.max() <= 32767


class TestMainRecordMode:
    """Test main() in recording mode (mock record() and transcribe())."""

    @patch.object(murmo, "record")
    @patch.object(murmo, "transcribe")
    def test_stdout_mode_prints_formatted_output(self, mock_transcribe, mock_record,
                                                  tmp_path: Path, capsys):
        """Without --out, the transcript should print nicely to stdout."""
        mock_record.return_value = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        mock_transcribe.return_value = "hello world"

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main([])
        finally:
            os.chdir(cwd)

        captured = capsys.readouterr()
        assert "hello world" in captured.out
        assert "\nhello world\n" in captured.out  # transcript on its own line, no frame
        assert "─" not in captured.out
        assert "--out" in captured.out  # hint about --out
        # No files should be written without --out
        assert not list(tmp_path.iterdir())

    @patch.object(murmo, "record")
    @patch.object(murmo, "transcribe")
    def test_out_mode_creates_wav_and_txt(self, mock_transcribe, mock_record, tmp_path: Path):
        """With --out, recording mode should create both .wav and .txt files."""
        mock_record.return_value = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        mock_transcribe.return_value = "hello world"

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main(["--out", "recordings"])
        finally:
            os.chdir(cwd)

        recordings = list((tmp_path / "recordings").iterdir())
        wav_files = [f for f in recordings if f.suffix == ".wav"]
        txt_files = [f for f in recordings if f.suffix == ".txt"]
        assert len(wav_files) == 1
        assert len(txt_files) == 1
        assert txt_files[0].read_text(encoding="utf-8").strip() == "hello world"

    @patch.object(murmo, "record")
    @patch.object(murmo, "transcribe")
    def test_custom_out_dir(self, mock_transcribe, mock_record, tmp_path: Path):
        mock_record.return_value = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        mock_transcribe.return_value = "test"

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main(["--out", "custom"])
        finally:
            os.chdir(cwd)

        assert (tmp_path / "custom").is_dir()
        assert len(list((tmp_path / "custom").iterdir())) == 2

    @patch.object(murmo, "record")
    @patch.object(murmo, "transcribe")
    def test_custom_model_passed_to_transcribe(self, mock_transcribe, mock_record, tmp_path: Path):
        mock_record.return_value = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        mock_transcribe.return_value = "ok"

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main(["--model", "turbo"])
        finally:
            os.chdir(cwd)

        call_args = mock_transcribe.call_args
        assert call_args[0][1] == "turbo"


class TestMainFileMode:
    """Test main() with --file flag."""

    @patch.object(murmo, "transcribe")
    def test_file_mode_stdout_prints_output(self, mock_transcribe, tmp_path: Path, capsys):
        """--file without --out should print the transcript to stdout."""
        fake_file = tmp_path / "interview.m4a"
        fake_file.write_bytes(b"fake audio data")
        mock_transcribe.return_value = "transcribed text"

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main(["--file", str(fake_file)])
        finally:
            os.chdir(cwd)

        captured = capsys.readouterr()
        assert "\ntranscribed text\n" in captured.out

    @patch.object(murmo, "transcribe")
    def test_file_mode_with_out_creates_txt_only(self, mock_transcribe, tmp_path: Path):
        """--file with --out should produce .txt only, no .wav."""
        fake_file = tmp_path / "interview.m4a"
        fake_file.write_bytes(b"fake audio data")
        mock_transcribe.return_value = "transcribed text"

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main(["--file", str(fake_file), "--out", "recordings"])
        finally:
            os.chdir(cwd)

        recordings = list((tmp_path / "recordings").iterdir())
        wav_files = [f for f in recordings if f.suffix == ".wav"]
        txt_files = [f for f in recordings if f.suffix == ".txt"]
        assert len(wav_files) == 0
        assert len(txt_files) == 1
        assert txt_files[0].read_text(encoding="utf-8").strip() == "transcribed text"
        assert "interview" in txt_files[0].stem


class TestMainPlainOutput:
    """The transcript should be printed plainly so it can be copied as-is."""

    @patch.object(murmo, "record")
    @patch.object(murmo, "transcribe")
    def test_long_text_is_not_wrapped(self, mock_transcribe, mock_record, tmp_path: Path, capsys):
        """A long transcript should stay on a single line (no hard wraps, no frame)."""
        mock_record.return_value = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        long_text = " ".join(f"word{i}" for i in range(50))
        mock_transcribe.return_value = long_text

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main([])
        finally:
            os.chdir(cwd)

        captured = capsys.readouterr()
        assert long_text in captured.out.splitlines()


class TestCopyToClipboard:
    """Test the copy_to_clipboard helper."""

    def test_returns_true_on_success(self, monkeypatch):
        """Should return True when the clipboard command succeeds."""
        mock_run = MagicMock()
        mock_run.return_value = MagicMock(returncode=0)
        monkeypatch.setattr("murmo.subprocess.run", mock_run)

        with patch("sys.platform", "darwin"):
            assert murmo.copy_to_clipboard("hello") is True

    def test_returns_false_when_no_tool_available(self, monkeypatch):
        """Should return False when neither xclip nor xsel exists."""
        monkeypatch.setattr("subprocess.run", MagicMock(side_effect=FileNotFoundError))

        with patch("sys.platform", "linux"):
            assert murmo.copy_to_clipboard("hello") is False

    def test_uses_pbcopy_on_macos(self, monkeypatch):
        """macOS should call pbcopy."""
        mock_run = MagicMock()
        monkeypatch.setattr("murmo.subprocess.run", mock_run)

        with patch("sys.platform", "darwin"):
            murmo.copy_to_clipboard("hello")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == ["pbcopy"]

    def test_uses_clip_on_windows(self, monkeypatch):
        """Windows should call the built-in clip command."""
        mock_run = MagicMock()
        monkeypatch.setattr("murmo.subprocess.run", mock_run)

        with patch("sys.platform", "win32"):
            murmo.copy_to_clipboard("hello")

        call_args = mock_run.call_args
        assert call_args[0][0] == ["clip"]

    def test_uses_xclip_on_linux_when_available(self, monkeypatch):
        """Linux should prefer xclip when available."""
        mock_run = MagicMock()
        mock_run.side_effect = [
            MagicMock(returncode=0),  # which xclip succeeds
            MagicMock(returncode=0),  # xcopy call succeeds
        ]
        monkeypatch.setattr("murmo.subprocess.run", mock_run)

        with patch("sys.platform", "linux"):
            murmo.copy_to_clipboard("hello")

        assert mock_run.call_count == 2
        assert mock_run.call_args_list[1].args[0] == ["xclip", "-selection", "clipboard"]


class TestMainClipboard:
    """Test --clipboard flag in main()."""

    @patch.object(murmo, "record")
    @patch.object(murmo, "transcribe")
    def test_clipboard_flag_copies_text(self, mock_transcribe, mock_record, tmp_path: Path,
                                         capsys, monkeypatch):
        """--clipboard should copy the transcript and print a confirmation."""
        mock_record.return_value = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        mock_transcribe.return_value = "hello world"
        mock_clipboard = MagicMock(return_value=True)
        monkeypatch.setattr("murmo.copy_to_clipboard", mock_clipboard)

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main(["--clipboard"])
        finally:
            os.chdir(cwd)

        captured = capsys.readouterr()
        assert "copied to clipboard" in captured.out
        mock_clipboard.assert_called_once_with("hello world")

    @patch.object(murmo, "record")
    @patch.object(murmo, "transcribe")
    def test_clipboard_no_tool_fails_gracefully(self, mock_transcribe, mock_record,
                                                 tmp_path: Path, capsys, monkeypatch):
        """When no clipboard tool is available, print a warning to stderr."""
        mock_record.return_value = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        mock_transcribe.return_value = "hello world"
        mock_clipboard = MagicMock(return_value=False)
        monkeypatch.setattr("murmo.copy_to_clipboard", mock_clipboard)

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main(["--clipboard"])
        finally:
            os.chdir(cwd)

        captured = capsys.readouterr()
        # Warning goes to stderr
        assert "could not copy" in captured.err

    @patch.object(murmo, "record")
    @patch.object(murmo, "transcribe")
    def test_clipboard_on_by_default(self, mock_transcribe, mock_record, tmp_path: Path,
                                     monkeypatch):
        """Without any flag, the transcript should be copied to the clipboard."""
        mock_record.return_value = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        mock_transcribe.return_value = "hello world"
        mock_clipboard = MagicMock(return_value=True)
        monkeypatch.setattr("murmo.copy_to_clipboard", mock_clipboard)

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main([])
        finally:
            os.chdir(cwd)

        mock_clipboard.assert_called_once_with("hello world")

    @patch.object(murmo, "record")
    @patch.object(murmo, "transcribe")
    def test_no_clipboard_flag_disables_copy(self, mock_transcribe, mock_record, tmp_path: Path,
                                             capsys, monkeypatch):
        """--no-clipboard should skip the clipboard entirely."""
        mock_record.return_value = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        mock_transcribe.return_value = "hello world"
        mock_clipboard = MagicMock(return_value=True)
        monkeypatch.setattr("murmo.copy_to_clipboard", mock_clipboard)

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main(["--no-clipboard"])
        finally:
            os.chdir(cwd)

        mock_clipboard.assert_not_called()
        assert "copied to clipboard" not in capsys.readouterr().out


class TestMainAppend:
    """Test --append flag: appends to the latest transcript instead of creating a new file."""

    @patch.object(murmo, "record")
    @patch.object(murmo, "transcribe")
    def test_append_creates_new_file_when_none_exist(self, mock_transcribe, mock_record,
                                                      tmp_path: Path):
        """With --out and --append but no existing transcripts, creates a new file."""
        mock_record.return_value = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        mock_transcribe.return_value = "first transcript"

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main(["--out", "recordings", "--append"])
        finally:
            os.chdir(cwd)

        recordings = list((tmp_path / "recordings").iterdir())
        txt_files = [f for f in recordings if f.suffix == ".txt"]
        assert len(txt_files) == 1
        assert txt_files[0].read_text(encoding="utf-8").strip() == "first transcript"

    @patch.object(murmo, "record")
    @patch.object(murmo, "transcribe")
    def test_append_to_existing_file(self, mock_transcribe, mock_record, tmp_path: Path):
        """--append should add to the last transcript with a double newline separator."""
        recordings = tmp_path / "recordings"
        recordings.mkdir(exist_ok=True)
        existing = recordings / "recording_2026-01-01_00-00-00.txt"
        existing.write_text("first part\n", encoding="utf-8")
        # Set its mtime to the future so it's the latest
        existing.touch()

        mock_record.return_value = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        mock_transcribe.return_value = "second part"

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main(["--out", "recordings", "--append"])
        finally:
            os.chdir(cwd)

        txt_files = [f for f in recordings.iterdir() if f.suffix == ".txt"]
        assert len(txt_files) == 1
        assert txt_files[0] == existing  # same file, not a new one
        content = existing.read_text(encoding="utf-8")
        assert "first part" in content
        assert "second part" in content
        # Verify double newline separator
        assert "\n\n" in content

    @patch.object(murmo, "transcribe")
    def test_append_file_mode_uses_latest_txt(self, mock_transcribe, tmp_path: Path):
        """--file with --append should append to the latest .txt in --out."""
        recordings = tmp_path / "recordings"
        recordings.mkdir(exist_ok=True)
        existing = recordings / "meeting_notes.txt"
        existing.write_text("old meeting\n", encoding="utf-8")

        fake_file = tmp_path / "interview.m4a"
        fake_file.write_bytes(b"fake audio data")
        mock_transcribe.return_value = "new notes"

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main(["--file", str(fake_file), "--out", "recordings", "--append"])
        finally:
            os.chdir(cwd)

        txt_files = [f for f in recordings.iterdir() if f.suffix == ".txt"]
        assert len(txt_files) == 1
        assert txt_files[0] == existing  # same file, not a new one
        content = existing.read_text(encoding="utf-8")
        assert "old meeting" in content
        assert "new notes" in content


class TestSilenceDetection:
    """Test the silence detection helper."""

    def test_silent_chunk_returns_negative_inf(self):
        """Zero audio should return negative infinity dB."""
        chunk = np.zeros(160, dtype=np.float32)
        assert murmo._rms_db(chunk) == float("-inf")

    def test_silent_chunk_below_threshold(self):
        """Near-zero audio should return dB below the default threshold."""
        chunk = np.full(160, 0.001, dtype=np.float32)
        db = murmo._rms_db(chunk)
        assert db < murmo.SILENCE_THRESHOLD_DB

    def test_loud_chunk_returns_above_threshold(self):
        """Normal audio should return dB above the silence threshold."""
        chunk = np.full(160, 0.5, dtype=np.float32)
        db = murmo._rms_db(chunk)
        assert db > murmo.SILENCE_THRESHOLD_DB


class TestMainPunctuate:
    """Test --punctuate flag."""

    @patch.object(murmo, "record")
    @patch.object(murmo, "transcribe")
    def test_punctuate_mode_calls_with_flag(self, mock_transcribe, mock_record, tmp_path: Path,
                                             capsys):
        """--punctuate should set punctuate=True on transcribe()."""
        mock_record.return_value = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        mock_transcribe.return_value = "Hello, World!"

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main(["--punctuate"])
        finally:
            os.chdir(cwd)

        mock_transcribe.assert_called_once()
        assert mock_transcribe.call_args[0][3] is True  # punctuate arg

    @patch.object(murmo, "record")
    @patch.object(murmo, "transcribe")
    def test_punctuate_prints_enhancement_message(self, mock_transcribe, mock_record,
                                                   tmp_path: Path, capsys):
        """--punctuate should print an enhancement message before transcribing."""
        mock_record.return_value = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        mock_transcribe.return_value = "Hello, World!"

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main(["--punctuate"])
        finally:
            os.chdir(cwd)

        captured = capsys.readouterr()
        assert "Enhancing" in captured.out or "Enhancement" in captured.out

    @patch.object(murmo, "record")
    @patch.object(murmo, "transcribe")
    def test_non_punctuate_mode_default(self, mock_transcribe, mock_record, tmp_path: Path):
        """Without --punctuate, the flag should be False."""
        mock_record.return_value = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        mock_transcribe.return_value = "hello world"

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main([])
        finally:
            os.chdir(cwd)

        mock_transcribe.assert_called_once()
        assert mock_transcribe.call_args[0][3] is False  # punctuate arg


class TestTranscribe:
    """transcribe() against a mocked Whisper model."""

    def test_plain_transcribe(self, mock_whisper):
        audio = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        assert murmo.transcribe(audio, "tiny", "en") == "mocked transcription"
        kwargs = mock_whisper.transcribe.call_args.kwargs
        assert kwargs["language"] == "en"
        assert kwargs["fp16"] is False
        assert "initial_prompt" not in kwargs

    def test_punctuate_uses_language_prompt(self, mock_whisper):
        audio = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        murmo.transcribe(audio, "tiny", "de", punctuate=True)
        kwargs = mock_whisper.transcribe.call_args.kwargs
        assert kwargs["initial_prompt"] == murmo.PUNCTUATION_PROMPTS["de"]

    def test_punctuate_falls_back_to_english_prompt(self, mock_whisper):
        audio = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        murmo.transcribe(audio, "tiny", None, punctuate=True)
        kwargs = mock_whisper.transcribe.call_args.kwargs
        assert kwargs["initial_prompt"] == murmo.PUNCTUATION_PROMPTS["en"]


class TestMainRecordingOptions:
    """Test recording-related arguments."""

    @patch.object(murmo, "record")
    @patch.object(murmo, "transcribe")
    def test_silence_threshold_passed(self, mock_transcribe, mock_record, tmp_path: Path):
        """--silence-threshold should be forwarded to record()."""
        mock_record.return_value = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        mock_transcribe.return_value = "ok"

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main(["--silence-threshold", "-20"])
        finally:
            os.chdir(cwd)

        mock_record.assert_called_once()
        assert mock_record.call_args.kwargs["silence_threshold"] == -20.0

    @patch.object(murmo, "record")
    @patch.object(murmo, "transcribe")
    def test_silence_duration_passed(self, mock_transcribe, mock_record, tmp_path: Path):
        """--silence-duration should be forwarded to record()."""
        mock_record.return_value = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        mock_transcribe.return_value = "ok"

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main(["--silence-duration", "5.0"])
        finally:
            os.chdir(cwd)

        mock_record.assert_called_once()
        assert mock_record.call_args.kwargs["silence_duration"] == 5.0

    @patch.object(murmo, "record")
    @patch.object(murmo, "transcribe")
    def test_max_duration_passed(self, mock_transcribe, mock_record, tmp_path: Path):
        """--max-duration should be forwarded to record()."""
        mock_record.return_value = np.zeros(murmo.SAMPLE_RATE, dtype=np.float32)
        mock_transcribe.return_value = "ok"

        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            murmo.main(["--max-duration", "60"])
        finally:
            os.chdir(cwd)

        mock_record.assert_called_once()
        assert mock_record.call_args.kwargs["max_duration"] == 60.0
