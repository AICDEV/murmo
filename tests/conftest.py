"""Shared fixtures for murmo tests."""

from unittest.mock import MagicMock, patch
import pytest

import murmo


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """Keep the cached Whisper model isolated across tests."""
    murmo._load_whisper_model.cache_clear()
    yield
    murmo._load_whisper_model.cache_clear()


@pytest.fixture
def mock_whisper():
    """Mock the Whisper model — returns a fixed transcript."""
    model = MagicMock()
    model.transcribe.return_value = {"text": "mocked transcription"}
    with patch("whisper.load_model", return_value=model):
        yield model


@pytest.fixture(autouse=True)
def _no_real_clipboard(request, monkeypatch):
    """Clipboard copy is on by default — never touch the real clipboard in tests.

    TestCopyToClipboard exercises the real helper (with subprocess mocked), so skip it there.
    """
    if request.node.cls is not None and request.node.cls.__name__ == "TestCopyToClipboard":
        return
    monkeypatch.setattr(murmo, "copy_to_clipboard", MagicMock(return_value=True))
