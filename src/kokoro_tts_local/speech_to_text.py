"""
Local speech-to-text for dictating part B's English source text.

Uses OpenAI's Whisper through transformers (already installed). The model is
downloaded once into the Kokoro model directory and loaded from disk after
that, like the translation models.
"""

import os
import threading
from pathlib import Path

import numpy as np

from .paths import get_model_dir

STT_MODEL = os.environ.get("KOKORO_STT_MODEL", "openai/whisper-base.en")
WHISPER_RATE = 16000
SILENCE_RMS = 0.01  # below this a stretch of microphone audio counts as silence

_loaded = None
_lock = threading.Lock()


class SpeechToTextError(Exception):
    """A speech-to-text failure whose message is safe to show in the UI."""


def get_stt_model_dir(repo_id: str = STT_MODEL) -> Path:
    return get_model_dir() / "speech_models" / repo_id.split("/")[-1]


def _offline() -> bool:
    return (os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1")


def is_loaded() -> bool:
    return _loaded is not None


def is_downloaded() -> bool:
    model_dir = get_stt_model_dir()
    return (model_dir / "config.json").is_file() and any(
        (model_dir / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))


def _load():
    global _loaded
    with _lock:
        if _loaded is not None:
            return _loaded
        try:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as e:
            raise SpeechToTextError(f"Speech-to-text backend not installed ({e}).") from e
        model_dir = get_stt_model_dir()
        if not is_downloaded():
            if _offline():
                raise SpeechToTextError(
                    f"Speech-to-text model not downloaded and offline mode is on. "
                    f"Run once with network access, or place {STT_MODEL} in {model_dir}")
            try:
                from huggingface_hub import snapshot_download
                print(f"Downloading speech-to-text model {STT_MODEL} to {model_dir}...")
                snapshot_download(STT_MODEL, local_dir=str(model_dir))
            except Exception as e:
                raise SpeechToTextError(
                    f"Could not download speech-to-text model {STT_MODEL}: "
                    f"{type(e).__name__}: {e}") from e
        try:
            processor = WhisperProcessor.from_pretrained(str(model_dir))
            model = WhisperForConditionalGeneration.from_pretrained(str(model_dir)).eval()
        except Exception as e:
            raise SpeechToTextError(
                f"Could not load the speech-to-text model from {model_dir} "
                f"({type(e).__name__}: {e}). If a download was interrupted, delete that "
                "folder and try again.") from e
        _loaded = (processor, model, torch)
        return _loaded


def to_mono_16k(sample_rate: int, audio: np.ndarray) -> np.ndarray:
    """Convert a microphone chunk (any rate, int or float, mono or stereo)
    to mono float32 at 16 kHz."""
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    audio = audio.astype(np.float32)
    if sample_rate != WHISPER_RATE and len(audio):
        duration = len(audio) / sample_rate
        target = max(1, int(round(duration * WHISPER_RATE)))
        audio = np.interp(np.linspace(0, len(audio) - 1, target),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0


class Dictation:
    """Turns a stream of microphone chunks into text appended to a base text.

    A phrase is transcribed and committed when the speaker pauses (or after
    FORCE_COMMIT_SEC of continuous speech). Between commits, a provisional
    transcription of the phrase in progress is shown so text appears while
    speaking; it is only recomputed while the phrase is short, which bounds
    the cost of each update.
    """
    COMMIT_SILENCE_SEC = 0.6
    MIN_PHRASE_SEC = 1.0
    FORCE_COMMIT_SEC = 12.0
    LIVE_PREVIEW_MAX_SEC = 6.0

    def __init__(self, base_text: str = ""):
        self.base = (base_text or "").rstrip()
        self.committed = []
        self.pending = np.zeros(0, dtype=np.float32)
        self.preview = ""

    def _commit(self):
        heard = transcribe(self.pending)
        if heard:
            self.committed.append(heard)
        self.pending = np.zeros(0, dtype=np.float32)
        self.preview = ""

    def feed(self, sample_rate: int, chunk) -> str:
        self.pending = np.concatenate([self.pending, to_mono_16k(sample_rate, chunk)])
        seconds = len(self.pending) / WHISPER_RATE
        tail = self.pending[-int(self.COMMIT_SILENCE_SEC * WHISPER_RATE):]
        paused = seconds >= self.COMMIT_SILENCE_SEC and rms(tail) < SILENCE_RMS
        if paused and rms(self.pending) < SILENCE_RMS:
            self.pending = np.zeros(0, dtype=np.float32)  # only silence so far
        elif (paused and seconds >= self.MIN_PHRASE_SEC) or seconds >= self.FORCE_COMMIT_SEC:
            self._commit()
        elif seconds <= self.LIVE_PREVIEW_MAX_SEC and rms(self.pending) >= SILENCE_RMS:
            self.preview = transcribe(self.pending)
        return self.text()

    def finish(self) -> str:
        if len(self.pending):
            self._commit()
        return self.text()

    def dictated_words(self) -> int:
        return sum(len(part.split()) for part in self.committed)

    def text(self) -> str:
        parts = [self.base, *self.committed, self.preview]
        return " ".join(p for p in parts if p)


def transcribe(audio_16k: np.ndarray) -> str:
    """Transcribe mono 16 kHz audio (30 s windows, so any length works)."""
    if len(audio_16k) == 0 or rms(audio_16k) < SILENCE_RMS:
        return ""  # Whisper invents words ("Thank you.") on pure silence
    processor, model, torch = _load()
    window = 30 * WHISPER_RATE
    pieces = []
    for start in range(0, len(audio_16k), window):
        chunk = audio_16k[start:start + window]
        if rms(chunk) < SILENCE_RMS:
            continue
        features = processor(chunk, sampling_rate=WHISPER_RATE, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(features, num_beams=1, do_sample=False)
        pieces.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return " ".join(p for p in pieces if p)
