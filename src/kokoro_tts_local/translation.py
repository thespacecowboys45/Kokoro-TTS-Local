"""
Local text translation for the Gradio interface.

Translation is an explicit, separate step: the UI translates, the user reviews
or edits the result, and only then is it spoken. Models are downloaded once
into the Kokoro model directory and loaded from disk afterwards, so restarts
and reboots never re-download them.
"""

import os
import re
import threading
from pathlib import Path
from typing import Dict, Tuple

from .paths import get_model_dir

# (source, destination) display names -> Hugging Face model repo
SUPPORTED_PAIRS: Dict[Tuple[str, str], str] = {
    ("English", "Japanese"): "staka/fugumt-en-ja",
}

# Destination display name -> Kokoro language code (models.LANGUAGE_CODES)
DEST_LANGUAGE_CODES: Dict[str, str] = {
    "Japanese": "j",
}

SOURCE_LANGUAGES = sorted({src for src, _ in SUPPORTED_PAIRS})
DEST_LANGUAGES = sorted({dst for _, dst in SUPPORTED_PAIRS})

MAX_TRANSLATE_CHARS = None  # None = unlimited (dxb, 2026-10-12); upstream-style cap was 5000
MAX_NEW_TOKENS = 512
# The Marian models take at most 512 tokens per input and truncate the rest,
# so longer sentences are split at clause boundaries before translating.
MAX_PIECE_CHARS = 300
BATCH_SIZE = 16

# A file that only exists once a model snapshot is fully usable.
_WEIGHT_FILES = ("pytorch_model.bin", "model.safetensors")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_models = {}
_model_lock = threading.Lock()


class TranslationError(Exception):
    """A translation failure whose message is safe to show in the UI."""


def get_translation_model_dir(repo_id: str) -> Path:
    """Persistent on-disk location of a translation model."""
    return get_model_dir() / "translation_models" / repo_id.split("/")[-1]


def _offline() -> bool:
    return (os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1")


def _has_model_files(model_dir: Path) -> bool:
    return (model_dir / "config.json").is_file() and any(
        (model_dir / name).is_file() for name in _WEIGHT_FILES
    )


def is_model_loaded(source: str, dest: str) -> bool:
    """True when the pair's model is already in memory (translation is fast)."""
    return SUPPORTED_PAIRS.get((source, dest)) in _models


def is_model_downloaded(source: str, dest: str) -> bool:
    repo_id = SUPPORTED_PAIRS.get((source, dest))
    return bool(repo_id) and _has_model_files(get_translation_model_dir(repo_id))


def _load(repo_id: str):
    """Return (tokenizer, model, device), downloading once if needed."""
    with _model_lock:
        if repo_id in _models:
            return _models[repo_id]

        try:
            import torch
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError as e:
            raise TranslationError(
                f"Translation backend not installed ({e}). "
                "Install it with: pip install -r requirements.txt"
            ) from e

        model_dir = get_translation_model_dir(repo_id)
        if not _has_model_files(model_dir):
            if _offline():
                raise TranslationError(
                    f"Translation model not downloaded and offline mode is on. "
                    f"Run once with network access, or place {repo_id} in {model_dir}"
                )
            try:
                from huggingface_hub import snapshot_download
                print(f"Downloading translation model {repo_id} to {model_dir}...")
                snapshot_download(repo_id, local_dir=str(model_dir))
            except Exception as e:
                raise TranslationError(
                    f"Could not download translation model {repo_id}: "
                    f"{type(e).__name__}: {e}"
                ) from e

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            tokenizer = MarianTokenizer.from_pretrained(str(model_dir))
            model = MarianMTModel.from_pretrained(str(model_dir)).to(device).eval()
        except Exception as e:
            raise TranslationError(
                f"Could not load translation model from {model_dir} "
                f"({type(e).__name__}: {e}). If a download was interrupted, "
                "delete that folder and translate again to re-download it."
            ) from e

        _models[repo_id] = (tokenizer, model, device)
        return _models[repo_id]


def _translate_sentences(sentences, tokenizer, model, device):
    import torch

    results = []
    for start in range(0, len(sentences), BATCH_SIZE):
        batch = tokenizer(sentences[start:start + BATCH_SIZE], return_tensors="pt",
                          padding=True, truncation=True).to(device)
        with torch.no_grad():
            # Greedy decoding on purpose: under transformers 5.x this model's
            # bundled num_beams=12 degenerates into repeated tokens, while
            # greedy output is correct.
            output = model.generate(**batch, num_beams=1, do_sample=False,
                                    max_new_tokens=MAX_NEW_TOKENS)
        results.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
    return results


def translate(text: str, source: str, dest: str) -> str:
    """Translate text, preserving line breaks so speech segmentation matches."""
    if not text or not text.strip():
        raise TranslationError("Enter text to translate.")
    if MAX_TRANSLATE_CHARS is not None and len(text) > MAX_TRANSLATE_CHARS:
        raise TranslationError(
            f"Text is {len(text)} characters; the translation limit is "
            f"{MAX_TRANSLATE_CHARS}."
        )
    repo_id = SUPPORTED_PAIRS.get((source, dest))
    if repo_id is None:
        raise TranslationError(f"Translation from {source} to {dest} is not supported.")

    tokenizer, model, device = _load(repo_id)

    # Split each line into sentences so no single input exceeds the model's
    # context; blank lines are kept as-is.
    lines = text.split("\n")
    sentences = []
    layout = []
    for line in lines:
        parts = _model_pieces([p for p in _SENTENCE_SPLIT.split(line.strip()) if p])
        layout.append(len(parts))
        sentences.extend(parts)

    try:
        translated = _translate_sentences(sentences, tokenizer, model, device) if sentences else []
    except Exception as e:
        raise TranslationError(f"Translation failed: {type(e).__name__}: {e}") from e

    if dest == "Japanese":
        translated = [_end_japanese_sentence(ja, en) for ja, en in zip(translated, sentences)]

    out_lines = []
    index = 0
    for count in layout:
        out_lines.append("".join(translated[index:index + count]))
        index += count
    return "\n".join(out_lines)


_JA_SENTENCE_END = re.compile(r"[。！？!?…」』）)]$")
_JA_TERMINATORS = {"?": "？", "!": "！", ".": "。"}


def _end_japanese_sentence(japanese: str, source_sentence: str) -> str:
    """Restore a sentence end the model dropped, so joined sentences stay
    visibly separate (e.g. a deliberately repeated sentence)."""
    japanese = japanese.rstrip()
    terminator = _JA_TERMINATORS.get(source_sentence.rstrip()[-1:])
    if terminator and japanese and not _JA_SENTENCE_END.search(japanese):
        return japanese + terminator
    return japanese


# Back-translation (translation review). Kept out of SUPPORTED_PAIRS so the
# Source/Destination dropdowns, which are derived from it, do not change.
BACK_TRANSLATION_PAIRS: Dict[Tuple[str, str], str] = {
    ("Japanese", "English"): "staka/fugumt-ja-en",
}

# Japanese ends sentences with 。！？ and no following space.
_JA_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])\s*")


def _split_sentences(line: str, lang: str):
    pattern = _JA_SENTENCE_SPLIT if lang == "Japanese" else _SENTENCE_SPLIT
    return [p for p in pattern.split(line.strip()) if p]


_CLAUSE_SPLIT = re.compile(r"(?<=[,;:、，；：])\s*")


def _model_pieces(sentences):
    """Split sentences longer than the model can take at clause boundaries
    (then at a fixed length), so the translation model never truncates."""
    pieces = []
    for sentence in sentences:
        if len(sentence) <= MAX_PIECE_CHARS:
            pieces.append(sentence)
            continue
        current = ""
        for clause in (c for c in _CLAUSE_SPLIT.split(sentence) if c):
            if current and len(current) + len(clause) > MAX_PIECE_CHARS:
                pieces.append(current.strip())
                current = ""
            # The split consumed the space after "," but not after "、".
            separator = " " if current and current[-1:].isascii() else ""
            current += separator + clause
            while len(current) > MAX_PIECE_CHARS:  # no clause breaks at all
                pieces.append(current[:MAX_PIECE_CHARS])
                current = current[MAX_PIECE_CHARS:]
        if current.strip():
            pieces.append(current.strip())
    return pieces


def is_back_model_loaded(source: str = "Japanese", dest: str = "English") -> bool:
    return BACK_TRANSLATION_PAIRS.get((source, dest)) in _models


def is_back_model_downloaded(source: str = "Japanese", dest: str = "English") -> bool:
    repo_id = BACK_TRANSLATION_PAIRS.get((source, dest))
    return bool(repo_id) and _has_model_files(get_translation_model_dir(repo_id))


def back_translate(text: str, source: str = "Japanese", dest: str = "English") -> str:
    """Translate the target text back into the source language, line by line.

    Only the text itself is an input: the original source is never passed, so
    the result cannot be biased toward reproducing it.
    """
    if not text or not text.strip():
        raise TranslationError("There is no text to back-translate.")
    if MAX_TRANSLATE_CHARS is not None and len(text) > MAX_TRANSLATE_CHARS:
        raise TranslationError(
            f"Text is {len(text)} characters; the back-translation limit is "
            f"{MAX_TRANSLATE_CHARS}."
        )
    repo_id = BACK_TRANSLATION_PAIRS.get((source, dest))
    if repo_id is None:
        raise TranslationError(f"Back-translation from {source} to {dest} is not supported.")

    tokenizer, model, device = _load(repo_id)

    layout = []
    sentences = []
    for line in text.split("\n"):
        parts = _model_pieces(_split_sentences(line, source))
        layout.append(len(parts))
        sentences.extend(parts)

    try:
        translated = _translate_sentences(sentences, tokenizer, model, device) if sentences else []
    except Exception as e:
        raise TranslationError(f"Back-translation failed: {type(e).__name__}: {e}") from e

    out_lines = []
    index = 0
    for count in layout:
        out_lines.append(" ".join(translated[index:index + count]))
        index += count
    return "\n".join(out_lines)
