"""
Kokoro-TTS Local Generator
-------------------------
A Gradio interface for the Kokoro-TTS-Local text-to-speech system.
Supports multiple voices and audio formats, with cross-platform compatibility.

Key Features:
- Multiple voice models support (54 voices across 8 languages)
- Real-time generation with progress logging
- WAV, MP3, and AAC output formats
- Network sharing capabilities
- Cross-platform compatibility (Windows, macOS, Linux)

Dependencies:
- kokoro: Official Kokoro TTS library
- gradio: Web interface framework
- soundfile: Audio file handling
- pydub: Audio format conversion
"""

import gradio as gr
import os
import sys
import platform
from datetime import datetime
import uuid
import ipaddress
import math
import queue
import re
import time
import shutil
import tempfile
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment
import torch
import numpy as np
import argparse
import inspect
import threading
from typing import Union, List, Optional, Tuple, Dict, Any
from contextlib import closing
from .models import (
    list_available_voices, build_model,
    download_voice_files, EnhancedKPipeline,
    get_safe_voice_path, get_language_code_from_voice, shutdown_pipelines,
    get_base_dir
)
from . import speed_dial
from . import translation
from . import translation_review
from . import video_export
from . import fork_version
from . import speech_to_text

# Constants
# Size/resource limits: None means unlimited. This instance is deliberately
# run without them (dxb, 2026-10-12: "think unlimited"; long jobs may take
# hours and the machine may bog down). Upstream's values are noted so any
# limit can be restored by setting a number again.
MAX_TEXT_LENGTH = None        # upstream: 5000 characters
DEFAULT_SAMPLE_RATE = 24000
MIN_SPEED = 0.1
MAX_SPEED = 3.0
DEFAULT_SPEED = 1.0
# Cost-envelope budget: synthesis cost scales with len(text) and inversely
# with speed, so a bound on their product keeps one request from pinning the
# single TTS worker for CPU-hours.
MAX_COST_CHARS = None         # upstream: 5_000
MAX_SEGMENTS = None           # upstream: 100 segments per request
MAX_AUDIO_SECONDS = None      # upstream: 600 s of output audio
LOW_MEMORY_TEXT_CAP = None    # upstream: 2000 characters when < 2 GB RAM free

# Define path type for consistent handling
PathLike = Union[str, Path]

# Configuration validation
def validate_sample_rate(rate: int) -> int:
    """Validate sample rate is within acceptable range"""
    valid_rates = [16000, 22050, 24000, 44100, 48000]
    if rate not in valid_rates:
        print(f"Warning: Unusual sample rate {rate}. Valid rates are {valid_rates}")
        return 24000  # Default to safe value
    return rate

# Global configuration
CONFIG_FILE = get_base_dir() / "tts_config.json"  # Stores user preferences and paths
DEFAULT_OUTPUT_DIR = get_base_dir() / "outputs"    # Directory for generated audio files
SAMPLE_RATE = validate_sample_rate(24000)  # Validated sample rate

# Retention budget for the outputs dir: nothing else in the package ever
# reclaims generated files, so with a budget set the oldest are deleted.
# None keeps every generated file (unlimited; upstream: 2 GiB).
MAX_OUTPUTS_DIR_BYTES = None


def enforce_output_retention(budget_bytes: Optional[int] = MAX_OUTPUTS_DIR_BYTES) -> None:
    """Delete oldest generated files until the outputs dir fits its budget."""
    if budget_bytes is None:
        return
    files = sorted(
        (p for p in DEFAULT_OUTPUT_DIR.glob("tts_*.*") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    total = sum(p.stat().st_size for p in files)
    for path in files:
        if total <= budget_bytes:
            break
        try:
            size = path.stat().st_size
            path.unlink()
            total -= size
        except OSError:
            pass

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_available_voices():
    """Get list of available voice models."""
    try:
        # Initialize model to trigger voice downloads
        build_model(None, device)

        voices = list_available_voices()
        if not voices:
            print("No voices found after initialization. Attempting to download...")
            download_voice_files()  # Try downloading again
            voices = list_available_voices()

        print("Available voices:", voices)
        return voices
    except Exception as e:
        print(f"Error getting voices: {e}")
        return []

def get_pipeline_for_voice(voice_name: str) -> EnhancedKPipeline:
    """
    Determine the language code from the voice prefix and return the associated pipeline.
    """
    lang_code = get_language_code_from_voice(voice_name)
    return build_model(None, device, lang_code=lang_code)

def convert_audio(input_path: PathLike, output_path: PathLike, format: str) -> Optional[PathLike]:
    """Convert audio to specified format.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        format: Output format ('wav', 'mp3', or 'aac')

    Returns:
        Path to output file or None on error
    """
    try:
        # Normalize paths
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()

        # Validate input file
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # For WAV format, just return the input path
        if format.lower() == "wav":
            return input_path

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert format
        audio = AudioSegment.from_wav(str(input_path))

        # Select proper format and options
        if format.lower() == "mp3":
            audio.export(str(output_path), format="mp3", bitrate="192k")
        elif format.lower() == "aac":
            audio.export(str(output_path), format="adts", codec="aac", bitrate="192k")
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Verify file was created
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise IOError(f"Failed to create {format} file")

        return output_path

    except (IOError, FileNotFoundError, ValueError) as e:
        print(f"Error converting audio: {type(e).__name__}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error converting audio: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_tts_with_logs(
    voice_name: str, text: str, format: str, speed: float = 1.0
) -> Tuple[Optional[PathLike], str]:
    """Generate TTS audio with progress logging and memory management.

    Args:
        voice_name: Name of the voice to use
        text: Text to convert to speech
        format: Output format ('wav', 'mp3', 'aac')
        speed: Speech speed multiplier

    Returns:
        ``(path, status)``. The status string is rendered in the UI: a remote
        user has no access to this process's console, so a bare ``None`` left
        them unable to tell a missing ffmpeg from a still-running request.
    """
    import psutil
    import gc

    notes = []
    try:
        # Check available memory before processing
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)

        if available_gb < 1.0:  # Less than 1GB available
            # Console only: with no limits, low memory just means slower.
            print(f"Warning: Low memory available ({available_gb:.1f}GB); "
                  "continuing without limits, this may be slow.")
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Create output directory
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        enforce_output_retention()

        # Validate input text
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        # Validate speed server-side. The slider's bounds are advisory: a
        # crafted request can send any float, and speed=1e-9 asks the vocoder
        # to stretch the utterance by a factor of a billion. NaN is rejected
        # explicitly because it fails both comparisons.
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            raise ValueError(f"Speed must be a number, got {speed!r}")
        if not (MIN_SPEED <= speed <= MAX_SPEED):
            raise ValueError(
                f"Speed must be between {MIN_SPEED} and {MAX_SPEED}, got {speed}"
            )

        # Dynamic text length limit based on available memory
        MAX_CHARS = MAX_TEXT_LENGTH
        if available_gb < 2.0 and LOW_MEMORY_TEXT_CAP is not None:
            MAX_CHARS = min(MAX_CHARS or LOW_MEMORY_TEXT_CAP, LOW_MEMORY_TEXT_CAP)
            message = f"Reduced text limit to {MAX_CHARS} characters due to low memory"
            print(message)
            notes.append(message)

        if MAX_CHARS is not None and len(text) > MAX_CHARS:
            message = (f"Text exceeded {MAX_CHARS} characters and was truncated; "
                       "only the beginning was synthesized.")
            print(f"Warning: {message}")
            notes.append(message)
            text = text[:MAX_CHARS] + "..."

        # Synthesis cost is driven by len(text)/speed; with a budget set, it
        # stops a single request (5000 chars at speed 0.1) pinning the only
        # TTS worker for CPU-hours.
        if MAX_COST_CHARS is not None and len(text) / speed > MAX_COST_CHARS:
            raise ValueError(
                f"Request too heavy: len(text)/speed must be "
                f"<= {MAX_COST_CHARS}; shorten the text or raise the speed"
            )

        # Generate base filename from text
        output_format = format.lower()
        if output_format not in {'wav', 'mp3', 'aac'}:
            raise ValueError(f"Unsupported format: {format}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"tts_{timestamp}_{uuid.uuid4().hex}"
        wav_path = DEFAULT_OUTPUT_DIR / f"{base_name}.wav"

        # Generate speech
        print(f"\nGenerating speech for: '{text}'")
        print(f"Using voice: {voice_name}")

        # Validate voice path using safe helper to prevent path traversal
        voice_path = get_safe_voice_path(voice_name)
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")

        try:
            pipeline = get_pipeline_for_voice(voice_name)

            all_audio = []
            max_segments = MAX_SEGMENTS  # None = unlimited
            # With a cap set, retained audio is bounded directly: per-parameter
            # limits (chars, speed) multiply with G2P digit expansion (~8x prose
            # density). 600 s at 24 kHz float32 is ~58 MB aggregate; unlimited
            # grows by about that much per 10 minutes of speech.
            max_samples = None if MAX_AUDIO_SECONDS is None else MAX_AUDIO_SECONDS * SAMPLE_RATE
            total_samples = 0
            segment_count = 0

            # The generator owns the model-family lock until it is closed, so
            # it must not outlive this loop: hitting max_segments would
            # otherwise hold the lock through file writing and conversion, and
            # block shutdown, which drains by acquiring every family lock.
            # Non-English text is pre-split into sentence-sized chunks; see
            # _speech_chunks for why Kokoro's own splitting is not enough.
            chunks = _speech_chunks(text, voice_name)
            truncated_segments = 0
            with closing(pipeline.iter_speech(
                chunks if chunks is not None else text,
                voice=str(voice_path), speed=speed, split_pattern=r'\n+'
            )) as generator:
                for gs, ps, audio in generator:
                    segment_count += 1
                    if ps and len(ps) >= KOKORO_MAX_PHONEMES:
                        truncated_segments += 1
                    if max_segments is not None and segment_count > max_segments:
                        message = (f"Reached the {max_segments}-segment limit; "
                                   "the audio is truncated.")
                        print(f"Warning: {message}")
                        notes.append(message)
                        break

                    if audio is not None:
                        if isinstance(audio, np.ndarray):
                            audio = torch.from_numpy(audio).float()
                        total_samples += audio.numel()
                        all_audio.append(audio)
                        if max_samples is not None and total_samples > max_samples:
                            message = ("Reached the total audio-length limit; "
                                       "the audio is truncated.")
                            print(f"Warning: {message}")
                            notes.append(message)
                            break
                        print(f"Generated segment: {gs}")
                        if ps:  # Only print phonemes if available
                            print(f"Phonemes: {ps}")

            if truncated_segments:
                message = (f"{truncated_segments} segment(s) reached Kokoro's "
                           f"{KOKORO_MAX_PHONEMES}-phoneme limit and were cut short; "
                           "some text was not spoken. Split long sentences.")
                print(f"Warning: {message}")
                notes.append(message)

            if not all_audio:
                raise Exception("No audio generated")
        except Exception as e:
            raise RuntimeError(f"Speech generation failed: {e}") from e

        # Combine audio segments and save
        if not all_audio:
            raise Exception("No audio segments were generated")

        # Handle single segment case without concatenation
        if len(all_audio) == 1:
            final_audio = all_audio[0]
        else:
            try:
                final_audio = torch.cat(all_audio, dim=0)
            except RuntimeError as e:
                raise Exception(f"Failed to concatenate audio segments: {e}")

        # Save audio file
        try:
            if isinstance(final_audio, torch.Tensor):
                final_audio = final_audio.detach().cpu().numpy()
            sf.write(wav_path, final_audio, SAMPLE_RATE)
        except Exception as e:
            raise Exception(f"Failed to save audio file: {e}")

        # Convert to requested format if needed
        if output_format != "wav":
            output_path = DEFAULT_OUTPUT_DIR / f"{base_name}.{output_format}"
            converted = convert_audio(wav_path, output_path, output_format)
            if converted is None:
                raise RuntimeError(
                    f"Conversion to {output_format} failed. This usually means "
                    "FFmpeg is not installed or not on PATH."
                )
            wav_path.unlink(missing_ok=True)
            return converted, _status(f"Generated {segment_count} segment(s) as "
                                      f"{output_format}.", notes)

        return wav_path, _status(f"Generated {segment_count} segment(s) as wav.", notes)

    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback
        traceback.print_exc()
        return None, _status(f"{type(e).__name__}: {e}", notes)


def _status(headline: str, notes: List[str]) -> str:
    """Join a headline with any warnings collected during the request."""
    return "\n".join([headline] + [f"- {note}" for note in notes])


# Kokoro's model accepts at most this many phonemes per segment and silently
# cuts anything longer.
KOKORO_MAX_PHONEMES = 510
# Characters per non-English chunk. Japanese runs ~3 phonemes per character,
# so 150 characters stays well under the 510-phoneme limit.
MAX_CHUNK_CHARS = 150
_SENTENCE_END = re.compile(r"(?<=[。！？!?.…])\s*")
_CLAUSE_END = re.compile(r"(?<=[、，,;；:：])\s*")


def _speech_chunks(text: str, voice_name: str) -> Optional[List[str]]:
    """Pre-split non-English text into chunks Kokoro can speak in full.

    Kokoro chunks non-English text only at ASCII . ! ? and in 400-character
    blocks, so a Japanese paragraph (。！？) becomes one chunk that is cut at
    510 phonemes (~29 s). English uses Kokoro's own chunker: returns None.
    """
    if get_language_code_from_voice(voice_name) in ("a", "b"):
        return None
    chunks = []
    for line in text.split("\n"):
        for sentence in (s for s in _SENTENCE_END.split(line.strip()) if s.strip()):
            if len(sentence) <= MAX_CHUNK_CHARS:
                chunks.append(sentence)
                continue
            current = ""
            for clause in (c for c in _CLAUSE_END.split(sentence) if c):
                if current and len(current) + len(clause) > MAX_CHUNK_CHARS:
                    chunks.append(current)
                    current = ""
                current += clause
                while len(current) > MAX_CHUNK_CHARS:  # no clause breaks at all
                    chunks.append(current[:MAX_CHUNK_CHARS])
                    current = current[MAX_CHUNK_CHARS:]
            if current:
                chunks.append(current)
    # A punctuation-only fragment has nothing to speak: attach it to the previous chunk.
    merged = []
    for chunk in chunks:
        if not re.search(r"\w", chunk) and merged:
            merged[-1] += chunk
        else:
            merged.append(chunk)
    return merged


def synthesize_sentences(voice_name: str, text: str, speed: float):
    """Synthesise text one sentence at a time for video subtitle timing.

    Applies the same limits as generate_tts_with_logs. Returns
    (clips, spans, sentences): each clip trimmed to its speech, and each span
    (onset, offset, clip_length) in seconds within its own clip.
    """
    if not text or not text.strip():
        raise ValueError("Text input cannot be empty")
    try:
        speed = float(speed)
    except (TypeError, ValueError):
        raise ValueError(f"Speed must be a number, got {speed!r}")
    if not (MIN_SPEED <= speed <= MAX_SPEED):
        raise ValueError(f"Speed must be between {MIN_SPEED} and {MAX_SPEED}, got {speed}")
    if MAX_TEXT_LENGTH is not None and len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Text is {len(text)} characters; the limit is {MAX_TEXT_LENGTH}.")
    if MAX_COST_CHARS is not None and len(text) / speed > MAX_COST_CHARS:
        raise ValueError(f"Request too heavy: len(text)/speed must be <= {MAX_COST_CHARS}; "
                         "shorten the text or raise the speed")

    voice_path = get_safe_voice_path(voice_name)
    if not voice_path.exists():
        raise FileNotFoundError(f"Voice file not found: {voice_path}")
    lang = "Japanese" if get_language_code_from_voice(voice_name) == "j" else "English"
    sentences = []
    for line in text.split("\n"):
        for piece in translation._split_sentences(line, lang):
            # A punctuation-only fragment ("。" from "。。", "！？") has nothing
            # to speak: attach it to the previous sentence instead.
            if not re.search(r"\w", piece):
                if sentences:
                    sentences[-1] += piece
                continue
            sentences.append(piece)
    if not sentences:
        raise ValueError("The text has nothing speakable in it.")

    pipeline = get_pipeline_for_voice(voice_name)
    max_samples = None if MAX_AUDIO_SECONDS is None else MAX_AUDIO_SECONDS * SAMPLE_RATE
    total_samples = 0
    clips, spans = [], []
    for sentence in sentences:
        chunks = []
        pieces = _speech_chunks(sentence, voice_name)
        # Closed before the next sentence so the model-family lock is released.
        with closing(pipeline.iter_speech(
            pieces if pieces is not None else sentence,
            voice=str(voice_path), speed=speed, split_pattern=r'\n+'
        )) as generator:
            for _, ps, audio in generator:
                if ps and len(ps) >= KOKORO_MAX_PHONEMES:
                    # A cut-short sentence would desynchronise the subtitles.
                    raise ValueError(
                        f"A sentence reached Kokoro's {KOKORO_MAX_PHONEMES}-phoneme limit and "
                        f"would be cut short: {sentence[:40]!r}... Split it into shorter sentences.")
                if audio is None:
                    continue
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()
                chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            raise RuntimeError(f"No audio generated for: {sentence!r}")
        clip, onset, offset = video_export.trim_silence(np.concatenate(chunks), SAMPLE_RATE)
        total_samples += len(clip)
        if max_samples is not None and total_samples > max_samples:
            raise ValueError(f"The speech would exceed the {MAX_AUDIO_SECONDS} s audio limit; "
                             "shorten the text.")
        clips.append(clip)
        spans.append((onset, offset, len(clip) / SAMPLE_RATE))
    return clips, spans, sentences

def create_interface(server_name="127.0.0.1", server_port=7860, auth=None):
    """Create and launch the Gradio interface."""

    # Get available voices
    voices = get_available_voices()
    if not voices:
        print("No voices found! Please check the voices directory.")
        return

    # Get speed dial presets
    preset_names = speed_dial.get_preset_names()

    # Create interface
    with gr.Blocks(title="Kokoro TTS Generator", fill_height=True) as interface:
        with gr.Row(equal_height=True):
            gr.Markdown("# Kokoro TTS Generator", scale=3)
            # This fork's identity, in the same rounded panel style as the parts.
            with gr.Column(variant="panel", scale=1, min_width=260):
                gr.Markdown(f"**FORK:** {fork_version.FORK_NAME}  \n"
                            f"Version **{fork_version.FORK_VERSION}**")

        # Part A: direct text to speech. Built-in panel containers only (no
        # tabs, no custom CSS) so the default theme's look is preserved.
        with gr.Column(variant="panel"):
            gr.Markdown("## A. Text to Speech\n"
                        "Type English text, pick a voice, and generate speech directly.")

            with gr.Row():
                with gr.Column(scale=2):
                    # Main TTS controls
                    voice = gr.Dropdown(
                        choices=voices,
                        value=voices[0] if voices else None,
                        label="Voice"
                    )
                    text = gr.Textbox(
                        lines=3,
                        placeholder="Enter text to convert to speech...",
                        label="Text"
                    )
                    with gr.Row():
                        format = gr.Radio(
                            choices=["wav", "mp3", "aac"],
                            value="wav",
                            label="Output Format"
                        )
                        speed = gr.Slider(
                            minimum=0.1,
                            maximum=3.0,
                            value=1.0,
                            step=0.1,
                            label="Speed"
                        )
                    generate = gr.Button("Generate Speech", variant="primary")

                with gr.Column(scale=1):
                    # Speed dial section
                    gr.Markdown("### Speed Dial\n"
                                "Save and reload voice, text, format and speed.")
                    preset_dropdown = gr.Dropdown(
                        choices=preset_names,
                        value=preset_names[0] if preset_names else None,
                        label="Saved Presets",
                        interactive=True
                    )
                    preset_name = gr.Textbox(
                        placeholder="Enter preset name...",
                        label="New Preset Name"
                    )
                    with gr.Row():
                        load_preset = gr.Button("Load")
                        save_preset = gr.Button("Save Current")
                        delete_preset = gr.Button("Delete")

            # Output section
            output = gr.Audio(label="Generated Audio")

            # Every handler reports here. Without it the only failure signal
            # was an empty audio player, identical for empty input, a missing
            # voice file, absent ffmpeg and CUDA OOM.
            status = gr.Textbox(
                label="Status",
                value="Ready.",
                lines=3,
                interactive=False
            )

        # Part B: translation. It has its own player and status box so the
        # direct TTS workflow above is never written to by these handlers.
        default_dest = translation.DEST_LANGUAGES[0]
        translation_voices = [
            v for v in voices
            if get_language_code_from_voice(v) == translation.DEST_LANGUAGE_CODES[default_dest]
        ]

        with gr.Column(variant="panel"):
            gr.Markdown("## B. Translation\n"
                        "Translate English into Japanese, review and edit the result, "
                        "then speak it with a Japanese voice.")

            gr.Markdown("### 1. Source")
            with gr.Row():
                source_lang = gr.Dropdown(
                    choices=translation.SOURCE_LANGUAGES,
                    value=translation.SOURCE_LANGUAGES[0],
                    label="Source Language",
                    interactive=True
                )
                dest_lang = gr.Dropdown(
                    choices=translation.DEST_LANGUAGES,
                    value=default_dest,
                    label="Destination Language",
                    interactive=True
                )
            source_text = gr.Textbox(
                lines=3,
                placeholder="Enter text to translate...",
                label="Source Text"
            )
            # Dictation: speech is transcribed locally (Whisper) and appended to
            # the Source Text while you speak. Part B only; part A is untouched.
            source_mic = gr.Audio(
                sources=["microphone"],
                streaming=True,
                type="numpy",
                label="🎤 Dictate into Source Text (English speech-to-text) — press Record and speak"
            )
            dictation_state = gr.State(None)
            with gr.Row():
                review_context = gr.Textbox(
                    lines=2,
                    label="Context (optional, used by Translate and Review)",
                    placeholder="Audience, intended meaning, tone, pronunciation, terminology"
                )
                review_glossary = gr.Textbox(
                    lines=2,
                    label="Glossary (optional, used by Translate and Review)",
                    placeholder="One per line: English term = 日本語訳  (e.g. lawn = 芝生)"
                )

            gr.Markdown("### 2. Translate\n"
                        "**Translate** is quick and plain. **Translate and Review** follows "
                        "the context and glossary, back-translates, and reviews the result "
                        "(needs Ollama running).")
            with gr.Row():
                translate_btn = gr.Button("Translate")
                translate_review_btn = gr.Button("Translate and Review", variant="primary")

            gr.Markdown("### 3. Review and edit the Japanese")
            translated_text = gr.Textbox(
                lines=3,
                placeholder="Translated text appears here. Review or edit it before generating speech.",
                label="Translated Text",
                interactive=True
            )
            recheck_btn = gr.Button("Recheck Edited Translation")
            # Live, ticking progress for Translate and Review / Recheck.
            review_progress_md = gr.Markdown("")
            gr.Markdown("*Back-translation is a diagnostic, not a certification.*")
            # Translation review: English → Japanese → independent English
            # back-translation, review findings, and at most two automatic
            # revision passes shown beside (never over) the editable Japanese.
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Review of the Japanese above (unrevised)")
                    back_translation_box = gr.Textbox(
                        label="English Back-translation", lines=3, interactive=False
                    )
                    review_findings_md = gr.Markdown()
                with gr.Column():
                    gr.Markdown("#### Revised candidate")
                    candidate_text = gr.Textbox(
                        label="Revised Japanese (candidate)", lines=3, interactive=False
                    )
                    candidate_back = gr.Textbox(
                        label="Back-translation of the candidate", lines=3, interactive=False
                    )
                    candidate_findings_md = gr.Markdown()
                    use_revised_btn = gr.Button("Use revised version")
            with gr.Accordion("Previous translations", open=False):
                history_md = gr.Markdown("None yet.")

            gr.Markdown("### 4. Speak the translation\n"
                        "Output Format mirrors part A's (change either one); Speed comes "
                        "from part A.")
            # Which version of the Japanese will be spoken (plain, unrevised,
            # revised, or the user's edit); kept current as the text changes.
            speech_source_md = gr.Markdown("**Will speak:** nothing yet. Translate first.")
            with gr.Row(equal_height=True):
                translation_voice = gr.Dropdown(
                    choices=translation_voices,
                    value=translation_voices[0] if translation_voices else None,
                    label="Translation Voice",
                    scale=2
                )
                # A mirror of part A's Output Format: both always hold the same value.
                format_mirror = gr.Radio(
                    choices=["wav", "mp3", "aac"],
                    value="wav",
                    label="Output Format",
                    scale=1
                )
                speak_translation_btn = gr.Button(
                    "Generate Speech from Translation", variant="primary", scale=1
                )
            # Stitching: the part C video continues with the original-language
            # text, spoken by a second voice, after a 1-second title card.
            original_voices = [v for v in voices if get_language_code_from_voice(v) in ("a", "b")]
            stitch_checkbox = gr.Checkbox(
                label="Stitch original translation",
                value=False,
                info="Adds the original English to the part C video, spoken by a second voice, "
                     "after a 1-second \"Original text next...\" card."
            )
            # Same controls, in the same order, as part A's voice and text.
            with gr.Column(visible=False) as stitch_area:
                stitch_voice = gr.Dropdown(
                    choices=original_voices,
                    value=("af_heart" if "af_heart" in original_voices
                           else (original_voices[0] if original_voices else None)),
                    label="Voice"
                )
                stitch_text = gr.Textbox(
                    lines=3,
                    placeholder="Original (English) text to speak; filled from Source Text.",
                    label="Text"
                )
            translated_output = gr.Audio(label="Translated Audio")
            translation_status = gr.Textbox(
                label="Translation Status",
                value="Ready.",
                lines=3,
                interactive=False
            )

        # Part C: video. A still picture, the Japanese speech from part B, and
        # the English source text as burned-in subtitles timed to the speech.
        with gr.Column(variant="panel"):
            gr.Markdown("## C. Video\n"
                        "Uses the Japanese text and voice from part B (with the Speed from "
                        "part A), and the English source text as subtitles.")
            video_source_md = gr.Markdown("**Video will speak:** nothing yet. Translate first.")
            stitch_note_md = gr.Markdown("")
            with gr.Row():
                with gr.Column(scale=2):
                    # image_mode="RGB": gradio re-saves each upload under its
                    # original extension, and its default RGBA mode cannot be
                    # written as JPEG, so every .jpg photo would fail. format
                    # only applies to unnamed blobs such as the composite.
                    video_picture = gr.ImageEditor(
                        type="filepath",
                        image_mode="RGB",
                        format="png",
                        sources=["upload", "clipboard"],
                        transforms=("crop",),
                        brush=False,
                        eraser=False,
                        layers=False,
                        label="Background picture (drag to crop)"
                    )
                    change_picture_btn = gr.UploadButton(
                        "🖼 Change Picture", file_types=["image"], file_count="single"
                    )
                with gr.Column(scale=1):
                    video_size = gr.Dropdown(
                        choices=list(video_export.FRAME_SIZES),
                        value=list(video_export.FRAME_SIZES)[0],
                        label="Video Size"
                    )
                    video_fit = gr.Radio(
                        choices=video_export.FIT_MODES,
                        value=video_export.FIT_MODES[0],
                        label="Picture Fit"
                    )
                    preview_btn = gr.Button("Preview Frame")
                    create_video_btn = gr.Button("Create Video", variant="primary")
            with gr.Row():
                video_preview = gr.Image(label="Frame Preview", interactive=False)
                video_output = gr.Video(label="Video")
            video_status = gr.Textbox(
                label="Video Status",
                value="Ready.",
                lines=3,
                interactive=False
            )

        # The inputs and displays of the current review, for stale detection.
        review_state = gr.State(None)
        history_state = gr.State([])
        # Where the Japanese in the editable box came from.
        speech_origin = gr.State("")

        # Generator so the first, slow model load is announced before it starts.
        def translate_fn(text, source, dest):
            if not translation.is_model_loaded(source, dest):
                if translation.is_model_downloaded(source, dest):
                    yield gr.update(), "Loading translation model..."
                else:
                    yield gr.update(), "Downloading translation model (first use only)..."
            try:
                result = translation.translate(text, source, dest)
            except translation.TranslationError as e:
                yield gr.update(), str(e)
                return
            except Exception as e:
                yield gr.update(), f"Translation failed: {type(e).__name__}: {e}"
                return
            yield result, f"Translated {len(text)} character(s) {source} → {dest}."

        # Never falls back to another voice or language: a voice that does not
        # speak the destination language is rejected before any synthesis.
        def speak_translation_fn(voice_name, text, dest, format, speed):
            expected = translation.DEST_LANGUAGE_CODES.get(dest)
            if expected is None:
                return None, f"Speech is not supported for destination language {dest!r}."
            if not voice_name:
                return None, f"No {dest} voice selected. Install a {dest} voice and restart."
            try:
                voice_lang = get_language_code_from_voice(voice_name)
            except ValueError as e:
                return None, f"Invalid voice {voice_name!r}: {e}"
            if voice_lang != expected:
                return None, (f"Voice {voice_name!r} does not speak {dest}. "
                              f"Choose a {dest} voice.")
            if not text or not text.strip():
                return None, "Translate some text (or type it) before generating speech."
            return generate_tts_with_logs(voice_name, text, format, speed)

        stale_text = "STALE: the text changed after this review. Click Recheck Edited Translation."

        def _review_displays(result, ambiguities_md):
            unrevised = result["unrevised"]
            candidate = result["candidate"]
            findings = translation_review.format_assessment_md(unrevised)
            if ambiguities_md:
                findings = f"{ambiguities_md}\n\n{findings}"
            final = candidate["final"] if candidate else None
            return {
                "back": unrevised["back"],
                "findings": findings,
                "candidate_text": final["japanese"] if final else "",
                "candidate_back": final["back"] if final else "",
                "candidate_findings": translation_review.format_candidate_md(candidate),
            }

        def _review_status(result):
            meaning = sum(1 for f in result["unrevised"]["findings"] if f["kind"] == "meaning")
            candidate = result["candidate"]
            passes = len(candidate["passes"]) if candidate else 0
            unresolved = len(candidate["unresolved"]) if candidate else 0
            return (f"Review done (diagnostic only). Meaning findings in the unrevised "
                    f"Japanese: {meaning}. Revision passes: {passes} of max "
                    f"{translation_review.MAX_REVISION_PASSES}. Unresolved: {unresolved}.")

        def _run_review_events(source, japanese, context, glossary):
            """Yield ('progress', msg) then ('result', result) from the review core."""
            if not translation.is_back_model_loaded():
                yield "progress", ("Loading back-translation model..."
                                   if translation.is_back_model_downloaded()
                                   else "Downloading back-translation model (first use only)...")
            yield from translation_review.run_review(source, japanese, context, glossary)

        def _progress_line(title, stage, fraction, elapsed, frame):
            cells = 24
            filled = int(fraction * cells)
            bar = "█" * filled + "░" * (cells - filled)
            spinner = "◐◓◑◒"[frame % 4]
            return (f"**{spinner} {title}** — {stage}  \n"
                    f"`{bar}` {int(fraction * 100)}% · {elapsed:.0f} s elapsed (estimated)")

        def _ticking(work, title, progress, tick_seconds=0.5, creep_seconds=15.0):
            """Run a ('progress', msg)/(kind, payload) event generator in a
            thread, yielding (kind, payload, progress_markdown) and a "tick"
            every tick_seconds, so the page keeps moving while a model call
            blocks.

            Progress is an estimate: each new stage takes half of the remaining
            distance to 95%, and creeps toward its ceiling while it runs. It only
            reaches 100% when the work actually finishes.
            """
            events = queue.Queue()

            def runner():
                try:
                    for event in work():
                        events.put(event)
                    events.put(("_done", None))
                except Exception as e:  # re-raised in the handler's thread
                    events.put(("_error", e))

            threading.Thread(target=runner, daemon=True).start()
            start = stage_start = time.time()
            low, high, stage, frame = 0.0, 0.5, "starting...", 0
            while True:
                try:
                    kind, payload = events.get(timeout=tick_seconds)
                except queue.Empty:
                    kind, payload = "tick", None
                now = time.time()
                if kind == "progress":
                    low = low + (high - low) * (1 - math.exp(-(now - stage_start) / creep_seconds))
                    high = low + (0.95 - low) * 0.5
                    stage, stage_start = payload, now
                if kind == "_error":
                    raise payload
                if kind == "_done":
                    progress(1.0, desc="Done")
                    yield "done", None, f"**✓ {title}** — finished in {now - start:.0f} s."
                    return
                frame += 1
                fraction = low + (high - low) * (1 - math.exp(-(now - stage_start) / creep_seconds))
                progress(fraction, desc=stage)
                yield kind, payload, _progress_line(title, stage, fraction, now - start, frame)

        def _stale_displays(state):
            displays = state["displays"]
            return (f"[{stale_text}]\n\n{displays['back']}",
                    f"**⚠ {stale_text}**\n\n{displays['findings']}",
                    f"[{stale_text}]\n\n{displays['candidate_back']}",
                    f"**⚠ {stale_text}**\n\n{displays['candidate_findings']}")

        # Writes nothing until the whole run succeeds, so a failure leaves
        # every text box exactly as the user left it.
        def translate_review_fn(source, context, glossary_text, state, progress=gr.Progress()):
            unchanged = (gr.update(),) * 6
            started = time.time()

            def status(message, bar=gr.update()):
                return (*unchanged, message, gr.update(), bar)

            try:
                glossary = translation_review.parse_glossary(glossary_text)
                if not source or not source.strip():
                    raise translation_review.ReviewError("Enter English text to translate.")

                def work():
                    yield "progress", f"Translating with context ({translation_review.REVIEW_MODEL})..."
                    done = translation_review.translate_with_context(source, context, glossary)
                    yield "translated", done
                    yield from _run_review_events(source, done["translation"], context, glossary)

                translated = result = None
                for kind, payload, bar in _ticking(work, "Translating and reviewing", progress):
                    if kind == "progress":
                        yield status(payload, bar)
                    elif kind in ("tick", "done"):
                        yield status(gr.update(), bar)
                    elif kind == "translated":
                        translated = payload
                    elif kind == "result":
                        result = payload
                japanese = translated["translation"]
            except translation.TranslationError as e:
                yield status(str(e), f"**✗ Translating and reviewing** — stopped after "
                                     f"{time.time() - started:.0f} s.")
                return
            except Exception as e:
                yield status(f"Translate and Review failed: {type(e).__name__}: {e}",
                             f"**✗ Translating and reviewing** — stopped after "
                             f"{time.time() - started:.0f} s.")
                return
            ambiguities_md = translation_review.format_ambiguities_md(translated["ambiguities"])
            displays = _review_displays(result, ambiguities_md)
            new_state = {"source": source, "japanese": japanese, "context": context,
                         "glossary": glossary_text, "stale": False,
                         "ambiguities_md": ambiguities_md, "displays": displays}
            yield (japanese, displays["back"], displays["findings"], displays["candidate_text"],
                   displays["candidate_back"], displays["candidate_findings"],
                   _review_status(result), new_state, gr.update())

        # Reviews the Japanese exactly as currently edited; never writes to it.
        def recheck_fn(source, japanese, context, glossary_text, state, progress=gr.Progress()):
            unchanged = (gr.update(),) * 5
            started = time.time()

            def status(message, bar=gr.update()):
                return (*unchanged, message, gr.update(), bar)

            try:
                glossary = translation_review.parse_glossary(glossary_text)

                def work():
                    yield from _run_review_events(source, japanese, context, glossary)

                result = None
                for kind, payload, bar in _ticking(work, "Rechecking the edited translation",
                                                   progress):
                    if kind == "progress":
                        yield status(payload, bar)
                    elif kind in ("tick", "done"):
                        yield status(gr.update(), bar)
                    elif kind == "result":
                        result = payload
            except translation.TranslationError as e:
                yield status(str(e), f"**✗ Rechecking** — stopped after "
                                     f"{time.time() - started:.0f} s.")
                return
            except Exception as e:
                yield status(f"Recheck failed: {type(e).__name__}: {e}",
                             f"**✗ Rechecking** — stopped after {time.time() - started:.0f} s.")
                return
            # Translator ambiguity notes describe the source, so keep them only
            # while the source is unchanged.
            ambiguities_md = (state or {}).get("ambiguities_md", "") \
                if state and state.get("source") == source else ""
            displays = _review_displays(result, ambiguities_md)
            new_state = {"source": source, "japanese": japanese, "context": context,
                         "glossary": glossary_text, "stale": False,
                         "ambiguities_md": ambiguities_md, "displays": displays}
            yield (displays["back"], displays["findings"], displays["candidate_text"],
                   displays["candidate_back"], displays["candidate_findings"],
                   _review_status(result), new_state, gr.update())

        # Fires only on user edits (.input), not on programmatic updates.
        def mark_stale_fn(source, japanese, context, glossary_text, state):
            unchanged = (gr.update(),) * 4
            if not state or state["stale"]:
                return (*unchanged, state)
            snapshot = (state["source"], state["japanese"], state["context"], state["glossary"])
            if (source, japanese, context, glossary_text) == snapshot:
                return (*unchanged, state)
            state = {**state, "stale": True}
            return (*_stale_displays(state), state)

        def _history_markdown(history):
            if not history:
                return "None yet."
            return "\n".join(f"{i}. 「{entry}」" for i, entry in enumerate(history, start=1))

        # The only way a revision enters the editable Japanese.
        def use_revised_fn(candidate, current, history, state):
            if not candidate or not candidate.strip():
                return (gr.update(), history, gr.update(), *(gr.update(),) * 4,
                        state, "There is no revised candidate to use.")
            history = history + [current] if current and current.strip() else history
            if state:
                state = {**state, "stale": True}
                stale = _stale_displays(state)
            else:
                stale = (gr.update(),) * 4
            return (candidate, history, _history_markdown(history), *stale, state,
                    "Revised candidate copied into the Japanese text. The previous "
                    "version is under Previous translations. Recheck to review it.")

        # Separate listener so the existing speech handler stays untouched; it
        # writes to the findings panel, not the status box the speech result uses.
        def speak_stale_reminder_fn(state):
            if not state or not state["stale"]:
                return gr.update()
            return ("**⚠ Speech uses the current Japanese, which has not been reviewed "
                    "since it changed.**\n\n" + _stale_displays(state)[1])

        def _picture_path(picture):
            # ImageEditor delivers {"background", "layers", "composite"}; the
            # composite is the picture as cropped in the browser.
            if isinstance(picture, dict):
                return picture.get("composite") or picture.get("background")
            return picture

        # Both video handlers write only to part C components.
        def preview_frame_fn(picture, size_label, fit_mode, source):
            path = _picture_path(picture)
            if not path:
                return None, "Add a background picture first."
            size = video_export.FRAME_SIZES[size_label]
            sentences = video_export.english_sentences(source)
            subtitle = video_export.split_for_reading(sentences[0])[0] if sentences \
                else "(subtitle preview)"
            try:
                with tempfile.TemporaryDirectory(prefix="kokoro_preview_") as work:
                    work = Path(work)
                    background = video_export.prepare_background(path, size, fit_mode,
                                                                 work / "bg.png")
                    frame = video_export.render_preview(background, subtitle, size,
                                                        work / "preview.png")
                    from PIL import Image
                    with Image.open(frame) as opened:
                        image = opened.copy()
            except video_export.VideoError as e:
                return None, str(e)
            except Exception as e:
                return None, f"Preview failed: {type(e).__name__}: {e}"
            return image, f"Preview of the first frame ({size_label}, {fit_mode.split(' ')[0]})."

        def create_video_fn(picture, size_label, fit_mode, source, japanese,
                            voice_name, dest, speed, state,
                            stitch=False, stitch_voice_name=None, stitch_source=""):
            path = _picture_path(picture)
            expected = translation.DEST_LANGUAGE_CODES.get(dest)
            problem = None
            if not path:
                problem = "Add a background picture first."
            elif not source or not source.strip():
                problem = "The English source text (part B) is empty; it is used for the subtitles."
            elif not japanese or not japanese.strip():
                problem = "The Japanese text (part B) is empty; translate first."
            elif not voice_name:
                problem = f"No {dest} voice selected in part B."
            elif stitch and not (stitch_source or "").strip():
                problem = "Stitching is on, but its Text (part B step 4) is empty."
            elif stitch and not stitch_voice_name:
                problem = "Stitching is on, but no voice is selected for the original text."
            else:
                try:
                    if get_language_code_from_voice(voice_name) != expected:
                        problem = f"Voice {voice_name!r} does not speak {dest}. Choose a {dest} voice."
                    elif stitch and get_language_code_from_voice(stitch_voice_name) not in ("a", "b"):
                        problem = (f"Voice {stitch_voice_name!r} does not speak English. "
                                   "Choose an English voice for the original text.")
                except ValueError as e:
                    problem = f"Invalid voice: {e}"
            if problem:
                yield None, problem
                return

            size = video_export.FRAME_SIZES[size_label]
            try:
                video_export.ffmpeg_available()
                yield gr.update(), "Generating Japanese speech sentence by sentence..."
                clips, spans, _ = synthesize_sentences(voice_name, japanese, speed)
                cues, pauses, method, detail = video_export.build_timeline(source, spans)
                audio = video_export.assemble_audio(clips, pauses, SAMPLE_RATE)
                cards = []
                if stitch:
                    # Part 2: a title card over silence, then the original text
                    # spoken by the second voice, with its own timed subtitles.
                    yield gr.update(), "Generating the original English speech for stitching..."
                    clips2, spans2, _ = synthesize_sentences(stitch_voice_name, stitch_source, speed)
                    cues2, pauses2, _, detail2 = video_export.build_timeline(stitch_source, spans2)
                    card_start = len(audio) / SAMPLE_RATE
                    part2_start = card_start + video_export.CARD_SECONDS
                    cards = [(card_start, part2_start, video_export.CARD_TEXT)]
                    cues = list(cues) + [(s + part2_start, e + part2_start, t) for s, e, t in cues2]
                    audio = np.concatenate([
                        audio,
                        np.zeros(int(round(video_export.CARD_SECONDS * SAMPLE_RATE)), dtype=np.float32),
                        video_export.assemble_audio(clips2, pauses2, SAMPLE_RATE),
                    ])
                yield gr.update(), "Encoding video..."
                DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                enforce_output_retention()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = DEFAULT_OUTPUT_DIR / f"tts_video_{timestamp}_{uuid.uuid4().hex}.mp4"
                with tempfile.TemporaryDirectory(prefix="kokoro_video_in_") as work:
                    work = Path(work)
                    sf.write(work / "speech.wav", audio, SAMPLE_RATE)
                    background = video_export.prepare_background(path, size, fit_mode,
                                                                 work / "bg.png")
                    video_export.render_video(background, work / "speech.wav",
                                              video_export.build_ass(cues, size, cards), out_path)
            except video_export.VideoError as e:
                yield None, str(e)
                return
            except Exception as e:
                yield None, f"Video failed: {type(e).__name__}: {e}"
                return

            seconds = len(audio) / SAMPLE_RATE
            megabytes = out_path.stat().st_size / (1024 * 1024)
            message = (f"Video created: {seconds:.1f} s, {megabytes:.1f} MB, {size_label}, "
                       f"{fit_mode.split(' ')[0]}.\nSubtitle timing: {detail}.")
            if stitch:
                message += (f"\nStitched: \"{video_export.CARD_TEXT}\" card at {card_start:.1f} s, "
                            f"then the original English ({stitch_voice_name}) from "
                            f"{part2_start:.1f} s; its subtitle timing: {detail2}.")
            if state and state.get("stale"):
                message += ("\n⚠ The Japanese has changed since its last review in part B "
                            "(the review is stale).")
            yield str(out_path), message

        # Function to load a preset. Returns gr.update() (leave unchanged)
        # rather than None on the error paths: None is not a valid Slider
        # value and would send speed=None into the pipeline on the next run.
        def load_preset_fn(preset_name):
            unchanged = (gr.update(), gr.update(), gr.update(), gr.update())
            if not preset_name:
                return (*unchanged, "Select a preset to load.")

            preset = speed_dial.get_preset(preset_name)
            if not preset:
                return (*unchanged, f"Preset {preset_name!r} could not be loaded.")

            return (preset["voice"], preset["text"], preset["format"],
                    preset["speed"], f"Loaded preset {preset_name!r}.")

        # Function to save a preset. The status string goes to the status box,
        # never into the dropdown — a dropdown whose selected value is an error
        # message will feed that message back to get_preset() on the next Load.
        def save_preset_fn(name, voice, text, format, speed):
            if not name or not voice or not text:
                return gr.update(), "Provide a name, a voice, and text to save a preset."

            success = speed_dial.save_preset(name, voice, text, format, speed)

            # Update the dropdown with the new preset list
            preset_names = speed_dial.get_preset_names()

            if success:
                return (gr.update(choices=preset_names, value=name),
                        f"Saved preset {name!r}.")
            return (gr.update(choices=preset_names),
                    f"Could not save preset {name!r}. Check the console for details.")

        # Function to delete a preset
        def delete_preset_fn(name):
            if not name:
                return gr.update(), "Select a preset to delete."

            success = speed_dial.delete_preset(name)

            # Update the dropdown with the new preset list
            preset_names = speed_dial.get_preset_names()

            if success:
                return (gr.update(choices=preset_names, value=None),
                        f"Deleted preset {name!r}.")
            return (gr.update(choices=preset_names),
                    f"Could not delete preset {name!r}. Check the console for details.")

        def _describe_speech(origin, japanese):
            text = (japanese or "").strip()
            if not text:
                empty = "nothing yet. Translate first."
                return f"**Will speak:** {empty}", f"**Video will speak:** {empty}"
            preview = " ".join(text.split())
            if len(preview) > 70:
                preview = preview[:70] + "…"
            label = origin or "Japanese typed into the box"
            return (f"**Will speak:** {label}  \n「{preview}」",
                    f"**Video will speak:** {label}  \n「{preview}」")

        def _origin_after(success_prefix, label):
            """Record where the Japanese came from, only when the action succeeded."""
            def set_origin(status_message, origin, japanese):
                if (status_message or "").startswith(success_prefix):
                    origin = label
                return (origin, *_describe_speech(origin, japanese))
            return set_origin

        def edited_origin_fn(origin, japanese):
            if not origin.startswith("Your edit"):
                origin = f"Your edit (started from: {origin or 'an empty box'})"
            return (origin, *_describe_speech(origin, japanese))

        origin_outputs = [speech_origin, speech_source_md, video_source_md]

        # Dictation into part B's Source Text.
        def dictation_start_fn(source):
            try:
                if not speech_to_text.is_loaded():
                    gr.Info("Loading the speech-to-text model (first use can take a minute)...")
                    speech_to_text._load()
            except speech_to_text.SpeechToTextError as e:
                gr.Warning(str(e))
                return None, str(e)
            gr.Info("🎤 Listening: speak now. Your words are added to Source Text as you talk.")
            return (speech_to_text.Dictation(source),
                    "🎤 Listening... speak now; text appears in Source Text after each pause.")

        def dictation_stream_fn(chunk, state, source):
            if chunk is None:
                return gr.update(), state
            if state is None:  # a chunk that arrived without (or after) start
                state = speech_to_text.Dictation(source)
            sample_rate, audio = chunk
            try:
                return state.feed(sample_rate, audio), state
            except speech_to_text.SpeechToTextError as e:
                gr.Warning(str(e))
                return gr.update(), state

        def dictation_stop_fn(state):
            if state is None:
                return gr.update(), None, "Dictation stopped."
            try:
                text = state.finish()
            except speech_to_text.SpeechToTextError as e:
                gr.Warning(str(e))
                return gr.update(), None, str(e)
            words = state.dictated_words()
            if words:
                gr.Success(f"Dictation added {words} word(s) to Source Text.")
            else:
                gr.Warning("No speech was recognised. Check the microphone and try again.")
            return text, None, f"Dictation finished: {words} word(s) added to Source Text."

        def _mirror(value):
            return value

        def change_picture_fn(uploaded):
            path = uploaded if isinstance(uploaded, str) else getattr(uploaded, "name", None)
            if not path:
                return gr.update(), gr.update(), "No picture was chosen."
            gr.Info("Picture replaced.")
            return ({"background": path, "layers": [], "composite": path}, None,
                    "Picture replaced. Use Preview Frame or Create Video with the new picture.")

        # Every button gives visible feedback: on click (without waiting for a
        # running job, see concurrency below) the button turns into a
        # disabled busy label, its status box says what is running, and a toast
        # confirms the start; when the work ends the button is restored and a
        # Success or Warning toast reports the outcome from the status box.
        def _with_feedback(button, busy_label, started, status_box, success_prefix, **listener):
            idle_label = button.value

            def begin():
                gr.Info(started, duration=4)
                return gr.update(value=busy_label, interactive=False), f"⏳ {started}"

            def finish(message):
                text = (message or "").strip()
                first = text.splitlines()[0] if text else ""
                notes = [line[2:] for line in text.splitlines()[1:] if line.startswith("- ")]
                if first.startswith(success_prefix) and notes:
                    # Done, but with caveats (e.g. audio cut short): never a green toast.
                    gr.Warning(f"{first}<br>" + "<br>".join(notes[:3]), duration=12)
                elif first.startswith(success_prefix):
                    gr.Success(first, duration=5)
                elif first.startswith("⏳"):
                    # The handler never ran (e.g. gradio rejected an input).
                    gr.Warning(f"{idle_label} did not complete: the request was rejected "
                               "before it started.", duration=10)
                else:
                    gr.Warning(text[:300] or f"{idle_label} did not complete.", duration=10)
                return gr.update(value=idle_label, interactive=True)

            # Toasts are only delivered to queued events, so begin/finish are
            # queued; concurrency_limit=None keeps them from waiting behind a
            # long job, so the feedback is still immediate.
            return (button.click(fn=begin, outputs=[button, status_box],
                                 concurrency_limit=None, show_progress="hidden")
                    .then(**listener)
                    .then(fn=finish, inputs=[status_box], outputs=[button],
                          concurrency_limit=None, show_progress="hidden"))

        # Connect the buttons to their functions
        _with_feedback(
            load_preset, "⏳ Loading...", "Loading preset...", status, "Loaded preset",
            fn=load_preset_fn,
            inputs=preset_dropdown,
            outputs=[voice, text, format, speed, status]
        )

        _with_feedback(
            save_preset, "⏳ Saving...", "Saving preset...", status, "Saved preset",
            fn=save_preset_fn,
            inputs=[preset_name, voice, text, format, speed],
            outputs=[preset_dropdown, status]
        )

        _with_feedback(
            delete_preset, "⏳ Deleting...", "Deleting preset...", status, "Deleted preset",
            fn=delete_preset_fn,
            inputs=preset_dropdown,
            outputs=[preset_dropdown, status]
        )

        # Connect the generate button
        _with_feedback(
            generate, "⏳ Generating speech...", "Generating speech...", status, "Generated",
            fn=generate_tts_with_logs,
            inputs=[voice, text, format, speed],
            outputs=[output, status]
        )

        _with_feedback(
            translate_btn, "⏳ Translating...", "Translating...", translation_status, "Translated",
            fn=translate_fn,
            inputs=[source_text, source_lang, dest_lang],
            outputs=[translated_text, translation_status]
        ).then(
            fn=_origin_after("Translated", "Plain translation (Translate button, not reviewed)"),
            inputs=[translation_status, speech_origin, translated_text],
            outputs=origin_outputs, queue=False, show_progress="hidden"
        )

        # Format and speed are shared with the direct TTS controls.
        _with_feedback(
            speak_translation_btn, "⏳ Generating speech...", "Generating speech from the translation...",
            translation_status, "Generated",
            fn=speak_translation_fn,
            inputs=[translation_voice, translated_text, dest_lang, format, speed],
            outputs=[translated_output, translation_status]
        )

        review_display_outputs = [back_translation_box, review_findings_md, candidate_text,
                                  candidate_back, candidate_findings_md]

        _with_feedback(
            translate_review_btn, "⏳ Translating and reviewing...",
            "Translating and reviewing (this can take a minute)...", translation_status, "Review done",
            fn=translate_review_fn,
            inputs=[source_text, review_context, review_glossary, review_state],
            outputs=[translated_text, *review_display_outputs, translation_status, review_state,
                     review_progress_md]
        ).then(
            fn=_origin_after("Review done", "**Unrevised** translation (from Translate and Review)"),
            inputs=[translation_status, speech_origin, translated_text],
            outputs=origin_outputs, queue=False, show_progress="hidden"
        )

        _with_feedback(
            recheck_btn, "⏳ Rechecking...", "Rechecking the edited translation...",
            translation_status, "Review done",
            fn=recheck_fn,
            inputs=[source_text, translated_text, review_context, review_glossary, review_state],
            outputs=[*review_display_outputs, translation_status, review_state, review_progress_md]
        )

        for edited in (translated_text, source_text, review_context, review_glossary):
            edited.input(
                fn=mark_stale_fn,
                inputs=[source_text, translated_text, review_context, review_glossary, review_state],
                outputs=[back_translation_box, review_findings_md, candidate_back,
                         candidate_findings_md, review_state]
            )

        _with_feedback(
            use_revised_btn, "⏳ Applying...", "Using the revised version...",
            translation_status, "Revised candidate copied",
            fn=use_revised_fn,
            inputs=[candidate_text, translated_text, history_state, review_state],
            outputs=[translated_text, history_state, history_md, back_translation_box,
                     review_findings_md, candidate_back, candidate_findings_md, review_state,
                     translation_status]
        ).then(
            fn=_origin_after("Revised candidate copied",
                             "**Revised** candidate (applied with Use revised version)"),
            inputs=[translation_status, speech_origin, translated_text],
            outputs=origin_outputs, queue=False, show_progress="hidden"
        )

        # Typing in the Japanese box makes it the user's own version.
        translated_text.input(
            fn=edited_origin_fn,
            inputs=[speech_origin, translated_text],
            outputs=origin_outputs, queue=False, show_progress="hidden"
        )

        speak_translation_btn.click(
            fn=speak_stale_reminder_fn,
            inputs=[review_state],
            outputs=[review_findings_md]
        )

        source_mic.start_recording(
            fn=dictation_start_fn,
            inputs=[source_text],
            outputs=[dictation_state, translation_status]
        )
        source_mic.stream(
            fn=dictation_stream_fn,
            inputs=[source_mic, dictation_state, source_text],
            outputs=[source_text, dictation_state],
            stream_every=1.0, time_limit=None, show_progress="hidden"
        )
        # Dictated text changes the source, so an existing review becomes stale.
        source_mic.stop_recording(
            fn=dictation_stop_fn,
            inputs=[dictation_state],
            outputs=[source_text, dictation_state, translation_status]
        ).then(
            fn=mark_stale_fn,
            inputs=[source_text, translated_text, review_context, review_glossary, review_state],
            outputs=[back_translation_box, review_findings_md, candidate_back,
                     candidate_findings_md, review_state]
        )

        # Part A's Output Format and part B's mirror always hold the same value.
        # A -> B on any change (including Load preset); B -> A on user input only,
        # so the pair can never loop.
        format.change(fn=_mirror, inputs=format, outputs=format_mirror,
                      queue=False, show_progress="hidden")
        format_mirror.input(fn=_mirror, inputs=format_mirror, outputs=format,
                            queue=False, show_progress="hidden")

        change_picture_btn.upload(
            fn=change_picture_fn,
            inputs=change_picture_btn,
            outputs=[video_picture, video_preview, video_status]
        )

        def _stitch_note(checked, voice_name):
            if not checked:
                return ""
            return (f"**Stitched video:** after the Japanese part, a 1-second "
                    f"\"{video_export.CARD_TEXT}\" card, then the original English spoken by "
                    f"**{voice_name or 'the selected voice'}** (part B step 4), on the same picture.")

        def stitch_toggle_fn(checked, text_now, source):
            # Prefill the original text from Source Text when it is still empty.
            fill = text_now if (text_now or "").strip() else (source or "")
            return gr.update(visible=checked), fill

        stitch_checkbox.change(
            fn=stitch_toggle_fn,
            inputs=[stitch_checkbox, stitch_text, source_text],
            outputs=[stitch_area, stitch_text],
            queue=False, show_progress="hidden"
        ).then(
            fn=_stitch_note, inputs=[stitch_checkbox, stitch_voice], outputs=stitch_note_md,
            queue=False, show_progress="hidden"
        )
        stitch_voice.change(
            fn=_stitch_note, inputs=[stitch_checkbox, stitch_voice], outputs=stitch_note_md,
            queue=False, show_progress="hidden"
        )

        _with_feedback(
            preview_btn, "⏳ Rendering preview...", "Rendering a preview frame...",
            video_status, "Preview of",
            fn=preview_frame_fn,
            inputs=[video_picture, video_size, video_fit, source_text],
            outputs=[video_preview, video_status]
        )

        # review_state is read only, to report a stale review; never written.
        _with_feedback(
            create_video_btn, "⏳ Creating video...", "Creating the video (speech, then encoding)...",
            video_status, "Video created",
            fn=create_video_fn,
            inputs=[video_picture, video_size, video_fit, source_text, translated_text,
                    translation_voice, dest_lang, speed, review_state,
                    stitch_checkbox, stitch_voice, stitch_text],
            outputs=[video_output, video_status]
        )

    # Bound the queue: gradio defaults to max_size=None (unlimited backlog)
    # with default_concurrency_limit=1, so one client can stack unbounded
    # pending jobs against the single TTS worker.
    interface.queue(max_size=32)

    # Launch interface
    launch_kwargs = dict(
        server_name=server_name,
        server_port=server_port,
        share=False
    )
    supported = inspect.signature(interface.launch).parameters
    if "strict_cors" in supported:
        # Tighten the framework's permissive origin reflection where the
        # installed gradio supports it (the any-origin echo with
        # credentials is an upstream gradio defect).
        launch_kwargs["strict_cors"] = True
    if auth is not None:
        launch_kwargs["auth"] = auth
    interface.launch(**launch_kwargs)

def cleanup_resources():
    """Properly clean up resources when the application exits"""
    try:
        print("Cleaning up resources...")
        shutdown_pipelines()

        # Clear CUDA memory explicitly
        if torch.cuda.is_available():
            try:
                # Get initial memory usage
                try:
                    initial = torch.cuda.memory_allocated()
                    initial_mb = initial / (1024 * 1024)
                    print(f"CUDA memory before cleanup: {initial_mb:.2f} MB")
                except:
                    pass

                # Free memory
                print("Clearing CUDA cache...")
                try:
                    torch.cuda.synchronize()
                except:
                    pass
                torch.cuda.empty_cache()

                # Get final memory usage
                try:
                    final = torch.cuda.memory_allocated()
                    final_mb = final / (1024 * 1024)
                    freed_mb = (initial - final) / (1024 * 1024)
                    print(f"CUDA memory after cleanup: {final_mb:.2f} MB (freed {freed_mb:.2f} MB)")
                except:
                    pass
            except Exception as ce:
                print(f"Error clearing CUDA memory: {type(ce).__name__}: {ce}")

        # Final garbage collection
        try:
            import gc
            collected = gc.collect()
            print(f"Garbage collection completed: {collected} objects collected")
        except Exception as gce:
            print(f"Error during garbage collection: {type(gce).__name__}: {gce}")

        print("Cleanup completed")

    except Exception as e:
        print(f"Error during cleanup: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

import atexit
import signal
import sys

def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}, shutting down...")
    raise SystemExit(0)

def register_cleanup_handlers():
    """Register process handlers once, and only from the main thread."""
    if threading.current_thread() is not threading.main_thread():
        return False
    atexit.register(cleanup_resources)
    supported = {signal.SIGINT}
    if hasattr(signal, 'SIGTERM'):
        supported.add(signal.SIGTERM)
    for sig in supported:
        try:
            signal.signal(sig, signal_handler)
        except (ValueError, OSError, RuntimeError):
            pass
    return True

def parse_arguments():
    """Parse command line arguments for host and port configuration."""
    parser = argparse.ArgumentParser(
        description="Kokoro TTS Local Generator - Gradio Web Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number to run the server on"
    )
    parser.add_argument(
        "--username",
        type=str,
        default=os.environ.get("KOKORO_TTS_USERNAME"),
        help="Username for Gradio authentication "
             "(or set KOKORO_TTS_USERNAME env var)"
    )
    parser.add_argument(
        "--password",
        type=str,
        default=os.environ.get("KOKORO_TTS_PASSWORD"),
        help="Password for Gradio authentication "
             "(or set KOKORO_TTS_PASSWORD env var; preferred to avoid "
             "leaking via shell history / process list)"
    )
    return parser.parse_args()

def main() -> None:
    cleanup_registered = False
    try:
        cleanup_registered = register_cleanup_handlers()
        args = parse_arguments()
        auth = None
        if args.username and args.password:
            auth = [(args.username, args.password)]
        elif args.username or args.password:
            raise ValueError("--username and --password must both be set")
        try:
            is_loopback = ipaddress.ip_address(args.host).is_loopback
        except ValueError:
            is_loopback = args.host.lower() == 'localhost'
        if not is_loopback and auth is None:
            raise ValueError("Non-loopback binding requires username and password")
        create_interface(server_name=args.host, server_port=args.port, auth=auth)
    finally:
        # Ensure cleanup even if Gradio encounters an error
        cleanup_resources()
        if cleanup_registered:
            atexit.unregister(cleanup_resources)

if __name__ == "__main__":
    main()
