"""
Video export for part C: a still background picture, Japanese speech, and the
original English as burned-in subtitles timed to that speech.

Subtitle timing anchors on measured audio (each Japanese sentence is
synthesised separately, so its real start and end are known). An English
speaking-cadence model is used only to share measured time between English
pieces when sentences do not map one-to-one, and to keep every subtitle on
screen long enough to read.
"""

import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------- constants

ENGLISH_SYLLABLES_PER_SEC = 4.5   # average conversational rate (~150 words/min)
MAX_CPS = 17                      # subtitle reading speed, characters per second
MIN_CUE_SEC = 1.0
BASE_PAUSE_SEC = 0.3              # pause inserted between Japanese sentences
MAX_EXTRA_PAUSE_SEC = 2.0         # most extra reading time added after a sentence
MAX_LINE_CHARS = 42
MAX_CUE_CHARS = 2 * MAX_LINE_CHARS

SILENCE_DBFS = -40.0              # speech onset/offset threshold (measured, 0003 step 1b)
LEAD_PAD_SEC = 0.05               # audio kept before the measured onset
TAIL_PAD_SEC = 0.15               # audio kept after the measured offset

NON_ONE_TO_ONE_PENALTY = 0.6
IMPLAUSIBLE_AVG_COST = 1.2
GROUPINGS = [(1, 1), (2, 1), (1, 2), (3, 1), (1, 3), (2, 2)]

FRAME_SIZES: Dict[str, Tuple[int, int]] = {
    "1920×1080 (16:9)": (1920, 1080),
    "1280×720 (16:9)": (1280, 720),
    "1080×1920 (9:16 vertical)": (1080, 1920),
}
FIT_MODES = ["Fit (whole picture, with bars)", "Fill (crop to frame)"]
# None = unlimited (dxb, 2026-10-12). This turns off Pillow's guard against
# decompression-bomb images; a huge picture just takes longer and more RAM.
MAX_IMAGE_PIXELS = None


class VideoError(Exception):
    """A video failure whose message is safe to show in the UI."""


# ---------------------------------------------------------------- cadence model

_VOWEL_GROUPS = re.compile(r"[aeiouy]+")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[A-Za-z']+")


def _word_syllables(word: str) -> int:
    w = word.lower().strip("'")
    if not w:
        return 0
    count = len(_VOWEL_GROUPS.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and count > 1:
        count -= 1  # silent final e: "make", "time"
    return max(1, count)


def _number_syllables(number: str) -> float:
    digits = number.replace(",", "").split(".")[0]
    # Roughly two syllables per spoken digit group, plus "hundred"/"thousand".
    return max(1.0, 1.5 * len(digits.lstrip("0") or "0") + (1 if len(digits) >= 3 else 0))


def spoken_weight(text: str) -> float:
    """Estimated syllables needed to say `text` aloud (English cadence model)."""
    weight = 0.0
    for number in _NUMBER.findall(text):
        weight += _number_syllables(number)
    for word in _WORD.findall(_NUMBER.sub(" ", text)):
        weight += _word_syllables(word)
    weight += 0.5 * len(re.findall(r"[,;:]", text))
    return max(1.0, weight)


def english_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


# ---------------------------------------------------------------- alignment

def align(english_weights: Sequence[float], ja_durations: Sequence[float]) -> Optional[List[Tuple[int, int, int, int]]]:
    """Monotonic length-based alignment (Gale–Church style).

    Returns groups as (en_start, en_end, ja_start, ja_end) half-open index
    ranges, or None when no plausible alignment exists.
    """
    n, m = len(english_weights), len(ja_durations)
    if n == 0 or m == 0:
        return None
    total_en, total_ja = float(sum(english_weights)), float(sum(ja_durations))
    if total_en <= 0 or total_ja <= 0:
        return None
    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if cost[i][j] == inf:
                continue
            for di, dj in GROUPINGS:
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                share_en = sum(english_weights[i:ni]) / total_en
                share_ja = sum(ja_durations[j:nj]) / total_ja
                step = abs(math.log(max(share_en, 1e-9) / max(share_ja, 1e-9)))
                step += 0.0 if (di, dj) == (1, 1) else NON_ONE_TO_ONE_PENALTY
                if cost[i][j] + step < cost[ni][nj]:
                    cost[ni][nj] = cost[i][j] + step
                    back[ni][nj] = (i, j)
    if cost[n][m] == inf:
        return None
    groups = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj = back[i][j]
        groups.append((pi, i, pj, j))
        i, j = pi, pj
    groups.reverse()
    if cost[n][m] / len(groups) > IMPLAUSIBLE_AVG_COST:
        return None
    return groups


# ---------------------------------------------------------------- readability

_CLAUSE_BREAKS = [re.compile(r"(?<=[,;:])\s+"),
                  re.compile(r"\s+(?=(?:and|but|or|so|because|when|if|while)\b)", re.I)]


def split_for_reading(text: str) -> List[str]:
    """Split a subtitle longer than two lines at clause, then word, boundaries."""
    text = text.strip()
    if len(text) <= MAX_CUE_CHARS:
        return [text]
    for pattern in _CLAUSE_BREAKS:
        parts = [p.strip() for p in pattern.split(text) if p.strip()]
        if len(parts) > 1:
            pieces, current = [], ""
            for part in parts:
                candidate = f"{current} {part}".strip()
                if current and len(candidate) > MAX_CUE_CHARS:
                    pieces.append(current)
                    current = part
                else:
                    current = candidate
            pieces.append(current)
            if len(pieces) > 1:
                return [q for p in pieces for q in split_for_reading(p)]
    words = text.split()
    pieces, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > MAX_CUE_CHARS:
            pieces.append(current)
            current = word
        else:
            current = candidate
    pieces.append(current)
    return pieces


def _reading_need(text: str) -> float:
    return max(MIN_CUE_SEC, len(text) / MAX_CPS)


def _share(texts: Sequence[str], start: float, end: float) -> List[Tuple[float, float, str]]:
    """Split [start, end] between texts: each first gets its reading time
    (when the window allows), and the rest is shared by spoken weight."""
    weights = [spoken_weight(t) for t in texts]
    needs = [_reading_need(t) for t in texts]
    window = end - start
    if sum(needs) <= window + 1e-6:
        spare = max(0.0, window - sum(needs))
        spans = [n + spare * w / sum(weights) for n, w in zip(needs, weights)]
    else:
        # Not enough time even after the pause limit: share by reading need,
        # so every cue is equally (and minimally) over the reading speed.
        spans = [window * n / sum(needs) for n in needs]
    cues, t = [], start
    for text, span in zip(texts, spans):
        cues.append((t, t + span, text))
        t += span
    return cues


# ---------------------------------------------------------------- timeline

def build_timeline(english_text: str, ja_spans: Sequence[Tuple[float, float, float]]):
    """Build subtitle cues for speech made of Japanese sentences.

    ja_spans: (onset, offset, clip_length) of each Japanese sentence, in
    seconds, measured within that sentence's own trimmed clip.

    Returns (cues, pauses, method, detail):
      cues    [(start, end, text)] on the final assembled timeline
      pauses  seconds of silence to insert after each Japanese sentence
      method  "exact" | "aligned" | "estimated"
      detail  human-readable description of how timing was decided
    """
    sentences = english_sentences(english_text)
    if not sentences:
        raise VideoError("The English text is empty; there is nothing to subtitle.")
    if not ja_spans:
        raise VideoError("There is no Japanese speech to time subtitles against.")
    clip_lengths = [length for _, _, length in ja_spans]
    speech = [offset - onset for onset, offset, _ in ja_spans]
    weights = [spoken_weight(s) for s in sentences]
    m = len(ja_spans)

    if len(sentences) == m:
        # Equal counts: sentence i was translated as sentence i, so the
        # pairing is known. Length differences between languages are normal.
        groups = [(k, k + 1, k, k + 1) for k in range(m)]
    else:
        groups = align(weights, speech)
    if groups is None:
        method = "estimated"
        detail = (f"estimated from speaking cadence (English {len(sentences)} "
                  f"sentence(s), Japanese {m}; no plausible sentence alignment)")
        groups = [(0, len(sentences), 0, m)]
        per_group_text = [sentences]
    else:
        merges = [f"{ei - es}↔{je - js} at Japanese sentence {js + 1}"
                  for es, ei, js, je in groups if (ei - es, je - js) != (1, 1)]
        method = "aligned" if merges else "exact"
        if not merges:
            detail = f"exact ({m} sentence pair{'s' if m != 1 else ''})"
        else:
            detail = f"aligned ({'; '.join(merges)})"
            if any(ei - es > 1 for es, ei, _, _ in groups):
                detail += "; timing inside merged groups estimated from speaking cadence"
        per_group_text = [sentences[es:ei] for es, ei, _, _ in groups]

    pauses = [BASE_PAUSE_SEC] * m

    def sentence_start(k):
        return sum(clip_lengths[:k]) + sum(pauses[:k])

    def group_window(js, je):
        start = sentence_start(js) + ja_spans[js][0]
        end = sentence_start(je - 1) + ja_spans[je - 1][1]
        return start, end

    # Readability pass: lengthen the pause after a group whose subtitles need
    # more reading time than the speech gives them.
    for (es, ei, js, je), texts in zip(groups, per_group_text):
        pieces = [p for t in texts for p in split_for_reading(t)]
        start, end = group_window(js, je)
        needed = sum(_reading_need(p) for p in pieces)
        available = end - start + pauses[je - 1]
        if needed > available:
            pauses[je - 1] += min(MAX_EXTRA_PAUSE_SEC, needed - available)

    cues = []
    for (es, ei, js, je), texts in zip(groups, per_group_text):
        start, end = group_window(js, je)
        end += pauses[je - 1]  # hold the subtitle through the following pause
        pieces = [p for t in texts for p in split_for_reading(t)]
        cues.extend(_share(pieces, start, end))
    # Guarantee order and no overlap.
    tidy = []
    for start, end, text in cues:
        if tidy and start < tidy[-1][1]:
            start = tidy[-1][1]
        tidy.append((round(start, 3), round(max(end, start + 0.05), 3), text))
    fast = sum(1 for s, e, t in tidy if len(t) / (e - s) > MAX_CPS + 0.05)
    if fast:
        detail += (f"; {fast} subtitle(s) faster than {MAX_CPS} characters/s "
                   f"(the {MAX_EXTRA_PAUSE_SEC:.0f} s extra-pause limit was reached)")
    return tidy, pauses, method, detail


# ---------------------------------------------------------------- audio helpers

def trim_silence(audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, float, float]:
    """Trim a sentence clip to its speech plus small pads.

    Returns (clip, onset, offset), with onset/offset in seconds within the clip.
    """
    audio = np.asarray(audio, dtype=np.float32)
    frame = max(1, int(0.02 * sample_rate))
    n = len(audio) // frame
    if n == 0:
        return audio, 0.0, len(audio) / sample_rate
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    voiced = np.where(20 * np.log10(rms) > SILENCE_DBFS)[0]
    if voiced.size == 0:
        return audio, 0.0, len(audio) / sample_rate
    first = max(0, voiced[0] * frame - int(LEAD_PAD_SEC * sample_rate))
    last = min(len(audio), (voiced[-1] + 1) * frame + int(TAIL_PAD_SEC * sample_rate))
    clip = audio[first:last]
    onset = voiced[0] * frame / sample_rate - first / sample_rate
    offset = (voiced[-1] + 1) * frame / sample_rate - first / sample_rate
    return clip, onset, offset


def assemble_audio(clips: Sequence[np.ndarray], pauses: Sequence[float], sample_rate: int) -> np.ndarray:
    parts = []
    for clip, pause in zip(clips, pauses):
        parts.append(np.asarray(clip, dtype=np.float32))
        parts.append(np.zeros(int(round(pause * sample_rate)), dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------- picture

def prepare_background(picture_path: str, size: Tuple[int, int], fit_mode: str, out_png: Path) -> Path:
    """Normalise a picture into an exact-size RGB PNG frame."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with Image.open(picture_path) as opened:
            opened.seek(0)  # first frame of an animated image
            image = ImageOps.exif_transpose(opened).convert("RGBA")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as e:
        raise VideoError(f"The picture could not be read ({type(e).__name__}: {e}).") from e

    # Flatten transparency onto black, matching the padding colour.
    flat = Image.new("RGB", image.size, (0, 0, 0))
    flat.paste(image, mask=image.getchannel("A"))
    width, height = size
    if fit_mode.startswith("Fill"):
        frame = ImageOps.fit(flat, (width, height), method=Image.LANCZOS)
    else:
        frame = ImageOps.pad(flat, (width, height), method=Image.LANCZOS, color=(0, 0, 0))
    frame.save(out_png)
    return out_png


# ---------------------------------------------------------------- subtitles

def _ass_time(seconds: float) -> str:
    centis = int(round(max(0.0, seconds) * 100))
    h, rem = divmod(centis, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_text(text: str) -> str:
    text = text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
    return " ".join(text.split())


def build_ass(cues: Sequence[Tuple[float, float, str]], size: Tuple[int, int]) -> str:
    width, height = size
    font_size = max(24, round(height * 0.055)) if width >= height else max(24, round(width * 0.06))
    outline = max(2, round(height * 0.004))
    margin_v = round(height * 0.06)
    margin_h = round(width * 0.05)
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,Arial,{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        f"0,0,0,0,100,100,0,0,1,{outline},{max(1, outline // 2)},2,{margin_h},{margin_h},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for start, end, text in cues:
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{_ass_text(text)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- ffmpeg

def ffmpeg_available() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise VideoError("ffmpeg was not found on PATH. Install ffmpeg to create videos.")
    return path


def _run_ffmpeg(args: List[str], cwd: Path) -> None:
    command = [ffmpeg_available(), "-y", "-hide_banner", "-loglevel", "error", *args]
    result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-8:])
        raise VideoError(f"ffmpeg failed (exit {result.returncode}):\n{tail}")


def render_video(bg_png: Path, wav_path: Path, ass_text: str, out_path: Path) -> Path:
    """Encode still picture + speech + burned-in subtitles to an mp4."""
    with tempfile.TemporaryDirectory(prefix="kokoro_video_") as work:
        work = Path(work)
        shutil.copyfile(bg_png, work / "bg.png")
        shutil.copyfile(wav_path, work / "speech.wav")
        (work / "subs.ass").write_text(ass_text, encoding="utf-8")
        # Relative paths inside the working folder: no Windows "C:\" escaping
        # problems in the subtitles filter.
        _run_ffmpeg(["-loop", "1", "-i", "bg.png", "-i", "speech.wav",
                     "-vf", "subtitles=subs.ass", "-c:v", "libx264", "-tune", "stillimage",
                     "-pix_fmt", "yuv420p", "-r", "25", "-c:a", "aac", "-b:a", "192k",
                     "-shortest", "-movflags", "+faststart", "out.mp4"], work)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(work / "out.mp4"), str(out_path))
    return out_path


def render_preview(bg_png: Path, subtitle: str, size: Tuple[int, int], out_png: Path) -> Path:
    """Render one frame with the real subtitle style, matching the video."""
    with tempfile.TemporaryDirectory(prefix="kokoro_preview_") as work:
        work = Path(work)
        shutil.copyfile(bg_png, work / "bg.png")
        (work / "subs.ass").write_text(build_ass([(0.0, 5.0, subtitle)], size), encoding="utf-8")
        _run_ffmpeg(["-loop", "1", "-i", "bg.png", "-vf", "subtitles=subs.ass",
                     "-ss", "0.5", "-frames:v", "1", "preview.png"], work)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(work / "preview.png"), str(out_png))
    return out_png
