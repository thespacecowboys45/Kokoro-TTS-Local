"""
Translation review: English → Japanese with context and glossary, an isolated
English back-translation, heuristic checks, an LLM review, and at most two
automatic revision passes.

Back-translation is a diagnostic, not a certification. Nothing here scores a
translation or declares it correct.
"""

import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, Iterator, List, Optional, Tuple

from .translation import TranslationError, back_translate

OLLAMA_URL = os.environ.get("KOKORO_REVIEW_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
REVIEW_MODEL = os.environ.get("KOKORO_REVIEW_MODEL", "qwen2.5:7b")
MAX_REVISION_PASSES = 2
# None = unlimited (dxb, 2026-10-12: long jobs may run as long as they need).
REQUEST_TIMEOUT = None        # was 300 s
MAX_REVIEW_CHARS = None       # was 5000
# Ollama's default context window is a few thousand tokens and silently drops
# the start of longer prompts, so it is sized to each prompt (qwen2.5: 32k max).
MIN_CONTEXT_TOKENS = 4096
MAX_CONTEXT_TOKENS = 32768

CATEGORIES = [
    "omitted", "added", "changed_fact_or_number", "changed_negation_or_certainty",
    "changed_actor_or_recipient", "changed_tone_or_intent", "terminology", "ambiguity",
]
CATEGORY_LABELS = {
    "omitted": "Omitted meaning",
    "added": "Added meaning",
    "changed_fact_or_number": "Changed fact or number",
    "changed_negation_or_certainty": "Changed negation or certainty",
    "changed_actor_or_recipient": "Changed actor or recipient",
    "changed_tone_or_intent": "Changed tone or intent",
    "terminology": "Terminology",
    "ambiguity": "Ambiguity",
}

NO_MEANING_DIFFERENCES = (
    "The reviewer found no meaning differences. This is diagnostic only. "
    "Check the Japanese yourself before publishing."
)

TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {
        "translation": {"type": "string"},
        "ambiguities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_span": {"type": "string"},
                    "readings": {"type": "array", "items": {"type": "string"}},
                    "reading_used": {"type": "string"},
                    "clarification_needed": {"type": "string"},
                },
                "required": ["source_span", "readings", "reading_used", "clarification_needed"],
            },
        },
    },
    "required": ["translation", "ambiguities"],
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": CATEGORIES},
                    "kind": {"type": "string", "enum": ["meaning", "wording"]},
                    "source_span": {"type": "string"},
                    "japanese_span": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["category", "kind", "source_span", "japanese_span", "explanation"],
            },
        },
        "revised_translation": {"type": "string"},
        "revision_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["findings", "revised_translation", "revision_notes"],
}

PRESERVATION_RULES = """\
Preserve exactly:
- facts, names, numbers, units, dates, and negation;
- who performs each action and who receives it;
- conditions, uncertainty, obligations, and prohibitions;
- questions, requests, emphasis, and intended tone;
- every sentence, in order, including sentences that repeat earlier ones.
  Repetition is deliberate: never merge, shorten, or drop a repeated sentence.
  The Japanese must have the same number of sentences as the source.
Do not add claims, explanations, or marketing promises. Do not drop anything."""

TRANSLATE_SYSTEM = f"""\
You are a professional English-to-Japanese translator. Translate the SOURCE
into natural Japanese suited to the CONTEXT (audience, intended meaning, tone,
pronunciation, terminology), if one is given.

{PRESERVATION_RULES}

GLOSSARY: when an English glossary term appears in the source, use exactly its
listed Japanese translation. Do not invent glossary entries; terms that are not
listed are translated normally.

AMBIGUITY: if any part of the source can reasonably be read in more than one
way, do not silently pick one. List it in "ambiguities" with the source span,
each possible reading, the reading you used in the translation, and the
clarification the author must provide. If nothing is ambiguous, return an
empty list.

Return only JSON matching the schema."""

REVIEW_SYSTEM = f"""\
You review an English-to-Japanese translation. You are given the ORIGINAL
English, the JAPANESE translation, and an English BACK-TRANSLATION of the
Japanese made by a different system that never saw the original. The
back-translation is only a diagnostic: its wording may differ harmlessly and it
can contain its own errors, so also compare the Japanese directly with the
original.

Report each concrete difference as a finding:
- category: one of {", ".join(CATEGORIES)};
- kind: "meaning" if the Japanese loses, adds, or changes meaning, or breaks
  the glossary; "wording" if it is only a harmless phrasing difference;
- source_span, japanese_span, and a short explanation.
Look specifically for omitted meaning, added meaning, changed facts or numbers,
changed negation or certainty, changed actor or recipient, changed tone or
intent, terminology/glossary problems, and ambiguity in the source.
The AUTOMATED CHECKS are heuristics: confirm a warning with a finding only if
it is a real problem.

The translation must follow these rules:
{PRESERVATION_RULES}

If there is at least one "meaning" finding, set "revised_translation" to the
complete corrected Japanese text, fixing only the meaning problems and
following the glossary, and list in "revision_notes" what each change fixes.
Otherwise set "revised_translation" to "" and "revision_notes" to [].

Never give scores, percentages, or confidence levels. Never describe a
translation as exact, validated, verified, certified, or perfect.
Return only JSON matching the schema."""


class ReviewError(TranslationError):
    """A review failure whose message is safe to show in the UI."""


# ---------------------------------------------------------------- LLM access

def _context_size(prompt: str) -> int:
    """Context window large enough for the prompt plus an answer of similar size.

    Rough token estimate: about one token per Japanese character and one per
    four Latin characters.
    """
    cjk = sum(1 for ch in prompt if ord(ch) > 0x2E7F)
    estimate = cjk + (len(prompt) - cjk) / 4
    needed = int(estimate * 2.5) + 1024  # prompt + answer + headroom
    if needed > MAX_CONTEXT_TOKENS:
        # The model's hard maximum: Ollama would silently drop the start.
        raise ReviewError(
            f"This text is too long for the review model's {MAX_CONTEXT_TOKENS:,}-token "
            "context (about 12,000 Japanese characters per review). Review it in parts.")
    size = MIN_CONTEXT_TOKENS
    while size < needed and size < MAX_CONTEXT_TOKENS:
        size *= 2
    return min(size, MAX_CONTEXT_TOKENS)


def _chat(system: str, user: str, schema: dict) -> dict:
    payload = json.dumps({
        "model": REVIEW_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "format": schema,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": _context_size(system + user),
                    "num_predict": -1},
    }).encode("utf-8")
    last_problem = None
    for _ in range(2):  # one retry for malformed output
        request = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code == 404 and "not found" in detail:
                raise ReviewError(
                    f"Review model {REVIEW_MODEL!r} is not installed. "
                    f"Run: ollama pull {REVIEW_MODEL}"
                ) from e
            raise ReviewError(f"Ollama returned HTTP {e.code}: {detail[:200]}") from e
        except (urllib.error.URLError, ConnectionError) as e:
            raise ReviewError(
                f"Ollama is not reachable at {OLLAMA_URL}. Start Ollama and try again."
            ) from e
        except TimeoutError as e:
            raise ReviewError(
                f"The review model did not answer within {REQUEST_TIMEOUT} s."
            ) from e
        try:
            content = json.loads(body["message"]["content"])
            missing = [k for k in schema["required"] if k not in content]
            if not missing:
                return content
            last_problem = f"missing fields {missing}"
        except (KeyError, TypeError, json.JSONDecodeError) as e:
            last_problem = f"{type(e).__name__}: {e}"
    raise ReviewError(f"The review model returned unusable output ({last_problem}).")


# ---------------------------------------------------------------- glossary

def parse_glossary(text: str) -> List[Tuple[str, str]]:
    """Parse 'English term = 日本語' lines; malformed lines are errors, never guesses."""
    entries = []
    for number, raw in enumerate((text or "").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("=") != 1:
            raise ReviewError(
                f"Glossary line {number} is not 'English term = 日本語訳': {raw!r}"
            )
        english, japanese = (part.strip() for part in line.split("="))
        if not english or not japanese:
            raise ReviewError(f"Glossary line {number} has an empty side: {raw!r}")
        entries.append((english, japanese))
    return entries


def _glossary_block(glossary) -> str:
    if not glossary:
        return "(none)"
    return "\n".join(f"- {en} => {ja}" for en, ja in glossary)


# ---------------------------------------------------------------- pipeline steps

def translate_with_context(source: str, context: str, glossary) -> dict:
    user = (f"CONTEXT:\n{(context or '').strip() or '(none)'}\n\n"
            f"GLOSSARY:\n{_glossary_block(glossary)}\n\n"
            f"SOURCE:\n{source}")
    result = _chat(TRANSLATE_SYSTEM, user, TRANSLATE_SCHEMA)
    if not str(result.get("translation", "")).strip():
        raise ReviewError("The translation model returned an empty translation.")
    return result


def back_translate_isolated(japanese: str, glossary) -> Tuple[str, List[str]]:
    """Back-translate with an independent model. The original English is never
    an input here; glossary pairs are only reported alongside the output."""
    english = back_translate(japanese)
    notes = [f"{ja} → {en}" for en, ja in glossary if ja in japanese]
    return english, notes


# ---------------------------------------------------------------- heuristic checks

_FULLWIDTH = str.maketrans("０１２３４５６７８９．，", "0123456789.,")
_KANJI_DIGITS = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
                 "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_KANJI_UNITS = {"十": 10, "百": 100, "千": 1000}
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _kanji_to_int(run: str) -> Optional[int]:
    total, digit = 0, None
    for ch in run:
        if ch in _KANJI_DIGITS:
            digit = (digit or 0) * 10 + _KANJI_DIGITS[ch]
        elif ch in _KANJI_UNITS:
            total += (digit if digit is not None else 1) * _KANJI_UNITS[ch]
            digit = None
        else:
            return None
    return total + (digit or 0)


def _numbers(text: str) -> List[str]:
    norm = (text or "").translate(_FULLWIDTH)
    norm = re.sub(r"(?<=\d),(?=\d{3}\b)", "", norm)
    found = _NUMBER.findall(norm)
    for run in re.findall(r"[〇零一二三四五六七八九十百千]+", norm):
        value = _kanji_to_int(run)
        if value is not None:
            found.append(str(value))
    return [n.rstrip("0").rstrip(".") if "." in n else n for n in found]


_UNITS = [
    (r"\bml\b|\bmillilit(?:re|er)s?\b", ["ml", "mL", "ｍｌ", "ミリリットル"]),
    (r"\blit(?:re|er)s?\b|\b\d+\s?l\b", ["L", "l", "ℓ", "リットル"]),
    (r"\bkg\b|\bkilograms?\b", ["kg", "キログラム", "キロ"]),
    (r"\bgrams?\b|\b\d+\s?g\b", ["g", "グラム"]),
    (r"\bcm\b|\bcentimet(?:re|er)s?\b", ["cm", "センチ"]),
    (r"\bmm\b|\bmillimet(?:re|er)s?\b", ["mm", "ミリ"]),
    (r"\bhours?\b", ["時間"]),
    (r"\bminutes?\b", ["分"]),
    (r"\bdays?\b", ["日"]),
    (r"\bweeks?\b", ["週"]),
    (r"\bmonths?\b", ["月", "ヶ月", "か月", "カ月"]),
    (r"°c|\bdegrees?\b", ["℃", "度"]),
    (r"%|\bper ?cent\b", ["%", "％", "パーセント"]),
]
_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]
_NEGATION_EN = re.compile(
    r"\b(?:not|never|no|none|nothing|nobody|neither|nor|without|cannot)\b|n't\b", re.I)
_NEGATION_JA = re.compile(r"ない|ません|ず|禁止|いけない|だめ|ダメ|不可|なし|無し|ぬ|な[。！!]|べからず")
_PROHIBITION_EN = re.compile(r"(?:^|[.!?]\s+)(?:do not|don't|never)\b", re.I)
_PROHIBITION_JA = re.compile(
    r"ないでください|ないで下さい|ないでね|ないこと|禁止|べからず|ないように|"
    r"てはいけない|てはならない|ではいけない|ご遠慮|な[。！!]")
_BRAND = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[ -][A-Z][A-Za-z0-9]*)*")


def _sentences_en(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def _sentences_ja(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[。！？!?])", (text or "").strip()) if s.strip()]


def _check(name: str, ok: bool, detail: str) -> Dict[str, str]:
    return {"check": name, "status": "ok" if ok else "warn", "detail": detail}


def deterministic_checks(source: str, japanese: str, back: str, glossary) -> List[Dict[str, str]]:
    """Heuristic checks that need no model. Evidence, not a verdict."""
    results = []
    src_lower = source.lower()

    src_numbers = _numbers(source)
    if src_numbers:
        ja_numbers, back_numbers = set(_numbers(japanese)), set(_numbers(back))
        missing_ja = [n for n in src_numbers if n not in ja_numbers]
        missing_back = [n for n in src_numbers if n not in back_numbers]
        detail = (f"Source numbers {src_numbers}. "
                  + (f"Missing in Japanese: {missing_ja}. " if missing_ja else "All present in Japanese. ")
                  + (f"Missing in back-translation: {missing_back}." if missing_back else ""))
        results.append(_check("Numbers", not missing_ja and not missing_back, detail.strip()))

    for pattern, forms in _UNITS:
        match = re.search(pattern, source, re.I)
        if match:
            present = any(f in japanese for f in forms)
            results.append(_check(f"Unit '{match.group(0).strip()}'", present,
                                  "Found in Japanese." if present
                                  else f"No Japanese form of it found (looked for {forms})."))
    for index, month in enumerate(_MONTHS, start=1):
        if re.search(rf"\b{month}\b", src_lower):
            present = f"{index}月" in japanese.translate(_FULLWIDTH)
            results.append(_check(f"Date '{month.title()}'", present,
                                  f"{index}月 found." if present else f"{index}月 not found in Japanese."))

    for english, ja in glossary:
        if english.lower() in src_lower:
            present = ja in japanese
            results.append(_check(f"Glossary '{english}'", present,
                                  f"Uses {ja}." if present else f"Required term {ja} is missing."))

    glossary_en = {en.lower(): ja for en, ja in glossary}
    sentence_starts = {m.end() for m in re.finditer(r"(?:^|[.!?]\s+)", source)}
    for match in _BRAND.finditer(source):
        name = match.group(0)
        if match.start() in sentence_starts:
            # Drop the ordinary capitalised sentence-initial word ("Apply", "Can").
            name = " ".join(re.split(r"[ -]", name)[1:])
        if not name or name == "I":
            continue
        expected = glossary_en.get(name.lower())
        present = name in japanese or (expected is not None and expected in japanese)
        results.append(_check(f"Name '{name}'", present,
                              "Kept." if present else
                              "Not found verbatim" + (f" or as {expected}." if expected else
                                                      " (add a glossary entry if a katakana form is intended).")))

    src_question = "?" in source
    ja_question = bool(re.search(r"[？?]|か[。\s]*$|か。", japanese.strip()))
    results.append(_check("Question vs statement", src_question == ja_question,
                          ("Source is a question; " if src_question else "Source is a statement; ")
                          + ("Japanese matches." if src_question == ja_question
                             else "Japanese does not match.")))

    if _NEGATION_EN.search(source):
        ja_neg = bool(_NEGATION_JA.search(japanese))
        back_neg = bool(_NEGATION_EN.search(back))
        results.append(_check("Negation", ja_neg and back_neg,
                              "Source contains negation; "
                              + ("Japanese has a negation/prohibition form; " if ja_neg
                                 else "no negation/prohibition form found in Japanese; ")
                              + ("back-translation keeps it." if back_neg
                                 else "back-translation has no negation.")))

    src_sentences = _sentences_en(source)
    ja_count, back_count = len(_sentences_ja(japanese)), len(_sentences_en(back))
    normalized = [re.sub(r"[^a-z0-9]+", " ", s.lower()).strip() for s in src_sentences]
    repeats = len(normalized) - len(set(normalized))
    counts_match = len(src_sentences) == ja_count == back_count
    results.append(_check(
        "Sentence count", counts_match,
        f"Source {len(src_sentences)}, Japanese {ja_count}, back-translation {back_count}"
        + (f" (the source repeats {repeats} sentence(s) on purpose)" if repeats else "")
        + ("." if counts_match else ". Sentences may have been dropped, merged, or split.")))

    if _PROHIBITION_EN.search(source):
        found = bool(_PROHIBITION_JA.search(japanese))
        results.append(_check("Prohibition", found,
                              "Source is a prohibition (\"Do not…\"); "
                              + ("Japanese uses a prohibition form." if found
                                 else "Japanese has no prohibition form (e.g. しないでください); "
                                      "a plain negative can read as a statement.")))
    return results


# ---------------------------------------------------------------- review & revision

def review(source: str, japanese: str, back: str, context: str, glossary, checks) -> dict:
    check_lines = "\n".join(f"- [{c['status']}] {c['check']}: {c['detail']}" for c in checks) or "(none)"
    user = (f"ORIGINAL (English):\n{source}\n\n"
            f"JAPANESE:\n{japanese}\n\n"
            f"BACK-TRANSLATION (English, independent system):\n{back}\n\n"
            f"CONTEXT:\n{(context or '').strip() or '(none)'}\n\n"
            f"GLOSSARY:\n{_glossary_block(glossary)}\n\n"
            f"AUTOMATED CHECKS (heuristic):\n{check_lines}")
    result = _chat(REVIEW_SYSTEM, user, REVIEW_SCHEMA)
    findings = [f for f in result.get("findings", [])
                if f.get("category") in CATEGORIES and f.get("kind") in ("meaning", "wording")]
    result["findings"] = findings
    return result


def _meaning(findings) -> list:
    return [f for f in findings if f["kind"] == "meaning"]


def _assess(source, japanese, context, glossary) -> dict:
    back, terms = back_translate_isolated(japanese, glossary)
    checks = deterministic_checks(source, japanese, back, glossary)
    result = review(source, japanese, back, context, glossary, checks)
    return {"japanese": japanese, "back": back, "terms": terms, "checks": checks,
            "findings": result["findings"],
            "revised_translation": (result.get("revised_translation") or "").strip(),
            "revision_notes": result.get("revision_notes") or []}


def run_review(source: str, japanese: str, context: str, glossary) -> Iterator[tuple]:
    """Review `japanese` unrevised, then revise at most MAX_REVISION_PASSES times.

    Yields ("progress", message) events and finally ("result", dict) with
    "unrevised" (the given text, never modified) and "candidate" (or None).
    """
    if not source or not source.strip():
        raise ReviewError("The original English text is empty; there is nothing to compare against.")
    if not japanese or not japanese.strip():
        raise ReviewError("The Japanese text is empty; translate or type it first.")
    if MAX_REVIEW_CHARS is not None and (len(source) > MAX_REVIEW_CHARS
                                         or len(japanese) > MAX_REVIEW_CHARS):
        raise ReviewError(f"Text is too long to review (limit {MAX_REVIEW_CHARS} characters).")

    yield "progress", "Back-translating and reviewing the unrevised Japanese..."
    unrevised = _assess(source, japanese, context, glossary)
    candidate = None

    if _meaning(unrevised["findings"]):
        passes = []
        current = unrevised
        stop_reason = ""
        for number in range(1, MAX_REVISION_PASSES + 1):
            revised = current["revised_translation"]
            if not revised:
                stop_reason = "The reviewer reported meaning differences but offered no revision."
                break
            if revised == current["japanese"]:
                stop_reason = "The reviewer's revision was identical to the text it reviewed."
                break
            yield "progress", f"Revision pass {number} of {MAX_REVISION_PASSES}: back-translating and reviewing..."
            assessed = _assess(source, revised, context, glossary)
            passes.append({"pass": number, "japanese": revised,
                           "what_it_fixes": current["revision_notes"]})
            current = assessed
            if not _meaning(current["findings"]):
                break
        candidate = {"passes": passes, "final": current if passes else None,
                     "unresolved": _meaning(current["findings"]),
                     "stop_reason": stop_reason}

    yield "result", {"unrevised": unrevised, "candidate": candidate}


# ---------------------------------------------------------------- display

_BANNED = re.compile(r"\b(?:exact(?:ly)?|validated?|verified|certif\w*|perfect(?:ly)?)\b", re.I)
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s*[%％]")


def _safe(text: str) -> str:
    """Strip certification words and percentages from model-written prose."""
    return _PERCENT.sub("[number removed]", _BANNED.sub("[word removed]", str(text or ""))).strip()


def _md(text: str) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ")


def format_ambiguities_md(ambiguities) -> str:
    if not ambiguities:
        return ""
    lines = ["**Ambiguity flagged in the source (clarification needed):**"]
    for a in ambiguities:
        readings = "; ".join(_md(r) for r in a.get("readings", []))
        lines.append(f"- \"{_md(a.get('source_span'))}\" can mean: {readings}. "
                     f"Translated as: {_md(_safe(a.get('reading_used')))}. "
                     f"**Needed:** {_md(_safe(a.get('clarification_needed')))}")
    return "\n".join(lines)


def format_assessment_md(assessment: dict) -> str:
    lines = []
    if assessment.get("terms"):
        lines.append("**Glossary terms in the Japanese:** " + ", ".join(assessment["terms"]))
    lines.append("**Automated checks (heuristic):**")
    for c in assessment["checks"]:
        mark = "✓" if c["status"] == "ok" else "⚠"
        lines.append(f"- {mark} {c['check']}: {_md(c['detail'])}")
    meaning = [f for f in assessment["findings"] if f["kind"] == "meaning"]
    wording = [f for f in assessment["findings"] if f["kind"] == "wording"]
    lines.append("**Reviewer findings: meaning changes**")
    if meaning:
        for f in meaning:
            lines.append(f"- **{CATEGORY_LABELS[f['category']]}**: \"{_md(f['source_span'])}\" → "
                         f"「{_md(f['japanese_span'])}」. {_md(_safe(f['explanation']))}")
    else:
        lines.append(f"- {NO_MEANING_DIFFERENCES}")
    lines.append("**Reviewer findings: wording only (harmless)**")
    if wording:
        for f in wording:
            lines.append(f"- {CATEGORY_LABELS[f['category']]}: \"{_md(f['source_span'])}\" → "
                         f"「{_md(f['japanese_span'])}」. {_md(_safe(f['explanation']))}")
    else:
        lines.append("- None reported.")
    return "\n".join(lines)


def format_candidate_md(candidate: Optional[dict]) -> str:
    if candidate is None:
        return ("**Revision passes: 0 of max "
                f"{MAX_REVISION_PASSES}.** No meaning differences were reported, so nothing was revised.")
    passes = candidate["passes"]
    lines = [f"**Revision passes: {len(passes)} of max {MAX_REVISION_PASSES}.**"]
    for p in passes:
        fixes = "; ".join(_md(_safe(n)) for n in p["what_it_fixes"]) or "(no notes given)"
        lines.append(f"- Pass {p['pass']} fixes: {fixes}")
    if candidate["stop_reason"]:
        lines.append(f"- Stopped: {candidate['stop_reason']}")
    if candidate["final"] is not None:
        lines.append("")
        lines.append(format_assessment_md(candidate["final"]))
    if candidate["unresolved"]:
        lines.append("")
        lines.append("**Unresolved, needs your review:**")
        for f in candidate["unresolved"]:
            lines.append(f"- {CATEGORY_LABELS[f['category']]}: \"{_md(f['source_span'])}\". "
                         f"{_md(_safe(f['explanation']))}")
    return "\n".join(lines)
