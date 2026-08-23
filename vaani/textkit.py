"""Script-aware text processing primitives, standard library only.

Every other module in Vaani tokenises and splits text through this file, which
matters more than it sounds: if the chunker, the embedder, the BM25 index and
the grounding guard disagree about what a "token" is, then chunk offsets drift,
IDF weights go wrong, and grounding scores become meaningless. One definition,
used everywhere.

The corpus is MSMARCO-XI, which is Indic-multilingual, so none of this can
assume Latin script or space-delimited words:

* ``tokenize`` classifies each character by Unicode script and emits runs, so
  Devanagari, Bengali, Tamil, Telugu, CJK and Latin all tokenise sensibly
  without a per-language model. CJK has no word spaces, so those runs are
  emitted as individual codepoints plus bigrams.
* ``split_sentences`` knows the Devanagari danda (।) and double danda (॥), the
  Arabic question mark, and CJK full stops -- not just ``.!?``.
* ``normalize`` folds Indic digits to ASCII and strips the zero-width joiners
  that speech-to-text output is full of, so "८" and "8" match.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# --------------------------------------------------------------------------- #
# Unicode ranges we care about. Kept as explicit tuples rather than regex
# character classes because we test codepoints in a hot loop.
# --------------------------------------------------------------------------- #

_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0041, 0x005A, "latin"),
    (0x0061, 0x007A, "latin"),
    (0x00C0, 0x024F, "latin"),
    (0x0370, 0x03FF, "greek"),
    (0x0400, 0x04FF, "cyrillic"),
    (0x0590, 0x05FF, "hebrew"),
    (0x0600, 0x06FF, "arabic"),
    (0x0900, 0x097F, "devanagari"),  # Hindi, Marathi, Sanskrit, Nepali
    (0x0980, 0x09FF, "bengali"),  # Bengali, Assamese
    (0x0A00, 0x0A7F, "gurmukhi"),  # Punjabi
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "oriya"),
    (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
    (0x0D80, 0x0DFF, "sinhala"),
    (0x0E00, 0x0E7F, "thai"),
    (0x3040, 0x30FF, "kana"),
    (0x4E00, 0x9FFF, "han"),
    (0xAC00, 0xD7AF, "hangul"),
)

# Scripts with no inter-word spacing: we cannot rely on whitespace to find word
# boundaries, so we index characters and character bigrams instead.
_NO_SPACE_SCRIPTS = frozenset({"han", "kana", "hangul", "thai"})

# Indic scripts, used to decide whether to apply danda sentence splitting and
# digit folding.
INDIC_SCRIPTS = frozenset(
    {
        "devanagari",
        "bengali",
        "gurmukhi",
        "gujarati",
        "oriya",
        "tamil",
        "telugu",
        "kannada",
        "malayalam",
        "sinhala",
    }
)

# Map each script to the ISO-639-1 code we tag chunks with. Several scripts are
# shared across languages (Devanagari covers hi/mr/ne), so this is a best-effort
# default that the dataset's own language field overrides when present.
_SCRIPT_TO_LANG = {
    "devanagari": "hi",
    "bengali": "bn",
    "gurmukhi": "pa",
    "gujarati": "gu",
    "oriya": "or",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
    "sinhala": "si",
    "arabic": "ar",
    "cyrillic": "ru",
    "greek": "el",
    "hebrew": "he",
    "han": "zh",
    "kana": "ja",
    "hangul": "ko",
    "thai": "th",
    "latin": "en",
}


_SCRIPT_CACHE: dict[str, str] = {}


def script_of(ch: str) -> str:
    """Unicode script name for one character, or "digit" / "other".

    Punctuation is resolved by Unicode *category* before script range, because
    several scripts place their punctuation inside their own block -- most
    importantly the Devanagari danda U+0964 (।) and double danda U+0965 (॥),
    which are sentence terminators sitting in the middle of the Devanagari
    letter range. A pure range lookup would glue them onto the preceding word
    and produce tokens like "है।" that never match the corpus.

    Combining marks (category M*) are deliberately *not* excluded: Indic vowel
    signs and viramas are part of the word, and they live in their script's
    block, so the range lookup keeps them attached where they belong.
    """
    cached = _SCRIPT_CACHE.get(ch)
    if cached is not None:
        return cached

    cp = ord(ch)
    if cp < 0x0041:  # fast path over ASCII space, punctuation and digits
        res = "digit" if 0x0030 <= cp <= 0x0039 else "other"
    else:
        cat = unicodedata.category(ch)
        if cat[0] in "PZSC":  # punctuation, separator, symbol, control
            res = "other"
        elif cat == "Nd":
            res = "digit"
        else:
            res = "other"
            for lo, hi, name in _SCRIPT_RANGES:
                if lo <= cp <= hi:
                    res = name
                    break

    _SCRIPT_CACHE[ch] = res
    return res


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

# Indic digit blocks -> ASCII. Speech-to-text and the corpus disagree about
# which form to use, so we fold to ASCII everywhere.
_DIGIT_FOLD: dict[int, str] = {}
for _base in (
    0x0966,  # Devanagari
    0x09E6,  # Bengali
    0x0A66,  # Gurmukhi
    0x0AE6,  # Gujarati
    0x0B66,  # Oriya
    0x0BE6,  # Tamil
    0x0C66,  # Telugu
    0x0CE6,  # Kannada
    0x0D66,  # Malayalam
    0x0660,  # Arabic-Indic
    0x06F0,  # Extended Arabic-Indic
):
    for _d in range(10):
        _DIGIT_FOLD[_base + _d] = str(_d)

# Zero-width and formatting characters that appear constantly in Indic text and
# in ASR output, and that would otherwise split tokens.
_ZERO_WIDTH = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0xFEFF, 0x00AD, 0x2060], ""
)

# Typographic characters folded to their ASCII equivalents so that a quoted
# passage and a spoken query agree.
_PUNCT_FOLD = {
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"',
    0x2013: "-", 0x2014: "-", 0x2015: "-", 0x2212: "-",
    0x2026: "...", 0x00A0: " ", 0x202F: " ", 0x2009: " ",
    0x0060: "'", 0x00B4: "'",
}

_TRANSLATE = {**_DIGIT_FOLD, **_ZERO_WIDTH, **_PUNCT_FOLD}

_WS_RE = re.compile(r"\s+")


def normalize(text: str, *, fold_case: bool = True) -> str:
    """Canonical form used for indexing and matching.

    NFKC first so that composed and decomposed Indic sequences unify, then digit
    folding, zero-width stripping and whitespace collapse.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = out.translate(_TRANSLATE)
    out = _WS_RE.sub(" ", out).strip()
    return out.casefold() if fold_case else out


def strip_accents(text: str) -> str:
    """Remove combining marks. Latin only -- Indic vowel signs are *not* accents
    and removing them destroys meaning, so we only touch characters whose base
    is Latin."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.combining(ch):
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #

# Characters that may appear *inside* a Latin token.
_INWORD = frozenset("'-_.@/+")


def tokenize(text: str, *, normalized: bool = False) -> list[str]:
    """Split text into comparable tokens across scripts.

    Latin/Cyrillic/Greek and Indic runs are split on non-letter boundaries.
    Space-less scripts (Han, Kana, Hangul, Thai) emit single characters plus
    adjacent bigrams, which is the standard trick for making BM25 work on CJK
    without a segmenter.
    """
    if not text:
        return []
    src = text if normalized else normalize(text)
    tokens: list[str] = []
    buf: list[str] = []
    buf_script = ""

    def flush() -> None:
        nonlocal buf, buf_script
        if not buf:
            return
        word = "".join(buf)
        if buf_script in _NO_SPACE_SCRIPTS:
            # Unigrams plus bigrams: bigrams carry most of the signal.
            tokens.extend(word)
            tokens.extend(word[i : i + 2] for i in range(len(word) - 1))
        else:
            word = word.strip("".join(_INWORD))
            if word:
                tokens.append(word)
        buf = []
        buf_script = ""

    for ch in src:
        sc = script_of(ch)
        if sc == "other":
            # In-word punctuation only counts if we are mid-token.
            if ch in _INWORD and buf:
                buf.append(ch)
            else:
                flush()
            continue
        if sc == "digit":
            if buf_script and buf_script not in ("digit", "latin"):
                flush()
            buf.append(ch)
            buf_script = buf_script or "digit"
            continue
        if buf_script and sc != buf_script and not (
            {sc, buf_script} <= {"latin", "digit"}
        ):
            flush()
        buf.append(ch)
        buf_script = sc
    flush()
    return tokens


def count_tokens(text: str) -> int:
    return len(tokenize(text))


def char_ngrams(token: str, n_min: int = 3, n_max: int = 5) -> list[str]:
    """Character n-grams of a token, boundary-marked.

    These give the retriever robustness to the two failure modes that dominate
    voice input: speech-to-text misspellings, and Indic morphology where the
    same stem appears with many different case suffixes.
    """
    padded = f"^{token}$"
    out: list[str] = []
    L = len(padded)
    for n in range(n_min, n_max + 1):
        if L < n:
            break
        out.extend(padded[i : i + n] for i in range(L - n + 1))
    return out


# --------------------------------------------------------------------------- #
# Stopwords. Small, hand-picked, multilingual. Deliberately conservative: an
# over-broad stoplist destroys short-question retrieval ("what is a cell wall"
# is almost entirely stopwords).
# --------------------------------------------------------------------------- #

STOPWORDS: frozenset[str] = frozenset(
    """
a an the and or but if then than that this these those of in on at to for from by with
as is are was were be been being do does did doing have has had having i you he she it
we they them his her its their our your my me him us not no nor so too very can will
just should now what which who whom when where why how all any both each few more most
other some such only own same s t don didn
""".split()
    + """
और का की के को है हैं था थे थी में से पर यह वह ये वे कि जो तो ही भी नहीं एक हो होता होती
क्या कैसे कब कहाँ क्यों कौन किस अपने लिए साथ तक बाद पहले
""".split()
    + """
এবং এর এই সেই যে না হয় হয়েছে ছিল করে জন্য থেকে সঙ্গে কি কেন কিভাবে কখন কোথায় একটি
""".split()
    + """
மற்றும் இந்த அந்த ஒரு என்று இல்லை உள்ளது ஆகிறது என்ன எப்படி எப்போது எங்கே ஏன்
""".split()
    + """
మరియు ఈ ఆ ఒక అని లేదు ఉంది అవుతుంది ఏమిటి ఎలా ఎప్పుడు ఎక్కడ ఎందుకు
""".split()
)

# Interrogatives must never be dropped by the stoplist even though they are
# high-frequency: they are what tells the generator which answer shape to use.
QUESTION_WORDS: frozenset[str] = frozenset(
    """
what which who whom whose when where why how is are do does did can could should would
will define explain list name many much long far
क्या कैसे कब कहाँ क्यों कौन कितना कितने किस
কি কেন কিভাবে কখন কোথায় কোন কত
என்ன எப்படி எப்போது எங்கே ஏன் எவ்வளவு
ఏమిటి ఎలా ఎప్పుడు ఎక్కడ ఎందుకు ఎంత
""".split()
)


def content_tokens(text: str, *, keep_question_words: bool = False) -> list[str]:
    """Tokens with stopwords removed. Used for IDF and for grounding overlap."""
    toks = tokenize(text)
    if keep_question_words:
        return [t for t in toks if t not in STOPWORDS or t in QUESTION_WORDS]
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]


# --------------------------------------------------------------------------- #
# Sentence segmentation
# --------------------------------------------------------------------------- #

# Terminators across the scripts we support. The danda (U+0964) is the sentence
# end in Devanagari/Bengali and is the single most important non-ASCII case.
_TERMINATORS = ".!?।॥؟۔。！？⁉‽;"

# Abbreviations after which a period is not a sentence break.
_ABBREV = frozenset(
    """
mr mrs ms dr prof sr jr st rev hon capt sgt lt col gen adm gov pres
vs etc eg ie al fig no vol ch pp ed eds repr trans approx est
inc ltd co corp dept univ inst assn bros
jan feb mar apr jun jul aug sep sept oct nov dec
mon tue wed thu fri sat sun
u.s u.k u.n a.m p.m i.e e.g
""".split()
)

_SENT_SPLIT_RE = re.compile(rf"([{re.escape(_TERMINATORS)}]+)")


def split_sentences(text: str, *, min_chars: int = 2) -> list[tuple[str, int, int]]:
    """Split into sentences, returning ``(sentence, start, end)`` char offsets.

    Offsets are into the *original* string, which is what lets a citation point
    at an exact span the UI can highlight. This is why we do not normalise
    before splitting.
    """
    if not text or not text.strip():
        return []

    out: list[tuple[str, int, int]] = []
    pos = 0
    pending_start = 0
    parts = _SENT_SPLIT_RE.split(text)

    buf = ""
    for i, part in enumerate(parts):
        if not part:
            continue
        is_term = bool(i % 2)
        if not buf:
            pending_start = pos
        buf += part
        pos += len(part)

        if not is_term:
            continue

        # Decide whether this terminator really ends a sentence.
        stripped = buf.rstrip()
        if stripped.endswith("."):
            last = stripped[:-1].rsplit(None, 1)
            tail = (last[-1] if last else "").casefold().rstrip(".")
            # "Dr." / "e.g." / a single initial like "J." are not breaks.
            if tail in _ABBREV or (len(tail) == 1 and tail.isalpha()):
                continue
            # A decimal point inside a number is not a break: "3.14"
            nxt = parts[i + 1] if i + 1 < len(parts) else ""
            if tail[-1:].isdigit() and nxt[:1].isdigit():
                continue

        sent = buf.strip()
        if len(sent) >= min_chars:
            lead = len(buf) - len(buf.lstrip())
            start = pending_start + lead
            out.append((sent, start, start + len(sent)))
        buf = ""

    if buf.strip():
        sent = buf.strip()
        if len(sent) >= min_chars:
            lead = len(buf) - len(buf.lstrip())
            start = pending_start + lead
            out.append((sent, start, start + len(sent)))

    # Nothing terminated? Treat the whole input as one sentence rather than
    # returning nothing, so a terminator-free passage still gets indexed.
    if not out and text.strip():
        s = text.strip()
        lead = len(text) - len(text.lstrip())
        out.append((s, lead, lead + len(s)))
    return out


def split_paragraphs(text: str) -> list[tuple[str, int, int]]:
    """Blank-line delimited blocks with offsets."""
    out: list[tuple[str, int, int]] = []
    for m in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text):
        block = m.group(0).strip()
        if block:
            lead = len(m.group(0)) - len(m.group(0).lstrip())
            out.append((block, m.start() + lead, m.start() + lead + len(block)))
    return out


# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #


def detect_lang(text: str, default: str = "en") -> str:
    """Guess a language code from script distribution.

    Script identity is a strong signal for Indic languages and useless for
    distinguishing languages that share a script (Hindi vs Marathi), which is
    fine: we use it to route tokenisation and to tag chunks, and the dataset's
    own label wins when it exists.
    """
    if not text:
        return default
    counts: dict[str, int] = {}
    for ch in text:
        sc = script_of(ch)
        if sc in ("other", "digit"):
            continue
        counts[sc] = counts.get(sc, 0) + 1
    if not counts:
        return default
    top = max(counts, key=lambda k: counts[k])
    return _SCRIPT_TO_LANG.get(top, default)


def dominant_script(text: str) -> str:
    counts: dict[str, int] = {}
    for ch in text:
        sc = script_of(ch)
        if sc in ("other", "digit"):
            continue
        counts[sc] = counts.get(sc, 0) + 1
    return max(counts, key=lambda k: counts[k]) if counts else "latin"


# --------------------------------------------------------------------------- #
# Lightweight stemming. Not linguistically principled -- a suffix stripper that
# improves recall on English and on Indic case endings without a dictionary.
# --------------------------------------------------------------------------- #

_EN_SUFFIXES = (
    "ational", "tional", "iveness", "fulness", "ousness", "ization", "ation",
    "ments", "ement", "ingly", "edly", "ness", "ions", "ing", "ies", "ied",
    "ers", "est", "ed", "es", "ly", "s",
)

_HI_SUFFIXES = ("ओं", "याँ", "ियों", "ाओं", "ों", "ें", "ाँ", "ीं", "ता", "ना", "ने", "नी", "का", "की", "के", "है")


@lru_cache(maxsize=100_000)
def stem(token: str) -> str:
    """Conservative suffix stripper. Never shortens below 3 characters."""
    if len(token) <= 3:
        return token
    sc = script_of(token[0])
    suffixes = _HI_SUFFIXES if sc in INDIC_SCRIPTS else _EN_SUFFIXES
    for suf in suffixes:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            base = token[: -len(suf)]
            # English doubled consonant: "running" -> "runn" -> "run"
            if suffixes is _EN_SUFFIXES and len(base) > 3 and base[-1] == base[-2]:
                if base[-1] not in "lsz":
                    base = base[:-1]
            return base
    return token


def stems(text: str) -> list[str]:
    return [stem(t) for t in content_tokens(text)]


# --------------------------------------------------------------------------- #
# Query classification. Drives which answer template the generator uses.
# --------------------------------------------------------------------------- #

_QTYPE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("definition", ("what is", "what are", "define", "meaning of", "what does", "क्या है", "अर्थ", "কি", "என்றால் என்ன")),
    ("numeric", ("how many", "how much", "how long", "how far", "how old", "what year", "what percentage", "कितना", "कितने", "কত", "எவ்வளவு")),
    ("person", ("who is", "who was", "who are", "who invented", "who discovered", "who wrote", "कौन", "কে", "யார்")),
    ("location", ("where is", "where are", "where can", "where does", "कहाँ", "কোথায়", "எங்கே")),
    ("temporal", ("when is", "when was", "when did", "when does", "कब", "কখন", "எப்போது")),
    ("causal", ("why is", "why are", "why does", "why did", "reason for", "क्यों", "কেন", "ஏன்")),
    ("procedural", ("how do", "how to", "how does", "how can", "steps to", "process of", "कैसे", "কিভাবে", "எப்படி")),
    ("comparison", ("difference between", "versus", " vs ", "compare", "better than", "अंतर", "পার্থক্য")),
    ("listing", ("list of", "types of", "kinds of", "examples of", "name the", "प्रकार", "ধরন", "வகைகள்")),
    ("boolean", ("is it", "are there", "does it", "can you", "did the", "is the", "क्या यह")),
)


def classify_query(query: str) -> str:
    """Coarse question type. Returns "factual" when nothing matches."""
    q = f" {normalize(query)} "
    for qtype, pats in _QTYPE_PATTERNS:
        for p in pats:
            if p in q:
                return qtype
    if q.strip().split()[:1] and q.strip().split()[0] in {"what", "which"}:
        return "definition"
    return "factual"


# --------------------------------------------------------------------------- #
# Entity-ish extraction for the hallucination guard. No NER model available, so
# we take the high-precision surface cues: capitalised multiword spans, numbers
# with units, years, and quoted strings.
# --------------------------------------------------------------------------- #

_NUM_RE = re.compile(
    r"(?<![\w.])"
    r"(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"\s?(?:%|percent|per cent|million|billion|trillion|thousand|lakh|crore|"
    r"kg|km|cm|mm|m|ft|mi|lb|kb|mb|gb|tb|hz|khz|mhz|ghz|"
    r"°c|°f|celsius|fahrenheit|years?|months?|days?|hours?|minutes?|seconds?)?"
    r"(?![\w])",
    re.IGNORECASE,
)

_YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
_PROPER_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+(?:of|de|van|der|the|and)?\s*[A-Z][a-z]{2,})*\b")
_QUOTED_RE = re.compile(r'"([^"]{2,80})"')


def extract_numbers(text: str) -> list[str]:
    return [m.group(0).strip() for m in _NUM_RE.finditer(text)]


def extract_years(text: str) -> list[str]:
    return _YEAR_RE.findall(text)


def extract_entities(text: str) -> list[str]:
    """Best-effort entity surface forms."""
    out: list[str] = []
    seen: set[str] = set()
    for rx in (_PROPER_RE, _QUOTED_RE):
        for m in rx.finditer(text):
            val = (m.group(1) if rx is _QUOTED_RE else m.group(0)).strip()
            key = val.casefold()
            if len(val) > 2 and key not in seen and key not in STOPWORDS:
                seen.add(key)
                out.append(val)
    return out


def extract_claims(text: str) -> list[str]:
    """Verifiable atoms of an answer: numbers, years and entities together.

    The hallucination guard requires every one of these to be traceable to the
    retrieved context. Free-text prose can be paraphrased; a number cannot.
    """
    out: list[str] = []
    seen: set[str] = set()
    for val in extract_numbers(text) + extract_years(text) + extract_entities(text):
        k = val.casefold()
        if k not in seen:
            seen.add(k)
            out.append(val)
    return out


# --------------------------------------------------------------------------- #
# Similarity helpers used by dedup, MMR and grounding.
# --------------------------------------------------------------------------- #


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter) if inter else 0.0


def overlap_coefficient(a: set[str], b: set[str]) -> float:
    """Asymmetric containment: how much of ``a`` is covered by ``b``.

    This is the right measure for grounding. An answer is grounded when its
    tokens are contained in the context; the context being much larger than the
    answer should not be penalised, which is exactly what Jaccard would do.
    """
    if not a:
        return 0.0
    return len(a & b) / len(a)


def token_f1(pred: list[str], gold: list[str]) -> float:
    """Token-level F1, the standard MS MARCO / SQuAD answer overlap metric."""
    if not pred or not gold:
        return 0.0
    from collections import Counter

    common = Counter(pred) & Counter(gold)
    same = sum(common.values())
    if not same:
        return 0.0
    p = same / len(pred)
    r = same / len(gold)
    return 2 * p * r / (p + r)


def truncate(text: str, limit: int, ellipsis: str = "…") -> str:
    """Truncate on a word boundary."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    sp = cut.rfind(" ")
    if sp > limit * 0.6:
        cut = cut[:sp]
    return cut.rstrip(" ,;:.") + ellipsis


def highlight_spans(text: str, terms: set[str]) -> list[tuple[int, int]]:
    """Character spans in ``text`` matching any of ``terms`` (stem-aware).

    Returned to the browser so the UI can highlight exactly why a chunk matched
    rather than making the user guess.
    """
    if not terms:
        return []
    stemmed = {stem(t) for t in terms}
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"\w+", text, re.UNICODE):
        w = m.group(0).casefold()
        if w in terms or stem(w) in stemmed:
            spans.append((m.start(), m.end()))
    # Merge adjacent spans so "vector database" highlights as one run.
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s - merged[-1][1] <= 1:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


__all__ = [
    "INDIC_SCRIPTS",
    "QUESTION_WORDS",
    "STOPWORDS",
    "char_ngrams",
    "classify_query",
    "content_tokens",
    "count_tokens",
    "detect_lang",
    "dominant_script",
    "extract_claims",
    "extract_entities",
    "extract_numbers",
    "extract_years",
    "highlight_spans",
    "jaccard",
    "normalize",
    "overlap_coefficient",
    "script_of",
    "split_paragraphs",
    "split_sentences",
    "stem",
    "stems",
    "strip_accents",
    "token_f1",
    "tokenize",
    "truncate",
]
