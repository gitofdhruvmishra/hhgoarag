"""Dense embeddings with no model download.

There is no sentence-transformers here, no ONNX file, no 400 MB checkpoint. The
semantics are *earned from the corpus itself* at index time, in three steps:

1. **Co-occurrence.** Slide a window over the corpus and count which terms
   appear near which. Distributional semantics in one sentence: words used in
   the same contexts mean similar things.

2. **PPMI.** Raw counts are dominated by frequency, so convert to Positive
   Pointwise Mutual Information, which measures how much more often two terms
   co-occur than chance would predict. Context-distribution smoothing
   (``alpha=0.75``) is applied to the context marginal -- Levy & Goldberg's
   correction, which stops rare contexts from producing wildly inflated PMI.

3. **Random indexing.** A full term-term matrix is |V|x|V|, which is far too
   big. Instead every term gets a sparse ternary random signature, and a term's
   embedding is the PPMI-weighted sum of its contexts' signatures. This is the
   Johnson-Lindenstrauss trick: random projection preserves distances well
   enough, and because signatures are *derived from a seeded hash of the term*
   there is no projection matrix to store or ship.

Text embedding then IDF-weights the term vectors and subtracts the corpus's
dominant direction (SIF-style common-component removal). That last step matters
more than it sounds: without it, every short query vector points mostly at the
"generic English prose" direction and cosine similarity stops discriminating.

All arithmetic runs through :func:`math.sumprod` over :class:`array.array`
buffers, which is a C loop. A pure-Python dot product would be ~40x slower and
the 200 ms budget would be unreachable.
"""

from __future__ import annotations

import json
import math
import struct
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from vaani.config import Config, get_config
from vaani.harness.errors import EmbeddingError, IndexCorruptError
from vaani.textkit import char_ngrams, content_tokens, stem

__all__ = [
    "l2_normalize",
    "cosine",
    "zeros",
    "add_scaled",
    "BaseEmbedder",
    "LexicalStats",
    "PPMIRandomIndexEmbedder",
    "CharNgramEmbedder",
    "CompositeEmbedder",
    "build_embedder",
]

_MAGIC = b"VAANIEMB"
_FORMAT_VERSION = 2


# ---------------------------------------------------------------------------
# Vector primitives
# ---------------------------------------------------------------------------


def zeros(dim: int) -> array:
    return array("f", bytes(4 * dim))


def l2_normalize(vec: array) -> array:
    """Scale ``vec`` to unit length in place; a zero vector is left alone."""
    norm = math.sqrt(math.sumprod(vec, vec))
    if norm > 1e-12:
        inv = 1.0 / norm
        for i in range(len(vec)):
            vec[i] *= inv
    return vec


def cosine(a: array, b: array) -> float:
    """Dot product. Assumes both inputs are already L2-normalised.

    Every vector this package stores is normalised at creation, so the division
    by norms is redundant work inside the hottest loop in the system. Callers
    that hold an unnormalised vector must normalise it first.
    """
    return math.sumprod(a, b)


def add_scaled(dst: array, src: array, scale: float) -> None:
    for i in range(len(dst)):
        dst[i] += src[i] * scale


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BaseEmbedder:
    """Common shape for everything implementing the ``Embedder`` protocol."""

    name = "base"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.fitted = False

    def fit(self, texts: Sequence[str]) -> "BaseEmbedder":  # pragma: no cover - abstract
        raise NotImplementedError

    def encode(self, text: str) -> array:  # pragma: no cover - abstract
        raise NotImplementedError

    def encode_batch(self, texts: Sequence[str]) -> list[array]:
        return [self.encode(t) for t in texts]

    # -- persistence --------------------------------------------------------
    def state(self) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def load_state(self, state: dict[str, Any]) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps(
            {"name": self.name, "dim": self.dim, "version": _FORMAT_VERSION, "state": self.state()},
            ensure_ascii=False,
        ).encode("utf-8")
        with path.open("wb") as fh:
            fh.write(_MAGIC)
            fh.write(struct.pack("<II", _FORMAT_VERSION, len(blob)))
            fh.write(blob)

    def load(self, path: str | Path) -> "BaseEmbedder":
        path = Path(path)
        with path.open("rb") as fh:
            if fh.read(len(_MAGIC)) != _MAGIC:
                raise IndexCorruptError(f"{path} is not a Vaani embedder file")
            version, length = struct.unpack("<II", fh.read(8))
            if version != _FORMAT_VERSION:
                raise IndexCorruptError(
                    f"embedder format v{version}, expected v{_FORMAT_VERSION}; "
                    "re-run `python3 run.py index --rebuild`"
                )
            payload = json.loads(fh.read(length).decode("utf-8"))
        if payload["dim"] != self.dim:
            self.dim = int(payload["dim"])
        self.load_state(payload["state"])
        self.fitted = True
        return self


# ---------------------------------------------------------------------------
# Lexical statistics
# ---------------------------------------------------------------------------


class LexicalStats:
    """Corpus term statistics, shared by BM25 and by every IDF weighting.

    One object owns these numbers so the sparse index, the dense embedder, the
    reranker and the grounding guard all agree on how rare a term is. When they
    disagree, scores stop being comparable and fusion weights become arbitrary.
    """

    def __init__(self) -> None:
        self.n_docs = 0
        self.avg_len = 0.0
        self.total_terms = 0
        self._df: dict[str, int] = {}
        self._tf: dict[str, int] = {}
        self._stem_surface: dict[str, set[str]] = defaultdict(set)
        self._max_idf = 1.0

    def fit(self, docs_tokens: Sequence[Sequence[str]]) -> "LexicalStats":
        df: dict[str, int] = defaultdict(int)
        tf: dict[str, int] = defaultdict(int)
        total = 0
        for toks in docs_tokens:
            total += len(toks)
            for t in toks:
                tf[t] += 1
            for t in set(toks):
                df[t] += 1
                self._stem_surface[stem(t)].add(t)
        self.n_docs = len(docs_tokens)
        self.total_terms = total
        self.avg_len = (total / self.n_docs) if self.n_docs else 0.0
        self._df = dict(df)
        self._tf = dict(tf)
        # An unseen term is maximally informative, so cache the ceiling rather
        # than recomputing it on every miss.
        self._max_idf = self._raw_idf(1) if self.n_docs else 1.0
        return self

    def _raw_idf(self, df: int) -> float:
        # BM25 probabilistic IDF with the +1 that keeps it non-negative.
        return math.log(((self.n_docs - df + 0.5) / (df + 0.5)) + 1.0)

    def df(self, term: str) -> int:
        return self._df.get(term, 0)

    def term_freq(self, term: str) -> int:
        return self._tf.get(term, 0)

    def idf(self, term: str) -> float:
        d = self._df.get(term)
        if not d:
            return self._max_idf
        return self._raw_idf(d)

    @property
    def vocab(self) -> set[str]:
        return set(self._df)

    def vocabulary(self, min_df: int = 1) -> set[str]:
        if min_df <= 1:
            return set(self._df)
        return {t for t, d in self._df.items() if d >= min_df}

    def surface_forms(self, term: str) -> set[str]:
        """Surface variants sharing a stem with ``term``, used for query expansion."""
        return set(self._stem_surface.get(stem(term), ()))

    def state(self) -> dict[str, Any]:
        return {
            "n_docs": self.n_docs,
            "avg_len": self.avg_len,
            "total_terms": self.total_terms,
            "df": self._df,
            "tf": self._tf,
        }

    def load_state(self, st: dict[str, Any]) -> None:
        self.n_docs = int(st["n_docs"])
        self.avg_len = float(st["avg_len"])
        self.total_terms = int(st["total_terms"])
        self._df = dict(st["df"])
        self._tf = dict(st.get("tf") or {})
        self._stem_surface = defaultdict(set)
        for t in self._df:
            self._stem_surface[stem(t)].add(t)
        self._max_idf = self._raw_idf(1) if self.n_docs else 1.0


# ---------------------------------------------------------------------------
# Random index signatures
# ---------------------------------------------------------------------------

_SIG_NONZEROS = 8


def _signature(term: str, dim: int, seed: int) -> tuple[tuple[int, float], ...]:
    """Sparse ternary random signature for one term.

    Derived from a seeded SplitMix-style hash of the term text, so the same term
    yields the same signature in every process without persisting a projection
    matrix. ``hash()`` is unusable here: PYTHONHASHSEED randomises it per run,
    which would silently invalidate a saved index.
    """
    h = seed & 0xFFFFFFFFFFFFFFFF
    for ch in term:
        h = (h ^ ord(ch)) * 0x100000001B3 & 0xFFFFFFFFFFFFFFFF
    out: list[tuple[int, float]] = []
    used: set[int] = set()
    while len(out) < _SIG_NONZEROS:
        h = (h * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        idx = (h >> 17) % dim
        if idx in used:
            continue
        used.add(idx)
        out.append((idx, 1.0 if (h >> 61) & 1 else -1.0))
    return tuple(out)


# ---------------------------------------------------------------------------
# PPMI + random indexing
# ---------------------------------------------------------------------------


class PPMIRandomIndexEmbedder(BaseEmbedder):
    """Distributional embeddings from corpus co-occurrence statistics."""

    name = "ppmi-ri"

    def __init__(
        self,
        dim: int | None = None,
        cfg: Config | None = None,
        *,
        window: int = 5,
        min_df: int = 2,
        max_contexts: int = 64,
        alpha: float = 0.75,
    ) -> None:
        cfg = cfg or get_config()
        super().__init__(dim or cfg.retrieval.embed_dim)
        self.seed = cfg.seed
        self.window = window
        self.min_df = min_df
        self.max_contexts = max_contexts
        self.alpha = alpha
        self.stats = LexicalStats()
        self._ppmi: dict[str, tuple[tuple[str, float], ...]] = {}
        self._term_vecs: dict[str, array] = {}
        self._common: array | None = None

    # -- fitting -----------------------------------------------------------
    def fit(self, texts: Sequence[str]) -> "PPMIRandomIndexEmbedder":
        docs = [content_tokens(t) for t in texts]
        self.stats.fit(docs)
        keep = self.stats.vocabulary(min_df=self.min_df)
        if not keep:
            # A corpus this small has no distributional signal to extract; the
            # char-ngram rail in CompositeEmbedder still works, so degrade
            # rather than fail.
            keep = self.stats.vocab

        cooc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        marginal: dict[str, float] = defaultdict(float)
        total = 0.0
        w = self.window
        for toks in docs:
            filtered = [t for t in toks if t in keep]
            n = len(filtered)
            for i, a in enumerate(filtered):
                hi = min(n, i + w + 1)
                for j in range(i + 1, hi):
                    b = filtered[j]
                    if a == b:
                        continue
                    # Distance weighting: adjacent terms are much stronger
                    # evidence of a semantic relationship than terms five apart.
                    weight = 1.0 / (j - i)
                    cooc[a][b] += weight
                    cooc[b][a] += weight
                    marginal[a] += weight
                    marginal[b] += weight
                    total += 2.0 * weight

        if total <= 0.0:
            self._ppmi = {}
            self._term_vecs = {}
            self.fitted = True
            return self

        # Context-distribution smoothing: flattening the context marginal by
        # alpha<1 raises the probability of rare contexts, which counteracts
        # PMI's well-known bias toward them.
        smoothed = {t: m**self.alpha for t, m in marginal.items()}
        smooth_total = sum(smoothed.values()) or 1.0

        ppmi: dict[str, tuple[tuple[str, float], ...]] = {}
        for a, ctxs in cooc.items():
            p_a = marginal[a] / total
            if p_a <= 0.0:
                continue
            scored: list[tuple[str, float]] = []
            for b, c in ctxs.items():
                p_ab = c / total
                p_b = smoothed.get(b, 0.0) / smooth_total
                if p_ab <= 0.0 or p_b <= 0.0:
                    continue
                val = math.log(p_ab / (p_a * p_b))
                if val > 0.0:
                    scored.append((b, val))
            if not scored:
                continue
            # Truncating to the strongest contexts bounds both memory and encode
            # time; the tail is mostly noise anyway.
            scored.sort(key=lambda kv: kv[1], reverse=True)
            ppmi[a] = tuple(scored[: self.max_contexts])

        self._ppmi = ppmi
        self._build_term_vectors()
        self._fit_common_direction(texts)
        self.fitted = True
        return self

    def _build_term_vectors(self) -> None:
        dim, seed = self.dim, self.seed
        sig_cache: dict[str, tuple[tuple[int, float], ...]] = {}
        vecs: dict[str, array] = {}
        for term, ctxs in self._ppmi.items():
            v = zeros(dim)
            for ctx, weight in ctxs:
                sig = sig_cache.get(ctx)
                if sig is None:
                    sig = _signature(ctx, dim, seed)
                    sig_cache[ctx] = sig
                for idx, sign in sig:
                    v[idx] += sign * weight
            vecs[term] = l2_normalize(v)
        self._term_vecs = vecs

    def _fit_common_direction(self, texts: Sequence[str], sample: int = 900) -> None:
        """Find the dominant direction across document vectors by power iteration.

        Every natural-language document vector has a large shared component --
        roughly "this is prose" -- and it swamps cosine similarity for short
        queries. Projecting it out is the single cheapest quality win available.
        """
        step = max(1, len(texts) // sample) if texts else 1
        picks = [self._raw_encode(t) for t in texts[::step][:sample]]
        picks = [v for v in picks if math.sumprod(v, v) > 1e-9]
        if len(picks) < 8:
            self._common = None
            return
        u = zeros(self.dim)
        for i in range(self.dim):
            u[i] = 1.0 / math.sqrt(self.dim)
        for _ in range(8):
            nxt = zeros(self.dim)
            for v in picks:
                add_scaled(nxt, v, math.sumprod(v, u))
            if math.sumprod(nxt, nxt) <= 1e-12:
                self._common = None
                return
            u = l2_normalize(nxt)
        self._common = u

    # -- encoding ----------------------------------------------------------
    def _raw_encode(self, text: str) -> array:
        v = zeros(self.dim)
        toks = content_tokens(text)
        if not toks:
            return v
        hits = 0
        for t in toks:
            tv = self._term_vecs.get(t)
            if tv is None:
                # Fall back to a stem-sharing surface form: speech-to-text and
                # Indic morphology both produce variants the corpus never had.
                for alt in self.stats.surface_forms(t):
                    tv = self._term_vecs.get(alt)
                    if tv is not None:
                        break
            if tv is None:
                continue
            add_scaled(v, tv, self.stats.idf(t))
            hits += 1
        if hits:
            l2_normalize(v)
        return v

    def encode(self, text: str) -> array:
        if not self.fitted:
            raise EmbeddingError("embedder used before fit(); build the index first")
        v = self._raw_encode(text)
        if self._common is not None and math.sumprod(v, v) > 1e-12:
            add_scaled(v, self._common, -math.sumprod(v, self._common))
            l2_normalize(v)
        return v

    def term_neighbours(self, term: str, k: int = 5) -> list[tuple[str, float]]:
        """Nearest distributional neighbours -- powers query expansion and the UI."""
        tv = self._term_vecs.get(term) or self._term_vecs.get(stem(term))
        if tv is None:
            return []
        scored = [
            (other, math.sumprod(tv, ov))
            for other, ov in self._term_vecs.items()
            if other != term
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:k]

    # -- persistence -------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "window": self.window,
            "min_df": self.min_df,
            "max_contexts": self.max_contexts,
            "alpha": self.alpha,
            "stats": self.stats.state(),
            # Only PPMI contexts are stored: term vectors are a pure function of
            # them plus the seed, so recomputing on load is cheaper than writing
            # dim floats per term to disk.
            "ppmi": {t: [[c, round(w, 5)] for c, w in ctxs] for t, ctxs in self._ppmi.items()},
            "common": list(self._common) if self._common is not None else None,
        }

    def load_state(self, st: dict[str, Any]) -> None:
        self.seed = int(st["seed"])
        self.window = int(st["window"])
        self.min_df = int(st["min_df"])
        self.max_contexts = int(st["max_contexts"])
        self.alpha = float(st["alpha"])
        self.stats = LexicalStats()
        self.stats.load_state(st["stats"])
        self._ppmi = {t: tuple((c, float(w)) for c, w in ctxs) for t, ctxs in st["ppmi"].items()}
        self._build_term_vectors()
        common = st.get("common")
        self._common = array("f", common) if common else None
        self.fitted = True


# ---------------------------------------------------------------------------
# Character n-grams
# ---------------------------------------------------------------------------


class CharNgramEmbedder(BaseEmbedder):
    """Hashed character n-grams -- the rail that survives misspelling.

    Whole-word matching fails exactly where voice input hurts most: a
    speech-to-text engine writes "Everest" as "Everast", and an Indic word
    appears in three inflected forms. Character n-grams degrade gracefully
    because most of the substrings still match.
    """

    name = "charngram"

    def __init__(self, dim: int | None = None, cfg: Config | None = None) -> None:
        cfg = cfg or get_config()
        super().__init__(dim or cfg.retrieval.embed_dim)
        self.seed = cfg.seed ^ 0x5EED
        self.stats = LexicalStats()
        self._sig: dict[str, tuple[tuple[int, float], ...]] = {}

    def fit(self, texts: Sequence[str]) -> "CharNgramEmbedder":
        self.stats.fit([content_tokens(t) for t in texts])
        self.fitted = True
        return self

    def _gram_sig(self, gram: str) -> tuple[tuple[int, float], ...]:
        sig = self._sig.get(gram)
        if sig is None:
            sig = _signature(gram, self.dim, self.seed)
            self._sig[gram] = sig
        return sig

    def encode(self, text: str) -> array:
        v = zeros(self.dim)
        toks = content_tokens(text)
        if not toks:
            return v
        for t in toks:
            weight = self.stats.idf(t) if self.fitted else 1.0
            grams = char_ngrams(t, 3, 5)
            if not grams:
                continue
            # Normalising by gram count stops long words from dominating purely
            # by having more substrings.
            share = weight / math.sqrt(len(grams))
            for g in grams:
                for idx, sign in self._gram_sig(g):
                    v[idx] += sign * share
        return l2_normalize(v)

    def state(self) -> dict[str, Any]:
        return {"seed": self.seed, "stats": self.stats.state()}

    def load_state(self, st: dict[str, Any]) -> None:
        self.seed = int(st["seed"])
        self.stats = LexicalStats()
        self.stats.load_state(st["stats"])
        self._sig = {}
        self.fitted = True


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


class CompositeEmbedder(BaseEmbedder):
    """Blend of distributional and sub-word signals behind one interface.

    The two rails fail in different places -- PPMI needs the term to have been
    seen, char-ngrams need only the spelling to be close -- so a weighted blend
    is strictly more robust than either. Downstream code never has to know.
    """

    name = "composite"

    def __init__(
        self,
        cfg: Config | None = None,
        *,
        w_ppmi: float = 0.7,
        w_char: float = 0.3,
    ) -> None:
        cfg = cfg or get_config()
        super().__init__(cfg.retrieval.embed_dim)
        self.cfg = cfg
        self.w_ppmi = w_ppmi
        self.w_char = w_char
        self.ppmi = PPMIRandomIndexEmbedder(self.dim, cfg)
        self.char = CharNgramEmbedder(self.dim, cfg)

    @property
    def stats(self) -> LexicalStats:
        return self.ppmi.stats

    def fit(self, texts: Sequence[str]) -> "CompositeEmbedder":
        self.ppmi.fit(texts)
        self.char.fit(texts)
        self.fitted = True
        return self

    def encode(self, text: str) -> array:
        a = self.ppmi.encode(text) if self.ppmi.fitted else zeros(self.dim)
        b = self.char.encode(text)
        v = zeros(self.dim)
        add_scaled(v, a, self.w_ppmi)
        add_scaled(v, b, self.w_char)
        return l2_normalize(v)

    def term_neighbours(self, term: str, k: int = 5) -> list[tuple[str, float]]:
        return self.ppmi.term_neighbours(term, k)

    def state(self) -> dict[str, Any]:
        return {
            "w_ppmi": self.w_ppmi,
            "w_char": self.w_char,
            "ppmi": self.ppmi.state(),
            "char": self.char.state(),
        }

    def load_state(self, st: dict[str, Any]) -> None:
        self.w_ppmi = float(st["w_ppmi"])
        self.w_char = float(st["w_char"])
        self.ppmi.load_state(st["ppmi"])
        self.char.load_state(st["char"])
        self.fitted = True


def build_embedder(cfg: Config | None = None) -> CompositeEmbedder:
    return CompositeEmbedder(cfg or get_config())


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover - manual exercise
    import time

    corpus = [
        "New Delhi is the capital city of India and seat of the national government.",
        "The national capital of France is Paris, located on the river Seine.",
        "Tokyo is the capital city of Japan and its largest metropolitan area.",
        "Photosynthesis is the process by which green plants convert light energy into glucose.",
        "Chlorophyll in plant leaves absorbs sunlight to drive photosynthesis and release oxygen.",
        "Plants use carbon dioxide and water during photosynthesis to build sugars.",
        "Mount Everest rises 8849 metres above sea level on the Nepal China border.",
        "K2 is the second highest mountain on Earth at 8611 metres in the Karakoram range.",
        "The Ganges is a major river flowing across northern India into Bangladesh.",
        "The Seine is a river in northern France flowing through the city of Paris.",
        "Cricket is a bat and ball sport played between two teams of eleven players.",
        "Football is played by two teams of eleven players on a rectangular pitch.",
    ] * 4

    t = time.perf_counter()
    emb = build_embedder()
    emb.fit(corpus)
    print(f"fit {len(corpus)} texts in {(time.perf_counter()-t)*1000:.1f}ms  dim={emb.dim}")

    q = emb.encode("capital city")
    for probe in ("national capital", "photosynthesis", "highest mountain", "river in France"):
        print(f"  cos('capital city', {probe!r:22s}) = {cosine(q, emb.encode(probe)):+.4f}")

    near = cosine(q, emb.encode("national capital"))
    far = cosine(q, emb.encode("photosynthesis"))
    print(f"\nassert near({near:+.4f}) > far({far:+.4f}):", "PASS" if near > far else "FAIL")

    print("\nneighbours of 'capital':", [t for t, _ in emb.term_neighbours("capital", 5)])
    print("neighbours of 'photosynthesis':", [t for t, _ in emb.term_neighbours("photosynthesis", 5)])

    t = time.perf_counter()
    n = 400
    for _ in range(n):
        emb.encode("what is the capital of india")
    print(f"\nencode: {(time.perf_counter()-t)*1000/n:.3f}ms per query")

    # Misspelling robustness is the char-ngram rail's whole job.
    ok = emb.encode("photosynthesis")
    typo = emb.encode("photosynthisis")
    print(f"typo tolerance cos = {cosine(ok, typo):+.4f}")
