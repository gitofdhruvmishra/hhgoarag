"""Typed data contracts shared by every stage of the Vaani pipeline.

Every stage in the harness declares its input and output types from this module.
Nothing here imports anything outside the standard library, and nothing here
imports another Vaani module -- this file is the root of the dependency graph so
that stage implementations stay decoupled and independently testable.

Design notes
------------
* ``slots=True`` on the hot-path dataclasses (Chunk, ScoredChunk) keeps per-object
  memory down and attribute access fast; we allocate tens of thousands of Chunks
  during indexing and hundreds of ScoredChunks per query.
* Every contract has ``to_dict()`` so the HTTP layer can serialise a whole
  ``QueryResult`` without a bespoke encoder.
* ``validate()`` methods implement cheap structural checks. The orchestrator
  calls them at stage boundaries when ``strict_contracts`` is on, which is how we
  catch a stage returning something malformed instead of letting it corrupt a
  downstream stage.
"""

from __future__ import annotations

import time
import uuid
from array import array
from dataclasses import dataclass, field, asdict
from typing import Any, Protocol, runtime_checkable

# --------------------------------------------------------------------------- #
# Vector type alias. We use array('f') rather than list[float] because
# math.sumprod over two array('f') buffers runs at C speed.
# --------------------------------------------------------------------------- #
Vector = array


def new_id(prefix: str) -> str:
    """Short, sortable-ish, human-readable identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# Corpus + chunking
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Passage:
    """One raw record as it arrives from the dataset loader.

    Mirrors the MSMARCO-XI record shape: a passage of text, the language it is
    written in, and whatever query/relevance metadata the dataset carries.
    """

    doc_id: str
    text: str
    lang: str = "en"
    title: str = ""
    source: str = "corpus"
    section: str = ""
    url: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.doc_id:
            raise ValueError("Passage.doc_id must be non-empty")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(f"Passage {self.doc_id} has empty text")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Passage":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(slots=True)
class Chunk:
    """An indexable unit produced by a chunking strategy.

    ``start``/``end`` are character offsets into the parent passage, which is
    what lets the UI highlight the exact span a citation came from.
    ``parent_id`` and ``level`` support hierarchical (small-to-big) retrieval:
    we match on precise level-0 chunks but can expand to the level-1 parent for
    generation context.
    """

    chunk_id: str
    doc_id: str
    text: str
    strategy: str
    lang: str = "en"
    start: int = 0
    end: int = 0
    token_count: int = 0
    char_count: int = 0
    level: int = 0
    parent_id: str | None = None
    title: str = ""
    section: str = ""
    ordinal: int = 0
    # Text that precedes/follows within the same document. Used by "sentence
    # window" retrieval: match tightly, generate from the widened window.
    window_before: str = ""
    window_after: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.chunk_id:
            raise ValueError("Chunk.chunk_id must be non-empty")
        if not self.text.strip():
            raise ValueError(f"Chunk {self.chunk_id} has empty text")
        if self.end < self.start:
            raise ValueError(f"Chunk {self.chunk_id} has end < start")

    @property
    def context_text(self) -> str:
        """The chunk widened by its window, for use as generation context."""
        parts = [p for p in (self.window_before, self.text, self.window_after) if p]
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(slots=True)
class ScoredChunk:
    """A chunk plus every score that contributed to its final ranking.

    We keep the component scores rather than just the fused number because the
    UI shows *why* a chunk was retrieved, and because the reranker and the
    grounding guard both consume the components independently.
    """

    chunk: Chunk
    score: float = 0.0
    lexical: float = 0.0
    dense: float = 0.0
    rerank: float = 0.0
    fusion: float = 0.0
    rank: int = 0
    matched_terms: tuple[str, ...] = ()
    retrievers: tuple[str, ...] = ()
    explain: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["matched_terms"] = list(self.matched_terms)
        d["retrievers"] = list(self.retrievers)
        return d


# --------------------------------------------------------------------------- #
# Speech to text
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Transcript:
    """Result of the speech-to-text stage."""

    text: str
    lang: str = "en"
    provider: str = "unknown"
    confidence: float = 1.0
    duration_ms: float = 0.0
    audio_ms: float = 0.0
    alternatives: tuple[str, ...] = ()
    degraded: bool = False  # True when we fell back off the primary provider
    raw: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("Transcript.text must be a string")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["alternatives"] = list(self.alternatives)
        return d


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Citation:
    """A pointer from a sentence of the answer back into the corpus."""

    marker: str  # "[1]"
    chunk_id: str
    doc_id: str
    title: str = ""
    quote: str = ""
    doc_start: int = 0
    doc_end: int = 0
    score: float = 0.0
    strategy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Answer:
    """Final generated answer plus everything needed to audit it."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    grounding: float = 0.0
    query_type: str = "unknown"
    generator: str = "extractive"
    abstained: bool = False
    reason: str = ""
    supporting_sentences: list[str] = field(default_factory=list)
    lang: str = "en"
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("Answer.text must be a string")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Answer.confidence out of range: {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["citations"] = [c.to_dict() for c in self.citations]
        return d


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #

# Actions the policy engine can take. Ordered by increasing restriction so the
# policy engine can simply take the max.
ACTION_ALLOW = "allow"
ACTION_CLARIFY = "clarify"
ACTION_ABSTAIN = "abstain"
ACTION_REFUSE = "refuse"

_ACTION_RANK = {
    ACTION_ALLOW: 0,
    ACTION_CLARIFY: 1,
    ACTION_ABSTAIN: 2,
    ACTION_REFUSE: 3,
}


def action_rank(action: str) -> int:
    return _ACTION_RANK.get(action, 0)


def stricter(a: str, b: str) -> str:
    return a if action_rank(a) >= action_rank(b) else b


@dataclass(slots=True)
class GuardCheck:
    """One individual guardrail evaluation.

    A check reports a normalised ``score`` in [0, 1] where higher means "more
    of the thing this check detects", plus the threshold it was compared
    against. Keeping both means the UI can show how close a query came to
    tripping a rail, not just a boolean.
    """

    name: str
    passed: bool
    score: float = 0.0
    threshold: float = 0.0
    action: str = ACTION_ALLOW
    severity: str = "none"  # none | low | medium | high | critical
    detail: str = ""
    evidence: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    stage: str = "input"  # input | retrieval | output

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GuardVerdict:
    """Aggregate decision across all guardrail checks for one request."""

    allowed: bool = True
    action: str = ACTION_ALLOW
    category: str = "safe"
    severity: str = "none"
    risk: float = 0.0
    message: str = ""  # user-facing explanation when we decline
    checks: list[GuardCheck] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def tripped(self) -> list[GuardCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["checks"] = [c.to_dict() for c in self.checks]
        d["tripped"] = [c.name for c in self.tripped]
        return d


# --------------------------------------------------------------------------- #
# Tracing + latency
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Span:
    """One timed unit of work inside a request."""

    name: str
    start_ms: float = 0.0
    duration_ms: float = 0.0
    status: str = "ok"  # ok | error | skipped | degraded | timeout
    attempt: int = 1
    error: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_ms": round(self.start_ms, 4),
            "duration_ms": round(self.duration_ms, 4),
            "status": self.status,
            "attempt": self.attempt,
            "error": self.error,
            "attrs": self.attrs,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass(slots=True)
class LatencyBreakdown:
    """Per-stage wall-clock in milliseconds.

    ``pipeline_ms`` is the number the task's 200 ms budget applies to: chunking
    lookup + vector retrieval + rerank + generation + guardrails. ``total_ms``
    additionally includes speech-to-text, which for a hosted provider is
    dominated by network round-trip and is reported separately so the two are
    never conflated.
    """

    stt_ms: float = 0.0
    guard_input_ms: float = 0.0
    query_transform_ms: float = 0.0
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    guard_output_ms: float = 0.0
    overhead_ms: float = 0.0
    pipeline_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: round(v, 3) for k, v in asdict(self).items()}


@dataclass(slots=True)
class Trace:
    """The span tree for one request."""

    trace_id: str = ""
    spans: list[Span] = field(default_factory=list)
    total_ms: float = 0.0
    started_at: float = field(default_factory=time.time)

    def flat(self) -> list[Span]:
        out: list[Span] = []

        def walk(spans: list[Span]) -> None:
            for s in spans:
                out.append(s)
                walk(s.children)

        walk(self.spans)
        return out

    def stage_ms(self) -> dict[str, float]:
        return {s.name: round(s.duration_ms, 3) for s in self.spans}

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "total_ms": round(self.total_ms, 3),
            "spans": [s.to_dict() for s in self.spans],
        }


# --------------------------------------------------------------------------- #
# Request / response
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class QueryRequest:
    """Everything the pipeline needs to answer one question."""

    text: str | None = None
    audio_b64: str | None = None
    audio_mime: str = "audio/webm"
    lang: str | None = None
    top_k: int = 5
    candidates: int = 120
    strategies: tuple[str, ...] = ()  # empty = all indexed strategies
    generator: str = "auto"  # auto | extractive | llm
    stt_provider: str = "auto"
    want_trace: bool = True
    use_cache: bool = True
    session_id: str = ""
    request_id: str = field(default_factory=lambda: new_id("req"))

    def validate(self) -> None:
        if not (self.text or self.audio_b64):
            raise ValueError("QueryRequest needs either text or audio_b64")
        if self.top_k < 1 or self.top_k > 50:
            raise ValueError(f"top_k out of range: {self.top_k}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Never echo raw audio back to the client or into logs.
        d["audio_b64"] = f"<{len(self.audio_b64)} b64 chars>" if self.audio_b64 else None
        d["strategies"] = list(self.strategies)
        return d


@dataclass(slots=True)
class QueryResult:
    """The full, auditable outcome of one request."""

    request_id: str
    query: str = ""
    normalized_query: str = ""
    transcript: Transcript | None = None
    retrieved: list[ScoredChunk] = field(default_factory=list)
    answer: Answer | None = None
    guard: GuardVerdict = field(default_factory=GuardVerdict)
    trace: Trace = field(default_factory=Trace)
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    cache_hit: bool = False
    degraded: list[str] = field(default_factory=list)
    query_expansions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "normalized_query": self.normalized_query,
            "transcript": self.transcript.to_dict() if self.transcript else None,
            "retrieved": [s.to_dict() for s in self.retrieved],
            "answer": self.answer.to_dict() if self.answer else None,
            "guard": self.guard.to_dict(),
            "trace": self.trace.to_dict(),
            "latency": self.latency.to_dict(),
            "cache_hit": self.cache_hit,
            "degraded": self.degraded,
            "query_expansions": self.query_expansions,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
# Stage interfaces. Implementations live in their own subpackages; the
# orchestrator only ever depends on these Protocols.
# --------------------------------------------------------------------------- #


@runtime_checkable
class Chunker(Protocol):
    """Splits one Passage into indexable Chunks."""

    name: str

    def chunk(self, passage: Passage) -> list[Chunk]: ...

    def describe(self) -> dict[str, Any]: ...


@runtime_checkable
class Embedder(Protocol):
    """Maps text to a dense unit-norm vector."""

    name: str
    dim: int

    def fit(self, texts: list[str]) -> None: ...

    def encode(self, text: str) -> Vector: ...

    def encode_batch(self, texts: list[str]) -> list[Vector]: ...


@runtime_checkable
class STTProvider(Protocol):
    """Converts audio bytes to text."""

    name: str

    def available(self) -> bool: ...

    def transcribe(self, audio: bytes, mime: str, lang: str | None) -> Transcript: ...


@runtime_checkable
class Generator(Protocol):
    """Composes a grounded answer from retrieved chunks."""

    name: str

    def generate(
        self, query: str, chunks: list[ScoredChunk], lang: str = "en"
    ) -> Answer: ...


__all__ = [
    "ACTION_ABSTAIN",
    "ACTION_ALLOW",
    "ACTION_CLARIFY",
    "ACTION_REFUSE",
    "Answer",
    "Chunk",
    "Chunker",
    "Citation",
    "Embedder",
    "Generator",
    "GuardCheck",
    "GuardVerdict",
    "LatencyBreakdown",
    "Passage",
    "QueryRequest",
    "QueryResult",
    "STTProvider",
    "ScoredChunk",
    "Span",
    "Trace",
    "Transcript",
    "Vector",
    "action_rank",
    "new_id",
    "stricter",
]
