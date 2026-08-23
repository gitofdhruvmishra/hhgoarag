"""Span tracing for the request pipeline.

Every request carries a :class:`Tracer`. Stages open spans on it, and the
resulting :class:`Trace` is what the UI renders as a waterfall and what the
benchmark reads per-stage numbers out of. Two properties make it worth having
rather than sprinkling ``perf_counter`` calls around:

* The tree is built implicitly from a stack, so a stage that internally times
  three sub-steps needs no plumbing to have them nest correctly.
* Failure is recorded, not swallowed. An exception passing through a span sets
  its status (``error``/``timeout``) and captures the exception type before it
  propagates, so a trace shows *where* a request broke and how long the broken
  stage took, not just that it broke.

Timing uses ``time.perf_counter``, the same monotonic clock as
:class:`vaani.harness.retry.Deadline`, so span durations and budget spend are
directly comparable.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Iterator

from ..config import HarnessConfig, get_config
from ..textkit import truncate
from .contracts import LatencyBreakdown, Span, Trace, new_id
from .errors import BudgetExceededError, StageTimeout, VaaniError

# Span statuses, mirroring the values documented on contracts.Span.
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"
STATUS_DEGRADED = "degraded"
STATUS_TIMEOUT = "timeout"

# Longest exception message we copy into a span. Traces are held in memory for
# the UI ring, so an exception carrying a whole document must not be retained.
_MAX_ERR_CHARS = 200


def status_for_exception(exc: BaseException) -> str:
    """Map an exception to a span status.

    Anything that means "ran out of time" becomes ``timeout`` rather than
    ``error``, because in the waterfall those two want different colours: a
    timeout is a budget problem, an error is a correctness problem.
    """
    if isinstance(exc, (StageTimeout, BudgetExceededError, TimeoutError)):
        return STATUS_TIMEOUT
    return STATUS_ERROR


def _annotate(span: Span, exc: BaseException) -> None:
    """Record an exception on a span without losing an explicit status."""
    if span.status == STATUS_OK:
        span.status = status_for_exception(exc)
    if not span.error:
        span.error = type(exc).__name__
    msg = str(exc)
    if msg:
        span.attrs["error_message"] = truncate(msg, _MAX_ERR_CHARS)
    if isinstance(exc, VaaniError):
        span.attrs["error_code"] = exc.code
        span.attrs["degradable"] = exc.degradable


# --------------------------------------------------------------------------- #
# Tracer
# --------------------------------------------------------------------------- #


class Tracer:
    """Collects the span tree for a single request.

    Nesting is per-thread. One Tracer belongs to one request, and stages are
    synchronous, so in the normal case there is exactly one stack. But a stage
    that fans out to worker threads (racing two STT providers, say) would
    otherwise interleave pushes and pops from different threads into one stack
    and produce a garbage tree -- so each thread gets its own stack, and spans
    opened on a worker thread attach at the root. The structural mutations are
    taken under a lock because the ``children`` lists are shared.

    Spans are always recorded, even when ``HarnessConfig.emit_traces`` is off:
    the latency breakdown is derived from them, so switching them off would
    cost the numbers we report. ``emit_traces`` governs whether the finished
    trace is *retained* in the ring for the UI.
    """

    __slots__ = ("trace_id", "_t0", "_roots", "_stacks", "_lock", "_trace")

    def __init__(self, trace_id: str = "") -> None:
        self.trace_id = trace_id or new_id("trc")
        self._t0 = time.perf_counter()
        self._roots: list[Span] = []
        self._stacks: dict[int, list[Span]] = {}
        self._lock = threading.Lock()
        self._trace: Trace | None = None

    # ------------------------------------------------------------------ #
    def _now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def _push(self, span: Span) -> None:
        tid = threading.get_ident()
        with self._lock:
            stack = self._stacks.get(tid)
            if stack is None:
                stack = []
                self._stacks[tid] = stack
            if stack:
                stack[-1].children.append(span)
            else:
                self._roots.append(span)
            stack.append(span)

    def _pop(self, span: Span) -> None:
        tid = threading.get_ident()
        with self._lock:
            stack = self._stacks.get(tid)
            if not stack:
                return
            if stack[-1] is span:
                stack.pop()
            elif span in stack:
                # Out-of-order exit (a generator span abandoned mid-flight).
                # Truncate rather than leave a dangling parent that would
                # capture every later span as a child.
                del stack[stack.index(span) :]

    # ------------------------------------------------------------------ #
    @contextmanager
    def start_span(self, name: str, **attrs: Any) -> Iterator[Span]:
        """Open a timed span, nested under whatever span is currently open.

        The yielded Span is mutable: a stage sets ``status`` to ``degraded``,
        adds ``attrs``, or bumps ``attempt`` as it goes. An explicitly-set
        status is never overwritten by the exception handler.
        """
        span = Span(name=name, start_ms=self._now_ms(), attrs=dict(attrs) if attrs else {})
        self._push(span)
        t = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            _annotate(span, exc)
            raise
        finally:
            span.duration_ms = (time.perf_counter() - t) * 1000.0
            self._pop(span)

    def mark(
        self,
        name: str,
        *,
        status: str = STATUS_OK,
        duration_ms: float = 0.0,
        **attrs: Any,
    ) -> Span:
        """Record a zero-work span (a skipped stage, a cache hit) and return it."""
        span = Span(
            name=name,
            start_ms=self._now_ms(),
            duration_ms=float(duration_ms),
            status=status,
            attrs=dict(attrs) if attrs else {},
        )
        tid = threading.get_ident()
        with self._lock:
            stack = self._stacks.get(tid)
            if stack:
                stack[-1].children.append(span)
            else:
                self._roots.append(span)
        return span

    def current(self) -> Span | None:
        """The innermost open span on this thread, if any."""
        stack = self._stacks.get(threading.get_ident())
        return stack[-1] if stack else None

    def set_attrs(self, **attrs: Any) -> None:
        """Attach attributes to the innermost open span, if there is one."""
        span = self.current()
        if span is not None:
            span.attrs.update(attrs)

    # ------------------------------------------------------------------ #
    def elapsed_ms(self) -> float:
        return self._now_ms()

    def finish(self) -> Trace:
        """Close the tracer and return the Trace. Idempotent.

        Idempotency matters because both the orchestrator's ``finally`` block
        and the HTTP layer's error handler may call it for the same request.
        """
        if self._trace is None:
            self._trace = Trace(
                trace_id=self.trace_id, spans=self._roots, total_ms=self._now_ms()
            )
        return self._trace

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Tracer({self.trace_id!r}, spans={len(self._roots)}, {self._now_ms():.1f}ms)"


@contextmanager
def time_block(tracer: Tracer | None, name: str, **attrs: Any) -> Iterator[Span]:
    """Time a block of work, tolerating a missing tracer.

    Lets code paths that may or may not be running inside a traced request use
    one spelling instead of branching on ``if tracer is not None`` everywhere.
    With no tracer the span is still timed and returned, just detached from any
    tree.
    """
    if tracer is not None:
        with tracer.start_span(name, **attrs) as span:
            yield span
        return

    span = Span(name=name, start_ms=0.0, attrs=dict(attrs) if attrs else {})
    t = time.perf_counter()
    try:
        yield span
    except Exception as exc:
        _annotate(span, exc)
        raise
    finally:
        span.duration_ms = (time.perf_counter() - t) * 1000.0


# --------------------------------------------------------------------------- #
# Latency breakdown
# --------------------------------------------------------------------------- #

# Top-level span name -> LatencyBreakdown field. Keys match the stage names in
# HarnessConfig.stage_timeouts_ms so a stage's deadline and its reported latency
# are the same concept spelled the same way.
_SPAN_TO_FIELD: dict[str, str] = {
    "stt": "stt_ms",
    "guard_input": "guard_input_ms",
    "query_transform": "query_transform_ms",
    "embed": "embed_ms",
    "retrieve": "retrieve_ms",
    "rerank": "rerank_ms",
    "generate": "generate_ms",
    "guard_output": "guard_output_ms",
}


def latency_from_trace(trace: Trace) -> LatencyBreakdown:
    """Roll a Trace's top-level spans up into a :class:`LatencyBreakdown`.

    ``pipeline_ms`` excludes speech-to-text deliberately: STT against a hosted
    provider is dominated by network round-trip, and folding that into the
    number the 200 ms budget applies to would make the budget meaningless.
    ``overhead_ms`` is whatever the request spent outside any recognised stage
    (serialisation, cache lookups, orchestration itself).
    """
    lb = LatencyBreakdown(total_ms=trace.total_ms)
    accounted = 0.0
    for span in trace.spans:
        field = _SPAN_TO_FIELD.get(span.name)
        if field is None:
            continue
        # Repeated spans with the same name (a retried stage that re-opened its
        # span, a two-pass retrieve) accumulate rather than overwrite.
        setattr(lb, field, getattr(lb, field) + span.duration_ms)
        accounted += span.duration_ms
    lb.pipeline_ms = max(0.0, trace.total_ms - lb.stt_ms)
    lb.overhead_ms = max(0.0, trace.total_ms - accounted)
    return lb


# --------------------------------------------------------------------------- #
# Trace ring
# --------------------------------------------------------------------------- #


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile over a pre-sorted list."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = int(round((pct / 100.0) * (len(values) - 1)))
    return values[max(0, min(len(values) - 1, idx))]


class TraceRing:
    """Bounded, thread-safe buffer of recent traces for ``GET /api/traces``.

    A ``deque(maxlen=...)`` rather than a list because the UI only ever wants
    the recent tail and an unbounded log would be a memory leak in a
    long-running demo. Lookups by id scan the ring, which is fine at the default
    size of 400 and avoids a second index that would have to be evicted in step
    with the deque.
    """

    __slots__ = ("_dq", "_lock", "maxlen")

    def __init__(self, size: int | None = None, *, cfg: HarnessConfig | None = None) -> None:
        h = cfg if cfg is not None else get_config().harness
        self.maxlen = int(h.trace_ring_size if size is None else size)
        if self.maxlen < 1:
            self.maxlen = 1
        self._dq: deque[Trace] = deque(maxlen=self.maxlen)
        self._lock = threading.Lock()

    def add(self, trace: Trace) -> Trace:
        with self._lock:
            self._dq.append(trace)
        return trace

    def recent(self, n: int | None = None) -> list[Trace]:
        """Most recent first."""
        with self._lock:
            items = list(self._dq)
        items.reverse()
        return items if n is None else items[: max(0, int(n))]

    def get(self, trace_id: str) -> Trace | None:
        with self._lock:
            for tr in reversed(self._dq):
                if tr.trace_id == trace_id:
                    return tr
        return None

    def as_dicts(self, n: int | None = None) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.recent(n)]

    def stats(self) -> dict[str, Any]:
        """Aggregate latency over the ring, plus mean time per stage.

        This is the cheap always-on view of pipeline health; the benchmark
        module produces the rigorous numbers.
        """
        traces = self.recent()
        totals = sorted(t.total_ms for t in traces)
        stage_totals: dict[str, float] = {}
        stage_counts: dict[str, int] = {}
        errors = 0
        for t in traces:
            for span in t.spans:
                stage_totals[span.name] = stage_totals.get(span.name, 0.0) + span.duration_ms
                stage_counts[span.name] = stage_counts.get(span.name, 0) + 1
                if span.status in (STATUS_ERROR, STATUS_TIMEOUT):
                    errors += 1
        return {
            "count": len(traces),
            "capacity": self.maxlen,
            "error_spans": errors,
            "total_ms": {
                "p50": round(_percentile(totals, 50), 3),
                "p90": round(_percentile(totals, 90), 3),
                "p95": round(_percentile(totals, 95), 3),
                "max": round(totals[-1], 3) if totals else 0.0,
            },
            "stage_mean_ms": {
                name: round(stage_totals[name] / stage_counts[name], 3)
                for name in sorted(stage_totals)
            },
        }

    def clear(self) -> None:
        with self._lock:
            self._dq.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._dq)

    def __iter__(self) -> Iterator[Trace]:
        return iter(self.recent())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TraceRing({len(self)}/{self.maxlen})"


__all__ = [
    "STATUS_DEGRADED",
    "STATUS_ERROR",
    "STATUS_OK",
    "STATUS_SKIPPED",
    "STATUS_TIMEOUT",
    "TraceRing",
    "Tracer",
    "latency_from_trace",
    "status_for_exception",
    "time_block",
]


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from .errors import RetrievalError

    def render(spans: list[Span], depth: int = 0) -> None:
        for s in spans:
            pad = "  " * depth
            extra = f" {s.error}" if s.error else ""
            attrs = {k: v for k, v in s.attrs.items() if k != "error_message"}
            print(
                f"  {pad}{s.name:<18} {s.duration_ms:7.2f}ms  {s.status:<8}"
                f"{extra}  {attrs if attrs else ''}"
            )
            render(s.children, depth + 1)

    print("== Tracer: nested spans, a degraded stage and a failed stage ==")
    tracer = Tracer()
    with tracer.start_span("stt", provider="browser") as span:
        time.sleep(0.004)
        span.attrs["chars"] = 42
    with tracer.start_span("guard_input"):
        time.sleep(0.001)
    with tracer.start_span("retrieve", candidates=160) as span:
        with tracer.start_span("bm25"):
            time.sleep(0.003)
        with tracer.start_span("dense", ivf_probe=8):
            time.sleep(0.002)
        span.attrs["hits"] = 37
    with tracer.start_span("rerank") as span:
        span.status = STATUS_DEGRADED  # explicit status survives the handler
        span.attrs["reason"] = "cross_encoder_unavailable"
    try:
        with tracer.start_span("generate"):
            time.sleep(0.001)
            raise RetrievalError("no candidates survived the guard", stage="generate")
    except RetrievalError:
        pass
    trace = tracer.finish()
    render(trace.spans)
    print(f"  total: {trace.total_ms:.2f}ms  trace_id={trace.trace_id}")
    print("  stage_ms:", trace.stage_ms())
    print("  flat span count:", len(trace.flat()))

    print("\n== latency_from_trace ==")
    lb = latency_from_trace(trace)
    print(" ", lb.to_dict())
    print(
        f"  pipeline_ms excludes stt: total={lb.total_ms:.2f} "
        f"stt={lb.stt_ms:.2f} pipeline={lb.pipeline_ms:.2f}"
    )

    print("\n== time_block without a tracer ==")
    with time_block(None, "detached", note="works with tracer=None") as s:
        time.sleep(0.002)
    print(f"  {s.name}: {s.duration_ms:.2f}ms attrs={s.attrs}")

    print("\n== TraceRing eviction + stats ==")
    ring = TraceRing(size=5)
    for i in range(9):
        t = Tracer(trace_id=f"trc_{i:02d}")
        with t.start_span("retrieve"):
            time.sleep(0.001 * (i % 3 + 1))
        with t.start_span("generate"):
            time.sleep(0.001)
        ring.add(t.finish())
    print("  len:", len(ring), "(capacity 5)")
    print("  ids newest-first:", [tr.trace_id for tr in ring.recent()])
    print("  lookup trc_07:", ring.get("trc_07") is not None, "| evicted trc_01:", ring.get("trc_01"))
    print("  stats:", ring.stats())

    print("\n== Tracer is safe from 6 concurrent threads ==")
    shared = Tracer()
    def worker(n: int) -> None:
        for i in range(20):
            with shared.start_span(f"t{n}"):
                with shared.start_span("inner"):
                    pass
    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    tr = shared.finish()
    print(f"  root spans={len(tr.spans)} (expected 120) flat={len(tr.flat())} (expected 240)")
