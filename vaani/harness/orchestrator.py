"""Stage orchestration: retries, tracing, circuit breaking, budget shedding.

This module is what makes Vaani a *harness* rather than a function call. A raw
RAG demo does ``retrieve(); generate()`` and hopes. Here every step is a
:class:`Stage` with a declared timeout, a retry policy, a contract validator and
a fallback, and the whole run is measured against a wall-clock deadline.

The single most important design decision: **optional stages are shed before the
budget is blown, not after.** A pipeline that notices at 210 ms that it was
supposed to finish in 200 ms has already failed. :class:`Deadline` is consulted
*between* stages, so query expansion and reranking are dropped while there is
still time for the stages that actually produce an answer.

Timeouts are enforced by measurement rather than by threads or signals. Every
stage here is synchronous, CPU-bound and short; wrapping each one in a thread
would cost more than the work itself, and ``signal.alarm`` has 1-second
granularity and is main-thread-only, which is useless inside a threaded server.
So a stage that overruns is detected immediately *after* it returns and raises
:class:`StageTimeout`. The one genuinely blocking stage -- a network STT call --
gets its timeout from the socket layer instead, which is where it belongs.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence

from vaani.config import Config, get_config
from vaani.harness.contracts import Span, Trace, new_id
from vaani.harness.errors import (
    BudgetExceededError,
    CircuitOpenError,
    ContractViolation,
    StageTimeout,
    VaaniError,
    classify,
)

__all__ = [
    "Deadline",
    "CircuitBreaker",
    "Tracer",
    "TraceRing",
    "Tool",
    "ToolRegistry",
    "Stage",
    "PipelineContext",
    "Pipeline",
    "retry_call",
    "time_block",
]


# ---------------------------------------------------------------------------
# Deadline
# ---------------------------------------------------------------------------


class Deadline:
    """Monotonic budget tracker for one request.

    Uses ``time.perf_counter`` rather than ``time.time`` so a clock adjustment
    mid-request cannot make the budget appear to jump.
    """

    __slots__ = ("budget_ms", "_start")

    def __init__(self, budget_ms: float) -> None:
        self.budget_ms = float(budget_ms)
        self._start = time.perf_counter()

    def reset(self) -> None:
        self._start = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0

    def remaining_ms(self) -> float:
        return self.budget_ms - self.elapsed_ms

    def spend_fraction(self) -> float:
        """Fraction of the budget consumed. May exceed 1.0."""
        if self.budget_ms <= 0:
            return 0.0
        return self.elapsed_ms / self.budget_ms

    def expired(self) -> bool:
        return self.remaining_ms() <= 0.0

    def check(self, stage: str = "") -> None:
        if self.expired():
            raise BudgetExceededError(
                f"latency budget of {self.budget_ms:.0f}ms exhausted",
                stage=stage,
                elapsed_ms=round(self.elapsed_ms, 2),
            )


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


def retry_call(
    fn: Callable[[], Any],
    *,
    attempts: int = 2,
    base_ms: float = 8.0,
    max_ms: float = 400.0,
    jitter: float = 0.35,
    deadline: Deadline | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    rng: random.Random | None = None,
) -> Any:
    """Call ``fn`` with exponential backoff, retrying only retryable errors.

    Retryability comes from the exception *type* via :func:`errors.classify`,
    never from string matching on the message. Matching on message text is how
    retry loops end up hammering a service that returned 401.

    ``attempts`` counts total tries, so ``attempts=2`` means one retry. Backoff
    uses full jitter (a uniform draw over ``[0, delay]`` scaled by ``jitter``),
    which decorrelates concurrent callers far better than fixed backoff.
    """
    if attempts < 1:
        attempts = 1
    rnd = rng or random
    last: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            retryable, _degradable = classify(exc)
            last = exc
            if not retryable or attempt >= attempts:
                raise
            delay = min(max_ms, base_ms * (2 ** (attempt - 1)))
            delay = delay * (1.0 - jitter) + rnd.random() * delay * jitter
            # Never sleep past the request deadline: a retry that lands after
            # the budget has gone is strictly worse than failing over now.
            if deadline is not None:
                room = deadline.remaining_ms()
                if room <= 0 or delay >= room:
                    raise
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            time.sleep(delay / 1000.0)

    assert last is not None  # unreachable
    raise last


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Per-key breaker: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.

    Keyed so one flaky STT vendor cannot trip the breaker for another. Guarded
    by a lock because the HTTP server is threaded and breaker state is shared.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        threshold: int = 5,
        cooldown_ms: float = 15000.0,
        half_open_probes: int = 1,
    ) -> None:
        self.threshold = max(1, threshold)
        self.cooldown_ms = cooldown_ms
        self.half_open_probes = max(1, half_open_probes)
        self._lock = threading.Lock()
        self._fails: dict[str, int] = {}
        self._state: dict[str, str] = {}
        self._opened_at: dict[str, float] = {}
        self._probes: dict[str, int] = {}

    def state(self, key: str) -> str:
        with self._lock:
            return self._resolve(key)

    def _resolve(self, key: str) -> str:
        """Compute current state, promoting OPEN -> HALF_OPEN after cooldown.

        Caller must hold the lock.
        """
        st = self._state.get(key, self.CLOSED)
        if st == self.OPEN:
            since = (time.perf_counter() - self._opened_at.get(key, 0.0)) * 1000.0
            if since >= self.cooldown_ms:
                self._state[key] = self.HALF_OPEN
                self._probes[key] = 0
                return self.HALF_OPEN
        return st

    def allow(self, key: str) -> bool:
        with self._lock:
            st = self._resolve(key)
            if st == self.CLOSED:
                return True
            if st == self.HALF_OPEN:
                if self._probes.get(key, 0) < self.half_open_probes:
                    self._probes[key] = self._probes.get(key, 0) + 1
                    return True
                return False
            return False

    def guard(self, key: str) -> None:
        """Raise :class:`CircuitOpenError` when the breaker is not allowing calls."""
        if not self.allow(key):
            raise CircuitOpenError(
                f"circuit open for {key!r}; not attempting the call",
                stage=key,
            )

    def record_success(self, key: str) -> None:
        with self._lock:
            self._fails[key] = 0
            self._state[key] = self.CLOSED
            self._probes[key] = 0

    def record_failure(self, key: str) -> None:
        with self._lock:
            st = self._resolve(key)
            n = self._fails.get(key, 0) + 1
            self._fails[key] = n
            # A failed half-open probe re-opens immediately: the service told us
            # it is still broken, so waiting for `threshold` more failures would
            # just be a slow way to learn the same thing.
            if st == self.HALF_OPEN or n >= self.threshold:
                self._state[key] = self.OPEN
                self._opened_at[key] = time.perf_counter()
                self._fails[key] = self.threshold

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                k: {"state": self._resolve(k), "failures": self._fails.get(k, 0)}
                for k in set(self._state) | set(self._fails)
            }


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


class Tracer:
    """Builds a tree of :class:`Span` for one request.

    Spans nest via an explicit stack, so a stage that internally traces a
    sub-step produces a child span rather than a sibling. The UI renders the
    tree, which is how a viewer can see that ``retrieve`` contains
    ``bm25`` + ``dense`` + ``fuse`` and where the milliseconds actually went.
    """

    def __init__(self, trace_id: str = "", enabled: bool = True) -> None:
        self.trace = Trace(trace_id=trace_id or new_id("trace"))
        self.enabled = enabled
        self._stack: list[Span] = []
        self._t0 = time.perf_counter()

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        sp = Span(
            name=name,
            start_ms=(time.perf_counter() - self._t0) * 1000.0,
            attrs=dict(attrs),
        )
        if self._stack:
            self._stack[-1].children.append(sp)
        else:
            self.trace.spans.append(sp)
        self._stack.append(sp)
        t = time.perf_counter()
        try:
            yield sp
        except BaseException as exc:  # noqa: BLE001 - annotate then re-raise
            sp.status = "timeout" if isinstance(exc, StageTimeout) else "error"
            sp.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            sp.duration_ms = (time.perf_counter() - t) * 1000.0
            self._stack.pop()

    # Kept as a distinct name because `span` is the common case and reads better
    # at call sites; `start_span` is the alias other tracing APIs use.
    start_span = span

    def note(self, **attrs: Any) -> None:
        """Attach attributes to the innermost open span."""
        if self._stack:
            self._stack[-1].attrs.update(attrs)

    def finish(self) -> Trace:
        self.trace.total_ms = (time.perf_counter() - self._t0) * 1000.0
        return self.trace


@contextmanager
def time_block(tracer: Tracer | None, name: str, **attrs: Any) -> Iterator[Span | None]:
    """Trace a block when a tracer exists, otherwise do nothing measurable."""
    if tracer is None or not tracer.enabled:
        yield None
        return
    with tracer.span(name, **attrs) as sp:
        yield sp


class TraceRing:
    """Bounded ring of recent traces, exposed by the server's ``/api/traces``."""

    def __init__(self, size: int = 400) -> None:
        self._dq: deque[Trace] = deque(maxlen=max(1, size))
        self._lock = threading.Lock()

    def add(self, trace: Trace) -> None:
        with self._lock:
            self._dq.append(trace)

    def recent(self, limit: int = 20) -> list[Trace]:
        with self._lock:
            items = list(self._dq)
        return items[-max(1, limit) :][::-1]

    def clear(self) -> None:
        with self._lock:
            self._dq.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._dq)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _validate_schema(value: Any, schema: dict[str, Any], path: str = "args") -> None:
    """Compact JSON-Schema subset validator.

    Supports type/required/properties/enum/minimum/maximum/items/minLength.
    Deliberately not a full implementation -- it covers what tool specs use,
    and failing loudly on an unsupported keyword would be worse than ignoring
    it here, since the keyword can only ever loosen what we already check.
    """
    expected = schema.get("type")
    if expected:
        py = _JSON_TYPES.get(expected)
        # bool is a subclass of int in Python; a boolean is not a number here.
        if py and (not isinstance(value, py) or (expected in ("number", "integer") and isinstance(value, bool))):
            raise ContractViolation(
                f"{path}: expected {expected}, got {type(value).__name__}"
            )

    if "enum" in schema and value not in schema["enum"]:
        raise ContractViolation(f"{path}: {value!r} not one of {schema['enum']}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if lo is not None and value < lo:
            raise ContractViolation(f"{path}: {value} < minimum {lo}")
        if hi is not None and value > hi:
            raise ContractViolation(f"{path}: {value} > maximum {hi}")

    if isinstance(value, str):
        ml = schema.get("minLength")
        if ml is not None and len(value) < ml:
            raise ContractViolation(f"{path}: shorter than minLength {ml}")

    if isinstance(value, dict):
        for req in schema.get("required", ()):
            if req not in value:
                raise ContractViolation(f"{path}: missing required field {req!r}")
        props = schema.get("properties") or {}
        for key, sub in props.items():
            if key in value:
                _validate_schema(value[key], sub, f"{path}.{key}")

    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            _validate_schema(item, schema["items"], f"{path}[{i}]")


class Tool:
    """One callable exposed to the harness (and shaped for LLM tool-use)."""

    __slots__ = ("name", "description", "parameters", "fn", "timeout_ms", "calls", "total_ms", "errors")

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable[..., Any],
        timeout_ms: float = 100.0,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.fn = fn
        self.timeout_ms = timeout_ms
        self.calls = 0
        self.total_ms = 0.0
        self.errors = 0

    def spec(self) -> dict[str, Any]:
        """Anthropic tool-use shaped spec, usable verbatim by an LLM caller."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


class ToolRegistry:
    """Validated dispatch table.

    Arguments are checked against each tool's JSON Schema *before* dispatch, so
    a malformed call fails with a precise message instead of a ``TypeError``
    from deep inside retrieval.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable[..., Any],
        timeout_ms: float = 100.0,
    ) -> Tool:
        tool = Tool(name, description, parameters, fn, timeout_ms)
        with self._lock:
            self._tools[name] = tool
        return tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            from vaani.harness.errors import ToolNotFound

            raise ToolNotFound(
                f"no tool named {name!r}; available: {sorted(self._tools)}",
                tool=name,
            ) from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def list_specs(self) -> list[dict[str, Any]]:
        return [self._tools[n].spec() for n in sorted(self._tools)]

    def call(self, name: str, args: dict[str, Any] | None = None) -> Any:
        from vaani.harness.errors import ToolError

        tool = self.get(name)
        payload = dict(args or {})
        _validate_schema(payload, tool.parameters, f"{name}.args")
        t = time.perf_counter()
        try:
            result = tool.fn(**payload)
        except VaaniError:
            tool.errors += 1
            raise
        except Exception as exc:  # noqa: BLE001 - normalised into ToolError
            tool.errors += 1
            raise ToolError(f"tool {name!r} failed: {exc}", tool=name) from exc
        finally:
            tool.calls += 1
            tool.total_ms += (time.perf_counter() - t) * 1000.0
        return result

    def stats(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for n, t in self._tools.items():
            out[n] = {
                "calls": t.calls,
                "errors": t.errors,
                "mean_ms": round(t.total_ms / t.calls, 3) if t.calls else 0.0,
            }
        return out


def build_tool_registry(
    *,
    search_corpus: Callable[..., Any] | None = None,
    get_chunk: Callable[..., Any] | None = None,
    corpus_stats: Callable[..., Any] | None = None,
    detect_language: Callable[..., Any] | None = None,
    classify_query: Callable[..., Any] | None = None,
) -> ToolRegistry:
    """Wire the built-in tools from injected callables.

    Dependency injection rather than imports: this module must not depend on
    retrieval or the vector store, or the import graph becomes a cycle and the
    harness stops being reusable.
    """
    reg = ToolRegistry()

    if search_corpus is not None:
        reg.register(
            "search_corpus",
            "Retrieve the most relevant corpus chunks for a natural-language query.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                    "strategies": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["query"],
            },
            search_corpus,
            timeout_ms=120.0,
        )

    if get_chunk is not None:
        reg.register(
            "get_chunk",
            "Fetch one indexed chunk verbatim by its chunk id.",
            {
                "type": "object",
                "properties": {"chunk_id": {"type": "string", "minLength": 1}},
                "required": ["chunk_id"],
            },
            get_chunk,
            timeout_ms=10.0,
        )

    if corpus_stats is not None:
        reg.register(
            "corpus_stats",
            "Summary statistics about the indexed corpus and chunk strategies.",
            {"type": "object", "properties": {}},
            corpus_stats,
            timeout_ms=20.0,
        )

    if detect_language is not None:
        reg.register(
            "detect_language",
            "Detect the language of a text snippet from its script and vocabulary.",
            {
                "type": "object",
                "properties": {"text": {"type": "string", "minLength": 1}},
                "required": ["text"],
            },
            detect_language,
            timeout_ms=10.0,
        )

    if classify_query is not None:
        reg.register(
            "classify_query",
            "Classify a question into one of the supported answer shapes.",
            {
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 1}},
                "required": ["query"],
            },
            classify_query,
            timeout_ms=10.0,
        )

    return reg


# ---------------------------------------------------------------------------
# Stages and pipeline
# ---------------------------------------------------------------------------


class Stage:
    """One step in the pipeline, with its own reliability policy.

    ``optional`` is the load-shedding flag: an optional stage is skipped when
    the budget is running out, and skipped silently rather than raising. That is
    the difference between a system that degrades and one that falls over.
    """

    __slots__ = ("name", "fn", "timeout_ms", "retries", "optional", "fallback", "validate", "skip_if")

    def __init__(
        self,
        name: str,
        fn: Callable[["PipelineContext"], Any],
        *,
        timeout_ms: float = 100.0,
        retries: int = 0,
        optional: bool = False,
        fallback: Callable[["PipelineContext", BaseException | None], Any] | None = None,
        validate: Callable[[Any], bool] | None = None,
        skip_if: Callable[["PipelineContext"], bool] | None = None,
    ) -> None:
        self.name = name
        self.fn = fn
        self.timeout_ms = timeout_ms
        self.retries = retries
        self.optional = optional
        self.fallback = fallback
        self.validate = validate
        self.skip_if = skip_if


class PipelineContext:
    """Mutable carrier threaded through every stage.

    A dict would have worked, but the named attributes are what let
    ``strict_contracts`` catch a stage that forgot to set its output.
    """

    __slots__ = (
        "request",
        "values",
        "tracer",
        "deadline",
        "degraded",
        "warnings",
        "stage_ms",
        "skipped",
        "cfg",
    )

    def __init__(
        self,
        request: Any = None,
        tracer: Tracer | None = None,
        deadline: Deadline | None = None,
        cfg: Config | None = None,
    ) -> None:
        self.request = request
        self.values: dict[str, Any] = {}
        self.tracer = tracer
        self.deadline = deadline
        self.degraded: list[str] = []
        self.warnings: list[str] = []
        self.stage_ms: dict[str, float] = {}
        self.skipped: list[str] = []
        self.cfg = cfg or get_config()

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.values[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


class Pipeline:
    """Ordered stages with per-stage reliability and global budget shedding."""

    def __init__(
        self,
        stages: Sequence[Stage],
        cfg: Config | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.stages = list(stages)
        self.cfg = cfg or get_config()
        h = self.cfg.harness
        self.breaker = breaker or CircuitBreaker(
            threshold=h.breaker_threshold,
            cooldown_ms=h.breaker_cooldown_ms,
            half_open_probes=h.breaker_half_open_probes,
        )

    def run(self, ctx: PipelineContext) -> PipelineContext:
        h = self.cfg.harness
        for stage in self.stages:
            self._run_stage(stage, ctx, h)
        return ctx

    def _run_stage(self, stage: Stage, ctx: PipelineContext, h: Any) -> None:
        # Conditional skip (e.g. the STT stage when the request carried text).
        if stage.skip_if is not None and stage.skip_if(ctx):
            self._mark_skipped(ctx, stage, "not_applicable")
            return

        # Budget shedding, checked *before* the work starts.
        if stage.optional and ctx.deadline is not None:
            if ctx.deadline.spend_fraction() > h.soft_budget_ratio:
                self._mark_skipped(ctx, stage, "budget")
                ctx.degraded.append(f"{stage.name}:shed")
                return

        if not self.breaker.allow(stage.name):
            if stage.optional:
                self._mark_skipped(ctx, stage, "circuit_open")
                ctx.degraded.append(f"{stage.name}:circuit_open")
                return
            err = CircuitOpenError(f"circuit open for stage {stage.name!r}", stage=stage.name)
            self._handle_failure(stage, ctx, err)
            return

        attempts = 1 + max(0, stage.retries)
        t0 = time.perf_counter()
        try:
            with time_block(ctx.tracer, stage.name, optional=stage.optional) as sp:

                def invoke() -> Any:
                    return stage.fn(ctx)

                def note_retry(attempt: int, exc: BaseException, delay: float) -> None:
                    if sp is not None:
                        sp.attempt = attempt + 1
                        sp.status = "degraded"
                    ctx.warnings.append(
                        f"{stage.name}: retry {attempt} after {type(exc).__name__}"
                    )

                result = retry_call(
                    invoke,
                    attempts=attempts,
                    base_ms=h.retry_base_ms,
                    max_ms=h.retry_max_ms,
                    jitter=h.retry_jitter,
                    deadline=ctx.deadline,
                    on_retry=note_retry,
                )

                took = (time.perf_counter() - t0) * 1000.0
                budget = stage.timeout_ms or h.stage_timeouts_ms.get(stage.name, 0.0)
                if budget and took > budget:
                    raise StageTimeout(
                        f"stage {stage.name!r} took {took:.1f}ms, over its "
                        f"{budget:.0f}ms budget",
                        stage=stage.name,
                    )

                if h.strict_contracts and stage.validate is not None:
                    if not stage.validate(result):
                        raise ContractViolation(
                            f"stage {stage.name!r} produced output failing its contract",
                            stage=stage.name,
                        )

                ctx[stage.name] = result
                if sp is not None:
                    sp.attrs["ms"] = round(took, 3)
        except BaseException as exc:  # noqa: BLE001 - routed by policy
            ctx.stage_ms[stage.name] = (time.perf_counter() - t0) * 1000.0
            self.breaker.record_failure(stage.name)
            self._handle_failure(stage, ctx, exc)
            return

        ctx.stage_ms[stage.name] = (time.perf_counter() - t0) * 1000.0
        self.breaker.record_success(stage.name)

    def _mark_skipped(self, ctx: PipelineContext, stage: Stage, reason: str) -> None:
        ctx.skipped.append(stage.name)
        ctx.stage_ms.setdefault(stage.name, 0.0)
        if ctx.tracer is not None and ctx.tracer.enabled:
            sp = Span(name=stage.name, status="skipped", attrs={"reason": reason})
            ctx.tracer.trace.spans.append(sp)

    def _handle_failure(
        self, stage: Stage, ctx: PipelineContext, exc: BaseException
    ) -> None:
        _retryable, degradable = classify(exc)
        recoverable = stage.optional or degradable
        if not recoverable:
            raise exc

        ctx.degraded.append(f"{stage.name}:{type(exc).__name__}")
        ctx.warnings.append(f"{stage.name} degraded: {type(exc).__name__}: {exc}")
        if stage.fallback is not None:
            try:
                ctx[stage.name] = stage.fallback(ctx, exc)
                return
            except BaseException as fb_exc:  # noqa: BLE001
                # A failing fallback on a required stage is unrecoverable; on an
                # optional stage we can still continue without its output.
                ctx.warnings.append(
                    f"{stage.name} fallback failed: {type(fb_exc).__name__}: {fb_exc}"
                )
                if not stage.optional:
                    raise exc from fb_exc
        elif not stage.optional:
            # Degradable but with nothing to degrade *to*: the caller must cope
            # with a missing value, which is only safe if it is optional.
            raise exc
        ctx.skipped.append(stage.name)


def build_pipeline(stages: Sequence[Stage], cfg: Config | None = None) -> Pipeline:
    return Pipeline(stages, cfg=cfg)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover - manual exercise
    from vaani.harness.errors import TransientError

    print("== retry ==")
    state = {"n": 0}

    def flaky() -> str:
        state["n"] += 1
        if state["n"] < 3:
            raise TransientError("not yet", stage="demo")
        return f"ok after {state['n']}"

    print(retry_call(flaky, attempts=4, base_ms=1))

    print("\n== non-retryable is not retried ==")
    calls = {"n": 0}

    def bad() -> None:
        calls["n"] += 1
        raise ContractViolation("permanent")

    try:
        retry_call(bad, attempts=5, base_ms=1)
    except ContractViolation:
        print(f"raised after {calls['n']} call(s) -- expected 1")

    print("\n== circuit breaker ==")
    cb = CircuitBreaker(threshold=3, cooldown_ms=40)
    for i in range(3):
        cb.record_failure("stt")
    print("state after 3 failures:", cb.state("stt"), "allow:", cb.allow("stt"))
    time.sleep(0.05)
    print("state after cooldown:", cb.state("stt"), "allow probe:", cb.allow("stt"))
    cb.record_success("stt")
    print("state after success:", cb.state("stt"))

    print("\n== tools ==")
    reg = build_tool_registry(
        search_corpus=lambda query, top_k=5, strategies=None: [f"hit:{query}:{top_k}"],
        detect_language=lambda text: "en",
    )
    print("specs:", [s["name"] for s in reg.list_specs()])
    print("call:", reg.call("search_corpus", {"query": "everest height", "top_k": 2}))
    try:
        reg.call("search_corpus", {"top_k": 2})
    except ContractViolation as exc:
        print("validation caught:", exc)

    print("\n== pipeline with shedding ==")
    tracer = Tracer()
    dl = Deadline(30.0)
    ctx = PipelineContext(tracer=tracer, deadline=dl)

    def slow(_c: PipelineContext) -> str:
        time.sleep(0.026)
        return "slow-done"

    stages = [
        Stage("embed", lambda c: "vec", timeout_ms=50),
        Stage("retrieve", slow, timeout_ms=200),
        Stage("rerank", lambda c: "reranked", timeout_ms=50, optional=True),
        Stage("generate", lambda c: "answer", timeout_ms=200),
    ]
    Pipeline(stages).run(ctx)
    print("values:", ctx.values)
    print("skipped:", ctx.skipped, "degraded:", ctx.degraded)
    tr = tracer.finish()
    for sp in tr.spans:
        print(f"  {sp.name:10s} {sp.duration_ms:7.2f}ms  {sp.status}")
