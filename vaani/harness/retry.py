"""Retries, circuit breaking and deadline tracking.

These three primitives are what let the orchestrator be *declarative* about
failure. A stage says "I am retryable twice, I am degradable, I get 45 ms" and
this module supplies the mechanics:

* ``retry_call`` decides whether to retry by asking :func:`errors.classify`
  about the exception *type*. A non-retryable error is never retried, no matter
  how many attempts were budgeted -- retrying a ``ContractViolation`` is pure
  latency with a guaranteed identical outcome.
* ``CircuitBreaker`` stops us hammering a dead dependency. The threaded HTTP
  server means several requests hit the same breaker concurrently, so every
  transition is taken under a lock.
* ``Deadline`` is the object the 200 ms budget lives in. Everything that wants
  to know "do I still have time for this?" asks a Deadline rather than doing its
  own arithmetic on timestamps.

All timing uses ``time.perf_counter``: it is monotonic (immune to wall-clock
adjustments mid-request) and it is the same clock the tracer uses, so a span
duration and a budget spend can be compared directly.
"""

from __future__ import annotations

import math
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

from ..config import HarnessConfig, get_config
from .errors import BudgetExceededError, CircuitOpenError, VaaniError, classify

T = TypeVar("T")

# Breaker states. Exported as plain strings so they can go straight into JSON
# for the UI without a serialiser.
BREAKER_CLOSED = "closed"
BREAKER_OPEN = "open"
BREAKER_HALF_OPEN = "half_open"


def _harness(cfg: HarnessConfig | None = None) -> HarnessConfig:
    """Resolve a HarnessConfig lazily.

    Deliberately *not* read at import time: tests and the benchmark harness
    mutate the config singleton after modules are imported, and a value captured
    at import would silently ignore them.
    """
    return cfg if cfg is not None else get_config().harness


# --------------------------------------------------------------------------- #
# Deadline
# --------------------------------------------------------------------------- #


class Deadline:
    """Monotonic budget tracker for one request.

    A Deadline is created once per request with the pipeline budget and then
    threaded through every stage. Two questions matter to callers:

    * ``expired()``        -- is the budget gone?
    * ``spend_fraction()`` -- how much of it have we used?

    ``spend_fraction`` is intentionally *not* clamped to 1.0. The orchestrator
    compares it against ``soft_budget_ratio`` to shed optional work, and the
    trace records the raw value, so an overrun shows up as 1.37 rather than
    being flattened into an indistinguishable 1.0.
    """

    __slots__ = ("budget_ms", "label", "_t0")

    def __init__(
        self,
        budget_ms: float | None = None,
        *,
        label: str = "pipeline",
        cfg: HarnessConfig | None = None,
    ) -> None:
        self.budget_ms: float = (
            float(budget_ms)
            if budget_ms is not None
            else float(_harness(cfg).pipeline_budget_ms)
        )
        self.label = label
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------ #
    @classmethod
    def unlimited(cls, label: str = "unlimited") -> "Deadline":
        """A Deadline that never expires. Used for index-time work and the CLI."""
        return cls(math.inf, label=label)

    def reset(self) -> "Deadline":
        self._t0 = time.perf_counter()
        return self

    # ------------------------------------------------------------------ #
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def remaining_ms(self) -> float:
        """Milliseconds left, floored at zero (callers use it to cap sleeps)."""
        if not math.isfinite(self.budget_ms):
            return math.inf
        return max(0.0, self.budget_ms - self.elapsed_ms())

    def overrun_ms(self) -> float:
        """How far past the budget we are, or 0.0 if still inside it."""
        if not math.isfinite(self.budget_ms):
            return 0.0
        return max(0.0, self.elapsed_ms() - self.budget_ms)

    def expired(self) -> bool:
        return self.remaining_ms() <= 0.0

    def spend_fraction(self) -> float:
        """Elapsed / budget. Unclamped; >1.0 means we are already over."""
        if not math.isfinite(self.budget_ms):
            return 0.0
        if self.budget_ms <= 0.0:
            return math.inf
        return self.elapsed_ms() / self.budget_ms

    def check(self, stage: str = "") -> None:
        """Raise :class:`BudgetExceededError` if the budget is spent."""
        if self.expired():
            raise BudgetExceededError(
                f"latency budget of {self.budget_ms:.0f}ms exhausted",
                stage=stage,
                budget_ms=round(self.budget_ms, 3),
                elapsed_ms=round(self.elapsed_ms(), 3),
            )

    def clamp_ms(self, want_ms: float) -> float:
        """The largest slice of ``want_ms`` that still fits in the budget."""
        return min(float(want_ms), self.remaining_ms())

    def child(self, budget_ms: float | None = None, *, label: str = "") -> "Deadline":
        """A sub-deadline that can never outlive its parent.

        Used when a stage wants to bound one of several internal steps: the
        child gets ``min(requested, parent remaining)`` so a generous per-step
        number can never overrun the request as a whole.
        """
        rem = self.remaining_ms()
        want = rem if budget_ms is None else min(float(budget_ms), rem)
        return Deadline(want, label=label or f"{self.label}.child")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "budget_ms": None if math.isinf(self.budget_ms) else round(self.budget_ms, 3),
            "elapsed_ms": round(self.elapsed_ms(), 3),
            "remaining_ms": None if math.isinf(self.remaining_ms()) else round(self.remaining_ms(), 3),
            "spend_fraction": round(self.spend_fraction(), 4),
            "expired": self.expired(),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Deadline({self.label!r}, budget={self.budget_ms:.0f}ms, "
            f"spent={self.spend_fraction() * 100:.0f}%)"
        )


# --------------------------------------------------------------------------- #
# Backoff
# --------------------------------------------------------------------------- #


def backoff_ms(
    attempt: int,
    base_ms: float,
    max_ms: float,
    jitter: float = 1.0,
    *,
    rand: Callable[[], float] = random.random,
) -> float:
    """Exponentially increasing delay with jitter, in milliseconds.

    ``attempt`` is 1-based, so the first retry waits ~``base_ms``.

    ``jitter`` interpolates between no jitter and AWS-style *full* jitter:
    ``0.0`` returns the raw exponential delay, ``1.0`` returns a uniform draw
    from ``[0, delay]``. The default config value (0.35) keeps most of the
    backoff shape while still spreading a thundering herd of retries that all
    failed at the same instant.
    """
    if attempt < 1:
        attempt = 1
    raw = min(float(max_ms), float(base_ms) * (2.0 ** (attempt - 1)))
    j = min(1.0, max(0.0, float(jitter)))
    if j <= 0.0:
        return raw
    return raw * (1.0 - j) + raw * j * rand()


# --------------------------------------------------------------------------- #
# retry_call
# --------------------------------------------------------------------------- #


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int | None = None,
    base_ms: float | None = None,
    max_ms: float | None = None,
    jitter: float | None = None,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    deadline: Deadline | None = None,
    sleep: Callable[[float], None] = time.sleep,
    cfg: HarnessConfig | None = None,
) -> T:
    """Call ``fn`` with exponential backoff, retrying only retryable errors.

    ``fn`` takes no arguments -- callers pass a closure. Attempt numbers are
    reported through ``on_retry(attempt, exc, delay_ms)`` rather than being
    handed to ``fn``, which keeps the retried callable trivial to write.

    Two independent gates must both pass for a retry to happen:

    1. the exception is an instance of ``retry_on`` (a coarse structural
       filter, defaulting to everything), and
    2. :func:`errors.classify` reports it as retryable.

    The second gate is the important one -- it is what makes ``retry_on`` safe
    to leave at its default, because the error taxonomy already knows which
    failures are worth repeating.

    When a ``deadline`` is supplied we refuse to retry if the backoff sleep
    alone would not fit in the remaining budget: burning the last 5 ms of a
    request on a sleep guarantees a timeout instead of merely risking one.
    """
    h = _harness(cfg)
    n = int(h.max_retries + 1 if attempts is None else attempts)
    if n < 1:
        n = 1
    base = h.retry_base_ms if base_ms is None else float(base_ms)
    cap = h.retry_max_ms if max_ms is None else float(max_ms)
    jit = h.retry_jitter if jitter is None else float(jitter)

    for attempt in range(1, n + 1):
        try:
            return fn()
        except Exception as exc:  # BaseException (KeyboardInterrupt) never retried
            if not isinstance(exc, retry_on):
                raise
            retryable, _degradable = classify(exc)
            if not retryable or attempt >= n:
                raise
            delay = backoff_ms(attempt, base, cap, jit)
            if deadline is not None:
                rem = deadline.remaining_ms()
                if rem <= 0.0 or delay > rem:
                    raise
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            if delay > 0.0:
                sleep(delay / 1000.0)
    # Unreachable: the loop either returns or raises.
    raise AssertionError("retry_call exhausted without returning or raising")


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #


class _KeyState:
    """Mutable per-key breaker state. Only ever touched under the owner's lock."""

    __slots__ = (
        "state",
        "consecutive_failures",
        "opened_at",
        "probes",
        "total_calls",
        "total_failures",
        "trips",
        "last_error",
        "last_change",
    )

    def __init__(self) -> None:
        self.state = BREAKER_CLOSED
        self.consecutive_failures = 0
        self.opened_at = 0.0
        self.probes = 0
        self.total_calls = 0
        self.total_failures = 0
        self.trips = 0
        self.last_error = ""
        self.last_change = time.perf_counter()


class CircuitBreaker:
    """Per-key CLOSED -> OPEN -> HALF_OPEN state machine.

    Keys are stage or provider names ("retrieve", "stt:sarvam"), so one flaky
    dependency cannot take the rest of the pipeline down with it.

    Transitions:

    * CLOSED    -- calls pass. ``breaker_threshold`` *consecutive* failures trip
      it to OPEN. A single success resets the counter, because an intermittent
      failure every other call is a different problem from a dependency that is
      simply down, and only the latter is worth failing fast on.
    * OPEN      -- calls are refused for ``breaker_cooldown_ms``, then the next
      caller is promoted into HALF_OPEN.
    * HALF_OPEN -- at most ``breaker_half_open_probes`` trial calls are let
      through. One success closes the circuit; one failure re-opens it and
      restarts the cooldown.

    Every method takes the lock. The HTTP server is threaded, so two requests
    can be inside ``allow()`` for the same key simultaneously, and probe
    admission has to be exact or a half-open circuit floods the dependency it
    was supposed to protect.
    """

    __slots__ = ("threshold", "cooldown_ms", "half_open_probes", "_states", "_lock")

    def __init__(
        self,
        *,
        threshold: int | None = None,
        cooldown_ms: float | None = None,
        half_open_probes: int | None = None,
        cfg: HarnessConfig | None = None,
    ) -> None:
        h = _harness(cfg)
        self.threshold = int(h.breaker_threshold if threshold is None else threshold)
        self.cooldown_ms = float(
            h.breaker_cooldown_ms if cooldown_ms is None else cooldown_ms
        )
        self.half_open_probes = int(
            h.breaker_half_open_probes if half_open_probes is None else half_open_probes
        )
        self._states: dict[str, _KeyState] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def _get(self, key: str) -> _KeyState:
        st = self._states.get(key)
        if st is None:
            st = _KeyState()
            self._states[key] = st
        return st

    def allow(self, key: str) -> bool:
        """True if a call for ``key`` may proceed. Consumes a half-open probe."""
        now = time.perf_counter()
        with self._lock:
            st = self._states.get(key)
            if st is None or st.state == BREAKER_CLOSED:
                return True
            if st.state == BREAKER_OPEN:
                if (now - st.opened_at) * 1000.0 < self.cooldown_ms:
                    return False
                st.state = BREAKER_HALF_OPEN
                st.probes = 0
                st.last_change = now
            if st.probes < self.half_open_probes:
                st.probes += 1
                return True
            return False

    def check(self, key: str) -> None:
        """Raise :class:`CircuitOpenError` instead of returning False.

        ``CircuitOpenError`` is declared non-retryable but degradable, so a
        stage guarded this way skips straight to its fallback.
        """
        if not self.allow(key):
            raise CircuitOpenError(
                f"circuit breaker open for {key!r}",
                stage=key,
                cooldown_ms=self.cooldown_ms,
            )

    def record_success(self, key: str) -> None:
        now = time.perf_counter()
        with self._lock:
            st = self._get(key)
            st.total_calls += 1
            st.consecutive_failures = 0
            if st.state != BREAKER_CLOSED:
                st.state = BREAKER_CLOSED
                st.probes = 0
                st.last_change = now

    def record_failure(self, key: str, error: BaseException | str = "") -> None:
        now = time.perf_counter()
        with self._lock:
            st = self._get(key)
            st.total_calls += 1
            st.total_failures += 1
            st.consecutive_failures += 1
            if error:
                st.last_error = (
                    error if isinstance(error, str) else type(error).__name__
                )
            if st.state == BREAKER_HALF_OPEN:
                # A failed probe means the dependency is still unhealthy.
                st.state = BREAKER_OPEN
                st.opened_at = now
                st.probes = 0
                st.trips += 1
                st.last_change = now
            elif (
                st.state == BREAKER_CLOSED
                and st.consecutive_failures >= self.threshold
            ):
                st.state = BREAKER_OPEN
                st.opened_at = now
                st.probes = 0
                st.trips += 1
                st.last_change = now

    # ------------------------------------------------------------------ #
    @contextmanager
    def guard(self, key: str) -> Iterator[None]:
        """Check the breaker, then record the outcome of the wrapped block."""
        self.check(key)
        try:
            yield
        except Exception as exc:
            self.record_failure(key, exc)
            raise
        else:
            self.record_success(key)

    def call(self, key: str, fn: Callable[[], T]) -> T:
        with self.guard(key):
            return fn()

    # ------------------------------------------------------------------ #
    def state(self, key: str) -> str:
        """Current state, resolving an elapsed cooldown to ``half_open``.

        Read-only: unlike ``allow()`` this does not consume a probe, so the UI
        can poll it freely.
        """
        with self._lock:
            st = self._states.get(key)
            if st is None:
                return BREAKER_CLOSED
            if (
                st.state == BREAKER_OPEN
                and (time.perf_counter() - st.opened_at) * 1000.0 >= self.cooldown_ms
            ):
                return BREAKER_HALF_OPEN
            return st.state

    def is_open(self, key: str) -> bool:
        return self.state(key) == BREAKER_OPEN

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._states.clear()
            else:
                self._states.pop(key, None)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Per-key state for the /api/health and /api/traces views."""
        now = time.perf_counter()
        out: dict[str, dict[str, Any]] = {}
        with self._lock:
            for key, st in self._states.items():
                state = st.state
                if (
                    state == BREAKER_OPEN
                    and (now - st.opened_at) * 1000.0 >= self.cooldown_ms
                ):
                    state = BREAKER_HALF_OPEN
                out[key] = {
                    "state": state,
                    "consecutive_failures": st.consecutive_failures,
                    "total_calls": st.total_calls,
                    "total_failures": st.total_failures,
                    "trips": st.trips,
                    "last_error": st.last_error,
                    "cooldown_remaining_ms": (
                        round(max(0.0, self.cooldown_ms - (now - st.opened_at) * 1000.0), 1)
                        if st.state == BREAKER_OPEN
                        else 0.0
                    ),
                }
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CircuitBreaker(threshold={self.threshold}, keys={len(self._states)})"


__all__ = [
    "BREAKER_CLOSED",
    "BREAKER_HALF_OPEN",
    "BREAKER_OPEN",
    "CircuitBreaker",
    "Deadline",
    "backoff_ms",
    "retry_call",
]


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from .errors import ContractViolation, TransientError

    print("== backoff shape (base=8ms, cap=400ms, jitter=0.35) ==")
    for a in range(1, 7):
        lo = backoff_ms(a, 8.0, 400.0, 0.35, rand=lambda: 0.0)
        hi = backoff_ms(a, 8.0, 400.0, 0.35, rand=lambda: 1.0)
        print(f"  attempt {a}: {lo:7.2f} .. {hi:7.2f} ms")

    print("\n== retry_call: flaky call succeeds on attempt 3 ==")
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientError("upstream hiccup", stage="demo")
        return f"ok after {calls['n']} attempts"

    seen: list[str] = []
    got = retry_call(
        flaky,
        attempts=4,
        base_ms=1.0,
        max_ms=4.0,
        on_retry=lambda a, e, d: seen.append(f"retry#{a} after {type(e).__name__} in {d:.2f}ms"),
    )
    print(" ", got)
    for line in seen:
        print("   ", line)

    print("\n== retry_call: non-retryable error is not retried ==")
    tries = {"n": 0}

    def broken() -> None:
        tries["n"] += 1
        raise ContractViolation("stage returned a str where a list was declared")

    try:
        retry_call(broken, attempts=5, base_ms=1.0)
    except ContractViolation as exc:
        print(f"  raised {type(exc).__name__} after {tries['n']} attempt(s) (expected 1)")

    print("\n== retry_call: retry refused when the deadline cannot fund the sleep ==")
    dl = Deadline(6.0, label="tiny")
    attempts_made = {"n": 0}

    def slow_flaky() -> None:
        attempts_made["n"] += 1
        time.sleep(0.004)
        raise TransientError("still flaky", stage="demo")

    try:
        retry_call(slow_flaky, attempts=5, base_ms=40.0, max_ms=80.0, deadline=dl)
    except TransientError:
        print(
            f"  gave up after {attempts_made['n']} attempt(s); "
            f"spend={dl.spend_fraction():.2f} remaining={dl.remaining_ms():.2f}ms"
        )

    print("\n== Deadline ==")
    d = Deadline(50.0)
    time.sleep(0.02)
    print(" ", d.to_dict())
    print("  child(30ms) budget:", round(d.child(30.0).budget_ms, 2), "ms")
    time.sleep(0.035)
    print("  expired now:", d.expired(), "overrun:", round(d.overrun_ms(), 2), "ms")
    try:
        d.check(stage="generate")
    except BudgetExceededError as exc:
        print(f"  check() -> {exc.code}: retryable={exc.retryable} degradable={exc.degradable}")

    print("\n== CircuitBreaker: 3 failures open it, cooldown half-opens it ==")
    cb = CircuitBreaker(threshold=3, cooldown_ms=25.0, half_open_probes=1)
    for i in range(3):
        cb.record_failure("stt:sarvam", TransientError("502"))
        print(f"  after failure {i + 1}: state={cb.state('stt:sarvam')} allow={cb.allow('stt:sarvam')}")
    try:
        cb.check("stt:sarvam")
    except CircuitOpenError as exc:
        print(f"  check() -> {exc.code} (degradable={exc.degradable}) -> stage falls back")
    time.sleep(0.03)
    print("  after cooldown: state =", cb.state("stt:sarvam"))
    print("  probe 1 admitted:", cb.allow("stt:sarvam"), "| probe 2 admitted:", cb.allow("stt:sarvam"))
    cb.record_success("stt:sarvam")
    print("  after a successful probe: state =", cb.state("stt:sarvam"))
    print("  snapshot:", cb.snapshot())

    print("\n== CircuitBreaker.guard is thread-safe (8 threads x 40 calls) ==")
    cb2 = CircuitBreaker(threshold=1000, cooldown_ms=1.0)
    errors: list[str] = []

    def worker() -> None:
        for i in range(40):
            try:
                with cb2.guard("shared"):
                    if i % 5 == 0:
                        raise TransientError("boom")
            except VaaniError:
                pass
            except Exception as exc:  # pragma: no cover - would be a real bug
                errors.append(repr(exc))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = cb2.snapshot()["shared"]
    print(
        f"  total_calls={snap['total_calls']} (expected 320) "
        f"failures={snap['total_failures']} (expected 64) unexpected={len(errors)}"
    )
