"""Error taxonomy for the Vaani harness.

The orchestrator's retry policy is driven entirely by exception *type*, not by
string matching on messages. Every error therefore declares two things:

* ``retryable`` -- whether re-running the same stage with the same input could
  plausibly succeed. Network blips are retryable; a malformed contract is not.
* ``degradable`` -- whether the orchestrator is allowed to fall back to a
  lower-quality path (a secondary STT provider, lexical-only retrieval) and
  carry on rather than failing the request.

This is the difference between a harness and a try/except: the failure mode is
part of the type system, so the orchestration logic stays declarative.
"""

from __future__ import annotations

from typing import Any


class VaaniError(Exception):
    """Base class for everything this system raises deliberately."""

    retryable: bool = False
    degradable: bool = False
    code: str = "vaani_error"
    http_status: int = 500

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "degradable": self.degradable,
            "context": self.context,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.context:
            extras = " ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} ({extras})"
        return self.message


# --------------------------------------------------------------------------- #
# Configuration and contract errors -- never retried, never degraded.
# --------------------------------------------------------------------------- #


class ConfigError(VaaniError):
    code = "config_error"
    http_status = 500


class ContractViolation(VaaniError):
    """A stage returned data that failed its declared contract."""

    code = "contract_violation"
    http_status = 500


class ValidationError(VaaniError):
    """Caller sent us something we cannot act on."""

    code = "validation_error"
    http_status = 400


# --------------------------------------------------------------------------- #
# Index lifecycle
# --------------------------------------------------------------------------- #


class IndexNotBuiltError(VaaniError):
    code = "index_not_built"
    http_status = 503


class IndexCorruptError(VaaniError):
    code = "index_corrupt"
    http_status = 500


class DatasetError(VaaniError):
    code = "dataset_error"
    http_status = 500


# --------------------------------------------------------------------------- #
# Stage failures -- these drive the retry / degrade machinery.
# --------------------------------------------------------------------------- #


class StageError(VaaniError):
    """A pipeline stage failed."""

    code = "stage_error"

    def __init__(self, message: str, stage: str = "", **context: Any) -> None:
        super().__init__(message, stage=stage, **context)
        self.stage = stage


class StageTimeout(StageError):
    """A stage exceeded its declared deadline."""

    code = "stage_timeout"
    retryable = True
    degradable = True
    http_status = 504


class TransientError(StageError):
    """Something flaky and external. Retry it."""

    code = "transient_error"
    retryable = True
    degradable = True
    http_status = 503


class CircuitOpenError(StageError):
    """The circuit breaker for this stage is open; we did not even try."""

    code = "circuit_open"
    retryable = False
    degradable = True
    http_status = 503


class BudgetExceededError(StageError):
    """The request's total latency budget is spent; stop doing optional work."""

    code = "budget_exceeded"
    retryable = False
    degradable = True
    http_status = 504


# --------------------------------------------------------------------------- #
# Speech to text
# --------------------------------------------------------------------------- #


class STTError(StageError):
    code = "stt_error"
    http_status = 502


class STTUnavailable(STTError):
    """No configured provider can serve this request."""

    code = "stt_unavailable"
    retryable = False
    degradable = True


class STTRateLimited(STTError):
    code = "stt_rate_limited"
    retryable = True
    degradable = True
    http_status = 429


class STTAuthError(STTError):
    """Bad or missing credentials. Retrying will not help; degrade instead."""

    code = "stt_auth_error"
    retryable = False
    degradable = True
    http_status = 401


class AudioDecodeError(STTError):
    code = "audio_decode_error"
    retryable = False
    degradable = False
    http_status = 400


# --------------------------------------------------------------------------- #
# Retrieval and generation
# --------------------------------------------------------------------------- #


class RetrievalError(StageError):
    code = "retrieval_error"
    degradable = True


class EmbeddingError(StageError):
    code = "embedding_error"
    degradable = True


class GenerationError(StageError):
    code = "generation_error"
    retryable = True
    degradable = True


class LLMError(GenerationError):
    code = "llm_error"
    http_status = 502


# --------------------------------------------------------------------------- #
# Guardrails. Being blocked is a *successful* outcome of the pipeline, not an
# exception -- so GuardrailTripped exists only for the rare case where a rail
# must hard-stop execution mid-flight (unsafe input, never generate anything).
# --------------------------------------------------------------------------- #


class GuardrailTripped(VaaniError):
    code = "guardrail_tripped"
    http_status = 200  # We answer the caller; we just decline to answer the question.

    def __init__(self, message: str, verdict: Any = None, **context: Any) -> None:
        super().__init__(message, **context)
        self.verdict = verdict


class ToolError(VaaniError):
    """A registered tool failed or was called with bad arguments."""

    code = "tool_error"

    def __init__(self, message: str, tool: str = "", **context: Any) -> None:
        super().__init__(message, tool=tool, **context)
        self.tool = tool


class ToolNotFound(ToolError):
    code = "tool_not_found"
    http_status = 400


def classify(exc: BaseException) -> tuple[bool, bool]:
    """Return ``(retryable, degradable)`` for any exception.

    Non-Vaani exceptions are treated conservatively: OS/network-level errors are
    retryable, everything else is a bug and is not.
    """
    if isinstance(exc, VaaniError):
        return exc.retryable, exc.degradable
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True, True
    return False, False


__all__ = [
    "AudioDecodeError",
    "BudgetExceededError",
    "CircuitOpenError",
    "ConfigError",
    "ContractViolation",
    "DatasetError",
    "EmbeddingError",
    "GenerationError",
    "GuardrailTripped",
    "IndexCorruptError",
    "IndexNotBuiltError",
    "LLMError",
    "RetrievalError",
    "STTAuthError",
    "STTError",
    "STTRateLimited",
    "STTUnavailable",
    "StageError",
    "StageTimeout",
    "ToolError",
    "ToolNotFound",
    "TransientError",
    "VaaniError",
    "ValidationError",
    "classify",
]
