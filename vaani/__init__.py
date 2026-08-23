"""Vaani -- a voice-native Retrieval-Augmented Generation platform.

Pipeline shape:

    voice -> speech-to-text -> guardrails(in) -> query transform -> hybrid
    retrieval over a multi-strategy chunk index -> rerank -> grounded answer
    generation -> guardrails(out)

Two properties shape every design decision in this package:

1. **Zero dependencies.** Nothing here imports anything outside the Python
   standard library. There is no pip install, no model download, no build step.
   ``python3 run.py`` works on a clean machine. The dense retriever is built
   from a PPMI co-occurrence matrix projected with random indexing, and dot
   products run through ``math.sumprod`` at C speed, which is what makes a pure
   stdlib vector search viable.

2. **A 200 ms budget.** The orchestrator tracks elapsed time against the budget
   and sheds optional work (query expansion, reranking, MMR diversification)
   before it blows through it, rather than after. Retrieval is two-stage --
   sparse candidate generation then dense scoring over candidates only -- so
   query cost grows with candidate count, not corpus size.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
