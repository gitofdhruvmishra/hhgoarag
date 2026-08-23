# 🎙️ Vaani — Voice-Native Multilingual RAG

> **Zero dependencies · 200 ms budget · 10 Indic languages · Pure Python stdlib**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](#zero-dependencies)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/demo-Vercel%20Live-brightgreen.svg)](https://hhgoarag-gules.vercel.app)
[![Demo Docs](https://img.shields.io/badge/docs-GitHub%20Pages-orange.svg)](https://gitofdhruvmishra.github.io/hhgoarag/)

Vaani is a production-quality **Retrieval-Augmented Generation (RAG)** pipeline built entirely on the Python standard library — no pip installs, no model downloads, no build steps. Speak a question in English, Hindi, Bengali, Tamil, Telugu, or 6 more Indic languages; get a grounded, cited answer in under 200 ms.

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| **Zero dependencies** | Runs with `python3 run.py` on a clean machine |
| **200 ms pipeline budget** | Sheds optional stages (query expansion, reranking, MMR) before blowing the budget |
| **7 chunking strategies** | Fixed, recursive, sentence window, semantic, proposition, metadata-aware, hierarchical |
| **Hybrid retrieval** | BM25 sparse + PPMI random-index dense + optional rerank, fused with RRF |
| **Indic multilingual** | 10 languages: en, hi, bn, ta, te, mr, gu, kn, ml, pa |
| **Voice input** | Sarvam AI / ElevenLabs STT, or browser Web Speech API (zero-key demo path) |
| **Full guardrail stack** | Input safety · prompt injection · PII redaction · off-topic detection · hallucination check |
| **Corpus: MSMARCO-XI** | ai4bharat/MSMARCO-XI — Indic multilingual passage retrieval dataset |

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/gitofdhruvmishra/hhgoarag.git
cd hhgoarag

# Run — no pip install needed!
python3 run.py
# → Opens http://127.0.0.1:7860 in your browser
```

### Optional: API Keys

```bash
# For cloud STT (Sarvam AI)
export SARVAM_API_KEY=your_key_here

# For ElevenLabs STT
export ELEVENLABS_API_KEY=your_key_here

# For LLM-enhanced answers (Claude)
export ANTHROPIC_API_KEY=your_key_here
```

> **Without any keys**, Vaani uses the browser's built-in Web Speech API for voice and extractive generation for answers — the full demo works with zero configuration.

---

## 🗺️ Pipeline Architecture

```
┌──────────────┐    ┌────────────┐    ┌──────────────────┐
│  Voice Input │───▶│    STT     │───▶│  Input Guardrail │
│  (mic/text)  │    │ Sarvam/    │    │  safety·inject·  │
└──────────────┘    │ ElevenLabs │    │  PII·length      │
                    │ /browser   │    └────────┬─────────┘
                    └────────────┘             │
                                    ┌──────────▼──────────┐
                                    │   Query Transform   │
                                    │  expand·repair·ASR  │
                                    └──────────┬──────────┘
                                               │
                          ┌────────────────────▼──────────────────────┐
                          │              Hybrid Retrieval              │
                          │  BM25 sparse → candidates → dense score   │
                          │  RRF fusion → rerank → MMR diversify      │
                          └────────────────────┬──────────────────────┘
                                               │
                          ┌────────────────────▼──────────────────────┐
                          │          Answer Generation                 │
                          │  extractive (default) | LLM (opt-in)      │
                          │  cited · grounded · sentence-scored        │
                          └────────────────────┬──────────────────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │  Output Guardrail   │
                                    │  grounding·entity·  │
                                    │  number·confidence  │
                                    └─────────────────────┘
```

---

## 📁 Project Structure

```
hhgoarag/
├── run.py                    # Entry point — starts the server
├── vaani/
│   ├── __init__.py           # Package docstring & version
│   ├── config.py             # All config dataclasses, env-var overlay
│   ├── textkit.py            # Script-aware tokenizer, normalizer, NLP utils
│   ├── embed.py              # PPMI + random-index dense embedder (no model files)
│   ├── chunking/             # 7 chunking strategy implementations
│   ├── retrieval/            # BM25, dense search, RRF fusion, MMR, reranker
│   ├── generation/           # Extractive + LLM answer generators
│   ├── guardrails/           # Input/output safety rails
│   ├── stt/                  # Sarvam + ElevenLabs + browser STT clients
│   ├── vectordb/             # In-process IVF vector store
│   ├── harness/              # Orchestrator, contracts, errors, tracing, retry
│   ├── server/               # HTTP server + static UI
│   ├── bench/                # Latency benchmarking harness
│   └── data/corpus/          # Offline MSMARCO-XI corpus subset
├── artifacts/                # Built index files (gitignored)
├── tests/                    # Test suite
└── docs/                     # GitHub Pages site
```

---

## ⚙️ Configuration

Everything is a dataclass with sane defaults. Override with environment variables:

| Variable | Default | Description |
|---|---|---|
| `SARVAM_API_KEY` | *(none)* | Sarvam AI STT key |
| `ELEVENLABS_API_KEY` | *(none)* | ElevenLabs STT key |
| `ANTHROPIC_API_KEY` | *(none)* | Claude LLM key (for enriched answers) |
| `VAANI_PORT` | `7860` | Server port |
| `VAANI_GENERATOR` | `extractive` | `extractive` or `llm` |
| `VAANI_DATASET_SOURCE` | `auto` | `auto` / `hf` / `offline` |
| `VAANI_BUDGET_MS` | `200` | Pipeline latency budget |
| `VAANI_GUARDRAILS` | `true` | Enable/disable guardrail stack |

---

## 🔬 Technical Highlights

### Dense Embeddings Without Model Files
No sentence-transformers. No ONNX. Vaani earns semantics **from the corpus itself** using:
1. **Co-occurrence** — slide a window, count which terms appear near which
2. **PPMI** — Positive PMI with Levy & Goldberg context-distribution smoothing
3. **Random indexing** — Johnson-Lindenstrauss projection using seeded term hashes → no matrix to store or ship

### 200 ms Budget Enforcement
The orchestrator tracks elapsed time against the budget and **proactively sheds** optional stages before they blow it — query expansion, reranking, MMR — not after.

### Indic Multilingual Tokenization
A script-aware tokenizer classifies each character by Unicode script (Devanagari, Bengali, Tamil, Telugu, CJK, Latin…) without any per-language model. CJK is handled by character + bigram indexing.

---

## 📖 Documentation

Full documentation available at **[gitofdhruvmishra.github.io/hhgoarag](https://gitofdhruvmishra.github.io/hhgoarag/)**

---

## 📄 License

MIT © 2026
