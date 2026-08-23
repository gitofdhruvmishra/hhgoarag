#!/usr/bin/env python3
"""Vaani entry point.

Usage:
    python3 run.py                          # uses defaults (port 7860, offline corpus)
    SARVAM_API_KEY=sk-... python3 run.py    # enables cloud STT
    VAANI_GENERATOR=llm python3 run.py      # LLM-enhanced answers (needs ANTHROPIC_API_KEY)
    VAANI_PORT=8080 python3 run.py          # different port

The server builds its index on first run (~5 seconds for the offline corpus)
and caches it under artifacts/index/. Subsequent starts are instant.
"""

from __future__ import annotations

import sys
import os


def _check_python() -> None:
    if sys.version_info < (3, 12):
        print(
            f"[vaani] Python 3.12+ required (found {sys.version.split()[0]}). "
            "Please upgrade: https://www.python.org/downloads/",
            file=sys.stderr,
        )
        sys.exit(1)


def _print_banner(cfg) -> None:
    from vaani.config import runtime_info
    info = runtime_info()
    width = 60
    print("\n" + "─" * width)
    print(f"  🎙️  Vaani v{__import__('vaani').__version__}  —  Voice-Native Multilingual RAG")
    print("─" * width)
    for line in cfg.summary_lines():
        print(f"  {line}")
    print("─" * width)
    print(f"  python  : {info['python']}  ({info['implementation']})")
    print(f"  server  : http://{cfg.server.host}:{cfg.server.port}")
    if not cfg.stt_status()["cloud_configured"]:
        print("  STT     : ⚡ browser Web Speech API  (no API key needed)")
    if not cfg.llm_available():
        print("  answers : extractive  (set ANTHROPIC_API_KEY for LLM mode)")
    print("─" * width + "\n")


def main() -> None:
    _check_python()

    # Import lazily so import errors surface clearly.
    from vaani.config import get_config, ensure_dirs

    cfg = get_config()
    ensure_dirs()
    _print_banner(cfg)

    # ------------------------------------------------------------------
    # Build index if needed
    # ------------------------------------------------------------------
    from vaani.config import INDEX_DIR
    index_ready = (INDEX_DIR / "meta.json").exists()

    if not index_ready:
        print("[vaani] Building index … (first run only, ~5 s for offline corpus)")
        try:
            from vaani._bootstrap import build_index
            build_index(cfg)
            print("[vaani] Index built. ✓\n")
        except ImportError:
            # Bootstrap module not yet wired; gracefully skip.
            print("[vaani] No bootstrap module found — skipping index build.\n")
        except Exception as exc:
            print(f"[vaani] Index build failed: {exc}", file=sys.stderr)
            print("[vaani] Continuing — retrieval will use in-memory fallback.\n", file=sys.stderr)

    # ------------------------------------------------------------------
    # Start server
    # ------------------------------------------------------------------
    try:
        from vaani._server import run
        run(cfg)
    except ImportError:
        _run_minimal_server(cfg)


def _run_minimal_server(cfg) -> None:
    """Fallback: serve the static docs site when the full server isn't wired."""
    import http.server
    import threading
    import webbrowser
    from pathlib import Path

    static_dir = Path(__file__).parent / "vaani" / "server" / "static"
    docs_dir = Path(__file__).parent / "docs"

    # Prefer the built UI; fall back to docs.
    serve_dir = static_dir if any(static_dir.iterdir()) else docs_dir

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def log_message(self, fmt, *args):
            if cfg.server.request_log:
                print(f"  [http] {fmt % args}")

    host, port = cfg.server.host, cfg.server.port
    server = http.server.HTTPServer((host, port), Handler)

    url = f"http://{host}:{port}"
    print(f"[vaani] Demo server running → {url}")
    print("[vaani] Press Ctrl+C to stop.\n")

    if cfg.server.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[vaani] Stopped.")


if __name__ == "__main__":
    main()
