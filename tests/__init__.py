"""Unit tests (standard-library ``unittest``).

* ``tests.core``: the production package, ``asah_segmentation``. Runs on a
  production-only install (``uv sync --no-dev``).
* ``tests.dev``: the development tooling, ``eval/`` (evaluation CLI and
  metrics) and ``scripts/`` (legacy checkpoint conversion).

Run from the project root:

    uv run python -m unittest                                  # everything
    uv run --no-dev python -m unittest discover -s tests/core -t .   # core only
"""
