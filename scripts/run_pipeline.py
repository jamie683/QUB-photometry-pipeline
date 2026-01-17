#!/usr/bin/env python
"""Thin wrapper for backwards compatibility.

Prefer running via the installed CLI:
  qub-pipeline run --config ...
"""
from qub_pipeline.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
