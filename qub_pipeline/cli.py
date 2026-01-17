#!/usr/bin/env python
"""Command-line interface for the QUB transit photometry pipeline.

Installed entrypoint: `qub-pipeline`
"""

# qub_pipeline/cli.py
from __future__ import annotations

import argparse
from pathlib import Path

from qub_pipeline.runner import main as runner_main
from qub_pipeline.reduction import main as reduction_main
from qub_pipeline.photometry import main as photometry_main


def _default_config_candidates() -> list[Path]:
    cwd = Path.cwd()
    return [
        cwd / "config.json",
        cwd / "config" / "config.json",
    ]


def _resolve_default_config_path() -> str | None:
    for p in _default_config_candidates():
        if p.exists():
            return str(p)
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qub-pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- run (full pipeline) ----
    pr = sub.add_parser("run", help="Run reduction + photometry")
    pr.add_argument("--config", default=_resolve_default_config_path(), help="Path to config JSON")
    pr.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    pr.add_argument("--science-dir", default=None, help="Override science directory")
    pr.add_argument("--dark-dir", default=None, help="Override dark directory")
    pr.add_argument("--outdir", default=None, help="Override output directory")
    def _runner_argv(a):
        argv = []
        if a.config:
            argv += ["--config", a.config]
        if a.verbose:
            argv += ["--verbose"]
        if a.science_dir:
            argv += ["--science-dir", a.science_dir]
        if a.dark_dir:
            argv += ["--dark-dir", a.dark_dir]
        if a.outdir:
            argv += ["--outdir", a.outdir]
        return argv
    
    pr.set_defaults(func=lambda a: runner_main(_runner_argv(a)))

    # ---- reduction only ----
    prd = sub.add_parser("reduction", help="Run reduction only")
    prd.add_argument("--config", default=_resolve_default_config_path(), help="Path to config JSON")
    prd.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    prd.set_defaults(func=lambda a: reduction_main(config_path=a.config, verbose=a.verbose))

    # ---- photometry only ----
    pp = sub.add_parser("photometry", help="Run photometry only")
    pp.add_argument("--config", default=_resolve_default_config_path(), help="Path to config JSON")
    pp.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    pp.set_defaults(func=lambda a: photometry_main(config_path=a.config, verbose=a.verbose))

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
