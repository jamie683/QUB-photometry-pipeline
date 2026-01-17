from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `updates` into `base` and return the merged dict (in-place)."""
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def default_config_path(script_file: str | os.PathLike, filename: str = "config.json") -> Path:
    """
    Choose a sensible default config path.

    Search order:
      1) ./config.json (current working directory)
      2) ./config/config.json (common repo layout)
      3) config.json beside the calling script/module (backward-compatible)
    Returns the first existing path; if none exist, returns ./config.json.
    """
    candidates = [
        Path.cwd() / filename,
        Path.cwd() / "config" / filename,
        Path(script_file).resolve().with_name(filename),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]



def load_config(
    config_path: Optional[Union[str, Path]] = None,
    script_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Load JSON config.

    Resolution order:
      1) explicit config_path, if provided (must exist)
      2) ./config.json
      3) ./config/config.json
      4) config.json next to script_file (if provided)
    If none found, return {}.
    """
    # 1) explicit
    if config_path:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        return json.loads(p.read_text(encoding="utf-8"))

    # 2) cwd candidates
    cwd = Path.cwd()
    candidates = [cwd / "config.json", cwd / "config" / "config.json"]

    # 3) script-adjacent candidate
    if script_file:
        sf = Path(script_file)
        # if a module file path is passed, use its directory
        candidates.append(sf.resolve().parent / "config.json")

    for p in candidates:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))

    return {}



def sha1_file(path: str | os.PathLike) -> str:
    p = Path(path)
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def collect_versions(packages: list[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in packages:
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", None)
            if ver is None and hasattr(mod, "version"):
                ver = getattr(mod.version, "__version__", None)
            out[name] = str(ver) if ver is not None else "unknown"
        except Exception:
            out[name] = "not_installed"
    return out


def setup_logging(
    log_file: Union[str, Path],
    *,
    verbose: bool = False,
    logger_name: str = "qub_pipeline",
) -> logging.Logger:
    """
    Create/return a logger that writes to both console and a file.

    Parameters
    ----------
    log_file : str | Path
        Path to the log file.
    verbose : bool
        If True, console level is DEBUG; otherwise INFO.
    logger_name : str
        Name passed to logging.getLogger.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    name = str(logger_name) if logger_name is not None else "qub_pipeline"
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if called multiple times
    if getattr(logger, "_qub_configured", False):
        return logger

    logger.setLevel(logging.DEBUG)  # keep file at DEBUG always
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # File handler (always DEBUG)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler (INFO or DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger._qub_configured = True
    return logger


def write_json(path: str | os.PathLike, obj: Any) -> None:
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def run_metadata_base(invocation: str) -> Dict[str, Any]:
    return {
        "timestamp_utc": utc_now_iso(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "invocation": invocation,
    }
