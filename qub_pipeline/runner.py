# --- Overview ---
"""
run_pipeline.py
Single entry point for QUB reduction -> photometry.

Design:
- config.json is the single source of truth.
- CLI flags are OPTIONAL overrides only.
- No "effective"/derived config JSON files are written.

Typical use:
  python run_pipeline.py
  python run_pipeline.py --config path/to/config.json
  python run_pipeline.py --out-root D:\runs --run-id test1
"""


# --- Library Imports ---
from __future__ import annotations


from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qub_pipeline.utils import (
    ensure_dir,
    load_config,
    setup_logging,
)

import argparse
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from importlib import metadata as importlib_metadata



DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def safe_version(dist_name: str) -> str | None:
    try:
        return importlib_metadata.version(dist_name)
    except Exception:
        return None


def write_run_metadata(outdir: Path, config_path: Path | None, cfg: dict, argv: list[str]) -> Path:
    """Write run_metadata.json to the run directory."""
    meta = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "platform": {
            "python": sys.version.split()[0],
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "invocation": " ".join(argv),
        "config": {
            "path": str(config_path) if config_path else None,
            "sha1": None,
        },
        "packages": {
            "numpy": safe_version("numpy"),
            "astropy": safe_version("astropy"),
            "photutils": safe_version("photutils"),
            "scipy": safe_version("scipy"),
            "matplotlib": safe_version("matplotlib"),
            "pandas": safe_version("pandas"),
            "batman-package": safe_version("batman-package"),
            "emcee": safe_version("emcee"),
            "corner": safe_version("corner"),
            "tqdm": safe_version("tqdm"),
        },
        "paths": {
            "run_dir": str(outdir),
        },
        "config_snapshot": cfg,
    }

    # Hash config file contents if present
    if config_path and config_path.exists():
        import hashlib
        meta["config"]["sha1"] = hashlib.sha1(config_path.read_bytes()).hexdigest()

    out = outdir / "run_metadata.json"
    out.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return out


def resolve_outdirs(cfg: dict, *, cli_out_root: str | None, cli_run_id: str | None, cli_no_nest: bool) -> tuple[Path, Path, Path]:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    target = cfg.get("target", {}) if isinstance(cfg.get("target"), dict) else {}
    reduction = cfg.get("reduction", {}) if isinstance(cfg.get("reduction"), dict) else {}
    phot = cfg.get("photometry", {}) if isinstance(cfg.get("photometry"), dict) else {}

    out_root = Path(cli_out_root or paths.get("out_root", ".")).expanduser().resolve()

    nest = bool(paths.get("nest_run_dir", True))
    if cli_no_nest:
        nest = False

    run_id = cli_run_id or paths.get("run_id")
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    target_name = (target.get("name") or "target").strip().replace(" ", "_")

    if nest:
        base_out = out_root / target_name / run_id
    else:
        base_out = out_root

    reduction_sub = reduction.get("outdir", "reduction")
    phot_sub = phot.get("outdir", "photometry")

    reduction_out = ensure_dir(base_out / reduction_sub)
    photometry_out = ensure_dir(base_out / phot_sub)
    return base_out, reduction_out, photometry_out

def pick_script(path_in_cfg: str | None, fallback: Path) -> Path:
    if path_in_cfg:
        return Path(path_in_cfg).expanduser().resolve()
    return fallback.expanduser().resolve()

def run_cmd(cmd: list[str], *, cwd: Path, logger: logging.Logger) -> None:
    logger.info("▶ Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run QUB reduction + photometry using a JSON config (config-first).")

    ap.add_argument("--config", default=None, help="Path to JSON config. Defaults to ./config.json beside this script.")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to use for subprocess calls.")

    # Optional overrides (these do NOT write any new configs; they only affect this run)
    ap.add_argument("--reduction-script", default=None, help="Override scripts.reduction_script.")
    ap.add_argument("--photometry-script", default=None, help="Override scripts.photometry_script.")
    ap.add_argument("--science-dir", default=None, help="Override reduction.science_dir.")
    ap.add_argument("--dark-dir", default=None, help="Override reduction.dark_dir.")
    ap.add_argument("--regions", default=None, help="Override photometry.regions_path.")
    ap.add_argument("--out-root", default=None, help="Override paths.out_root.")
    ap.add_argument("--run-id", default=None, help="Override paths.run_id.")
    ap.add_argument("--no-nest", action="store_true", help="Disable nesting even if config paths.nest_run_dir is true.")
    ap.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = ap.parse_args(argv)


    # If the default config.json (alongside this script) doesn't exist and the user did not explicitly
    # provide --config, treat config as empty instead of erroring.
    _default_cfg = Path(DEFAULT_CONFIG_PATH)
    if args.config == str(_default_cfg) and not _default_cfg.exists():
        args.config = None
    cfg: dict = {}
    cfg_path: Path | None = None
    if args.config is not None:
        cfg_path = Path(args.config).expanduser().resolve()
        cfg = load_config(cfg_path)

    # Apply CLI overrides into cfg IN-MEMORY (no file writes)
    if args.out_root:
        cfg.setdefault("paths", {})["out_root"] = args.out_root
    if args.run_id:
        cfg.setdefault("paths", {})["run_id"] = args.run_id
    if args.science_dir:
        cfg.setdefault("reduction", {})["science_dir"] = args.science_dir
    if args.dark_dir:
        cfg.setdefault("reduction", {})["dark_dir"] = args.dark_dir
    if args.regions:
        cfg.setdefault("photometry", {})["regions_path"] = args.regions

    # Resolve outdirs
    base_out, reduction_out, photometry_out = resolve_outdirs(
        cfg, cli_out_root=args.out_root, cli_run_id=args.run_id, cli_no_nest=args.no_nest
    )

    logger = setup_logging(Path(base_out) / "pipeline.log", verbose=args.verbose)
    logger.info("Run directory: %s", base_out)

    # Write metadata snapshot for reproducibility
    try:
        meta_path = write_run_metadata(Path(base_out), cfg_path, cfg, sys.argv)
        logger.info("Wrote %s", meta_path.name)
    except Exception as e:
        logger.warning("Could not write run_metadata.json: %s", e)

    here = Path(__file__).parent.resolve()

    scripts = cfg.get("scripts", {}) if isinstance(cfg.get("scripts"), dict) else {}
    reduction_script = pick_script(args.reduction_script or scripts.get("reduction_script"), here / "qub_reduction.py")
    photometry_script = pick_script(args.photometry_script or scripts.get("photometry_script"), here / "qub_photometry.py")

    py = args.python

    # Resolve regions path (optional)
    phot_cfg = cfg.get("photometry", {}) if isinstance(cfg.get("photometry"), dict) else {}
    regions_path = phot_cfg.get("regions_path") or phot_cfg.get("regions")

    # -- Run reduction --
    red_cmd = [
        py, str(reduction_script),
        "--outdir", str(reduction_out),
    ]
    if cfg_path is not None:
        red_cmd += ["--config", str(cfg_path)]
    # Only pass dirs if explicitly present (script also reads from config)
    red = cfg.get("reduction", {}) if isinstance(cfg.get("reduction"), dict) else {}
    if red.get("science_dir"):
        red_cmd += ["--science-dir", str(red["science_dir"])]
    if red.get("dark_dir"):
        red_cmd += ["--dark-dir", str(red["dark_dir"])]

    run_cmd(red_cmd, cwd=here, logger=logger)

    # Prefer explicit products.json from reduction step (avoids glob guessing)
    products_path = Path(reduction_out) / "products.json"
    cube_path = None
    if products_path.exists():
        try:
            with open(products_path, "r", encoding="utf-8") as f:
                products = json.load(f)
            cand = products.get("cube_fits") or products.get("cube")
            if cand:
                cube_path = Path(cand)
        except Exception as e:
            logger.warning("Could not read products.json (%s): %s", products_path, e)

    if cube_path is None:
        # Fallback: reduction writes cube to outdir/cube_name
        cube_name = red.get("cube_name", "QUB_cube.fits")
        cube_path = Path(reduction_out) / cube_name
        if not cube_path.exists():
            # Best-effort search
            cands = list(Path(reduction_out).glob("*.fits")) + list(Path(reduction_out).glob("*.fit*"))
            cube_path = cands[0] if cands else None

    if cube_path is None or not Path(cube_path).exists():
        raise FileNotFoundError(f"Could not find cube output in {reduction_out} (expected products.json or a FITS cube)")
    phot_cmd = [
        py, str(photometry_script),
        "--cube", str(cube_path),
        "--outdir", str(photometry_out),
    ]
    if cfg_path is not None:
        phot_cmd += ["--config", str(cfg_path)]
    if regions_path:
        phot_cmd += ["--regions", str(regions_path)]

    run_cmd(phot_cmd, cwd=here, logger=logger)

    logger.info("✅ Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())