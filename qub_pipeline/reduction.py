# QUB Photometric Data Reduction Pipeline

# --- Overview ---
"""QUB data reduction: calibrate frames and build a science cube.

This script is intended to be called by scripts/run_pipeline.py.
"""

# --- Library Imports ---
from __future__ import annotations


from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qub_pipeline.utils import (
    deep_update,
    load_config,
    setup_logging,
)

from pathlib import Path
import json
import argparse
import csv
import math
import re
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
from tqdm.auto import tqdm
import numexpr as ne
from astropy.stats import sigma_clip

# Default: config.json alongside this script
DEFAULT_CONFIG_PATH = Path(__file__).with_name('config.json')


def _cfg_get(cfg: dict, path: str, default=None):
    """Get nested config value by dotted path, e.g. 'site.lat_deg'."""
    if not isinstance(cfg, dict):
        return default
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def resolve_airmass_inputs(args, cfg: dict):
    """
    Decide target/site values from CLI (preferred) or config (fallback).
    Returns (ra, dec, lat, lon, height) where any may be None if unavailable.

    Reduction can run without these; they are only needed to compute AltAz airmass and BJD_TDB.
    """
    ra  = getattr(args, "target_ra", None)  or _cfg_get(cfg, "target.ra")
    dec = getattr(args, "target_dec", None) or _cfg_get(cfg, "target.dec")

    lat = getattr(args, "obs_lat", None)
    lon = getattr(args, "obs_lon", None)
    hgt = getattr(args, "obs_height", None)

    if lat is None: lat = _cfg_get(cfg, "site.lat_deg", None)
    if lon is None: lon = _cfg_get(cfg, "site.lon_deg", None)
    if hgt is None: hgt = _cfg_get(cfg, "site.height_m", 0.0)

    return ra, dec, (None if lat is None else float(lat)), (None if lon is None else float(lon)), float(hgt or 0.0)

def get_exptime_from_header(hdr, preferred_key=None):
    keys = []
    if preferred_key:
        keys.append(str(preferred_key).strip())
    keys += ["EXPTIME", "EXPOSURE", "ITIME", "EXPOS", "EXP_TIME", "DARKTIME", "ELAPTIME"]
    for key in keys:
        if key in hdr:
            try:
                return float(hdr[key])
            except Exception:
                pass
    return float("nan")


def is_bias_frame(hdr, exptime: float, *, max_exptime: float = 0.1) -> bool:
    """Heuristic bias-frame classifier (INT-style).

    Bias frames are typically zero/very-short exposure images with header types like BIAS/zero.
    We intentionally keep this permissive so a 'dark' folder containing mixed BIAS+DARK still works.
    """
    try:
        exp = float(exptime)
    except Exception:
        exp = float('nan')

    # Header-based classification (common keywords across instruments)
    type_keys = ("IMAGETYP", "IMAGETYPE", "OBSTYPE", "FRAME", "FRAMETYP")
    typ = ""
    for k in type_keys:
        if k in hdr:
            try:
                typ = str(hdr[k]).strip().upper()
                break
            except Exception:
                pass

    if typ:
        if "BIAS" in typ or "ZERO" in typ:
            return True
        # Guard: don't misclassify explicit DARK/SCIENCE/FLAT as bias
        if "DARK" in typ or "FLAT" in typ or "SCI" in typ or "LIGHT" in typ:
            return False

    # Exposure-based fallback (covers missing/dirty headers)
    if np.isfinite(exp) and exp <= float(max_exptime):
        return True

    return False

def get_airmass_from_header(hdr) -> float:
    """Return airmass from a variety of common header keywords; NaN if unavailable."""
    for key in ("AIRMASS", "AIRMASS1", "SECZ", "AMSTART", "AMEND"):
        if key in hdr:
            try:
                return float(hdr[key])
            except Exception:
                pass
    return float("nan")

def parse_target_skycoord(ra_str: str, dec_str: str) -> SkyCoord:
    ra_str = str(ra_str).strip()
    dec_str = str(dec_str).strip()

    # Sexagesimal if contains ":"
    if ":" in ra_str:
        return SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg), frame="icrs")

    # Otherwise assume degrees
    ra_deg = float(ra_str)
    dec_deg = float(dec_str)
    return SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")

def compute_airmass(date_obs_utc: str, ra_str: str, dec_str: str,
                    lat_deg: float, lon_deg: float, height_m: float) -> float:
    """
    Compute airmass (sec(z)) from UTC time, target coords, and observatory location.
    Returns np.nan if it fails.
    """
    try:
        t = Time(date_obs_utc, format="isot", scale="utc")
    except Exception:
        try:
            t = Time(date_obs_utc, format="iso", scale="utc")
        except Exception:
            return float("nan")

    try:
        loc = EarthLocation(lat=lat_deg*u.deg, lon=lon_deg*u.deg, height=height_m*u.m)
        target = parse_target_skycoord(ra_str, dec_str)
        altaz = target.transform_to(AltAz(obstime=t, location=loc))
        
        # sec(z) = 1/cos(z) where z = 90-alt. astropy gives secz directly.
        am = altaz.secz.value
        if not np.isfinite(am) or am <= 0:
            return float("nan")
        return float(am)
    except Exception:
        return float("nan")

def parse_time_utc(date_obs: str, *, location: EarthLocation | None = None) -> Time:
    """Parse DATE-OBS robustly (supports 'isot' and 'iso')."""
    # Astropy can often infer, but we try a couple common formats explicitly
    for fmt in ("isot", "iso"):
        try:
            return Time(date_obs, format=fmt, scale="utc", location=location)
        except Exception:
            continue
    # Last resort: let Time try to parse
    return Time(date_obs, scale="utc", location=location)

def bias_read_noise(bias_files, gain: float, box_size=200):
    """
    Estimate read noise from a stack of bias frames.
    
    Parameters
    ----------
    bias_files : list[Path]
        Paths to bias FITS files.
    gain : float
        Gain in e-/ADU.
    box_size : int
        Size of central box to use for statistics.
    
    Returns
    -------
    sigma_bias_adu : float
        Median per-pixel std in ADU.
    read_noise_e : float
        Read noise in electrons.
    """
    bias_imgs = []
    
    for f in bias_files:
        with fits.open(f, memmap=False) as hdul:
            image = hdul[0].data.astype(np.float32)
        
        # Crop to central box
        ny, nx = image.shape
        b = min(box_size, ny, nx)
        x0 = (nx - b) // 2
        y0 = (ny - b) // 2
        img_sub = image[y0:y0 + b, x0:x0 + b]
        
        bias_imgs.append(img_sub)
    
    # Stack cropped bias frames
    bias_cube = np.stack(bias_imgs, axis=0)
    
    # Sigma-clip along the stack axis (axis=0), then compute std ignoring clipped points
    clipped = sigma_clip(bias_cube, sigma=5.0, maxiters=5, axis=0)
    sigma_map_adu = np.nanstd(clipped.filled(np.nan), axis=0, ddof=1)
    
    # Use median (robust to outliers)
    sigma_bias_adu = np.median(sigma_map_adu)
    
    # Convert to electrons
    read_noise_e = sigma_bias_adu * gain
    
    return float(sigma_bias_adu), float(read_noise_e)


# NUMEXPR CONFIG
ne.set_num_threads(1)

def clean_path(s: str) -> str:
    """Sanitise user-supplied paths.

    Handles:
      - surrounding quotes
      - accidental nested quotes
      - file:///C:/... URLs (Spyder/Jupyter sometimes emits these)
      - forward slashes on Windows
    """
    s = str(s).strip()

    # Strip surrounding quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # Remove any remaining stray quotes
    s = s.replace('"', "").replace("'", "")

    # Convert file:// URLs to local paths
    if s.lower().startswith("file:///"):
        s = s[8:]  # drop 'file:///'
    elif s.lower().startswith("file://"):
        s = s[7:]  # drop 'file://'

    # On Windows, normalise leading slash in /C:/...
    if re.match(r"^/[A-Za-z]:/", s):
        s = s[1:]

    # Convert forward slashes to backslashes for Windows-style paths
    s = s.replace("/", "\\")
    return s

def compute_bjd_tdb(date_obs: str, *, target: SkyCoord | None, location: EarthLocation | None) -> float:
    """Compute BJD_TDB from DATE-OBS (UTC) if target & location are available.
    Returns NaN if computation fails.
    """
    try:
        if target is None or location is None:
            return float("nan")
        t = parse_time_utc(date_obs, location=location)
        ltt = t.light_travel_time(target, kind="barycentric")
        bjd = (t.tdb + ltt).jd
        return float(bjd)
    except Exception:
        return float("nan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QUB dark-calibration + cube builder (INT-style manifest output).")
    p.add_argument("--science-dir", type=str, default=None, help="Root directory containing science FITS.")
    p.add_argument("--dark-dir", type=str, default=None, help="Root directory containing dark FITS.")
    p.add_argument("--bias-dir", type=str, default=None, help="Optional root directory containing bias FITS. If omitted, biases are auto-detected inside dark-dir (and ignored for dark selection).")
    p.add_argument("--bias-max-exptime", type=float, default=0.1, help="Max exposure time (s) to classify a frame as BIAS when auto-detecting biases.")
    p.add_argument("--outdir", type=str, default=None, help="Output directory.")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="JSON config file (site/target defaults, etc.).")
    p.add_argument("--cube-name", type=str, default="QUB_cube.fits", help="Output cube filename.")
    p.add_argument("--exptime", type=float, default=None, help="Science/dark exposure time to select (seconds).")
    p.add_argument("--exptime-tol", type=float, default=0.1, help="Exposure time tolerance (seconds).")

    # Optional: enable BJD_TDB in manifest
    p.add_argument("--target-ra", type=str, default=None, help="Target RA (e.g. '16:33:36.09').")
    p.add_argument("--target-dec", type=str, default=None, help="Target Dec (e.g. '+54:54:45.2').")
    p.add_argument("--obs-lat", type=float, default=None, help="Observatory latitude (deg).")
    p.add_argument("--obs-lon", type=float, default=None, help="Observatory longitude (deg, East +).")
    p.add_argument("--obs-height", type=float, default=0.0, help="Observatory height (m).")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return p.parse_args()

def main() -> None:
    args = parse_args()


    # If the default config.json (alongside this script) doesn't exist and the user did not explicitly
    # provide --config, treat config as empty instead of erroring.
    _default_cfg = Path(DEFAULT_CONFIG_PATH)
    if args.config == str(_default_cfg) and not _default_cfg.exists():
        args.config = None

    # - Apply JSON config defaults (CLI preferred) -
    cfg = load_config(args.config, __file__)
    cfg = deep_update(cfg, load_config(getattr(args, "config", None)))

    # Resolve target/site parameters (CLI preferred, config fallback).
    # These are optional; reduction can run without them unless you want computed airmass / BJD_TDB.
    target_ra, target_dec, obs_lat, obs_lon, obs_height = resolve_airmass_inputs(args, cfg)



    # - Fill remaining required inputs from config -
    # Support both modern keys (reduction.*) and legacy keys (paths.*)
    if args.science_dir is None:
        args.science_dir = _cfg_get(cfg, "reduction.science_dir", None) or _cfg_get(cfg, "paths.science_dir", None)
    if args.dark_dir is None:
        args.dark_dir = _cfg_get(cfg, "reduction.dark_dir", None) or _cfg_get(cfg, "paths.dark_dir", None)
    
    if args.bias_dir is None:
        args.bias_dir = _cfg_get(cfg, "reduction.bias_dir", None) or _cfg_get(cfg, "paths.bias_dir", None)

    if args.outdir is None:
        # Prefer explicit reduction.outdir; if relative and out_root exists, anchor it.
        out_root = _cfg_get(cfg, "paths.out_root", None)
        sub = _cfg_get(cfg, "reduction.outdir", None)
        if out_root and sub and not Path(str(sub)).expanduser().is_absolute():
            args.outdir = str(Path(str(out_root)).expanduser().resolve() / str(sub))
        elif sub:
            args.outdir = str(Path(str(sub)).expanduser().resolve())
        elif out_root:
            args.outdir = str(Path(str(out_root)).expanduser().resolve() / "reduction")
    
    if args.exptime is None:
        args.exptime = _cfg_get(cfg, "reduction.exptime", None)
    
    missing = []
    if args.science_dir is None: missing.append("reduction.science_dir (or paths.science_dir)")
    if args.dark_dir is None:    missing.append("reduction.dark_dir (or paths.dark_dir)")
    if args.outdir is None:      missing.append("reduction.outdir (or paths.out_root)")
    if args.exptime is None:     missing.append("reduction.exptime")
    
    if missing:
        raise ValueError("Missing required inputs. Provide via CLI or config.json: " + ", ".join(missing))

    science_root = Path(clean_path(args.science_dir)).expanduser().resolve()
    dark_root    = Path(clean_path(args.dark_dir)).expanduser().resolve()
    outdir       = Path(clean_path(args.outdir)).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(outdir / "reduction.log", verbose=bool(getattr(args, "verbose", False)))
    logger.info("Output directory: %s", outdir)

    output_cube = outdir / args.cube_name
    manifest_csv = outdir / "cube_manifest.csv"
    header_csv   = outdir / "Header_r.csv"  # Photometry script expects this name by default

    EXPTIME_TARGET = float(args.exptime)
    EXPTIME_TOL    = float(args.exptime_tol)

    # Optional BJD_TDB
    target = None
    location = None
    if target_ra and target_dec and (obs_lat is not None) and (obs_lon is not None):
        target = parse_target_skycoord(target_ra, target_dec)
        location = EarthLocation(lat=obs_lat * u.deg, lon=obs_lon * u.deg, height=obs_height * u.m)


    # 0.) Build master bias and master dark
    # --- master bias ---
    if not args.bias_dir:
        raise ValueError(
            "No bias_dir provided. Set it via --bias-dir or config (reduction.bias_dir / paths.bias_dir)."
        )
    bias_root = Path(clean_path(args.bias_dir)).expanduser().resolve()
    bias_files = sorted(bias_root.rglob("*.fit*"))
    if not bias_files:
        raise FileNotFoundError(f"No FITS files found in bias_dir: {bias_root}")
    
    bias_selected: list[Path] = []
    for f in bias_files:
        try:
            with fits.open(f, memmap=False) as hdul:
                hdr = hdul[0].header
                preferred = cfg.get("instrument", {}).get("exptime_key")
                exptime = get_exptime_from_header(hdr, preferred_key=preferred)
                if is_bias_frame(hdr, exptime, max_exptime=float(args.bias_max_exptime)):
                    bias_selected.append(f)
        except Exception as e:
            logger.warning("Skipping bias candidate %s: %s", f, e)

    if not bias_selected:
        raise RuntimeError(
            f"No bias frames found in {bias_root}. Pipeline cannot run without bias frames. "
            f"Check --bias-dir path or adjust --bias-max-exptime (currently {args.bias_max_exptime}s)."
        )
    
    logger.info("Found %d bias frames (bias_root=%s)", len(bias_selected), bias_root)
    
    # Build master bias
    bias_stack = []
    for f in tqdm(bias_selected, desc="Loading bias", ncols=90):
        with fits.open(f, memmap=False) as hdul:
            bias_stack.append(hdul[0].data.astype(np.float32))
    master_bias = np.nanmedian(np.stack(bias_stack, axis=0), axis=0).astype(np.float32)
    
    # Estimate read noise
    gain = float(cfg.get("instrument", {}).get("gain_e_per_adu", 1.0))
    sigma_bias_adu, read_noise_e = bias_read_noise(bias_selected, gain=gain, box_size=200)
    logger.info("✅ Estimated read noise: %.3f ADU → %.3f e-", sigma_bias_adu, read_noise_e)

    # --- master dark (bias-subtracted if master_bias is available) ---
    dark_files = sorted(dark_root.rglob("*.fit*"))
    dark_selected: list[Path] = []
    for f in dark_files:
        try:
            with fits.open(f) as hdul:
                hdr = hdul[0].header
                preferred = cfg.get("instrument", {}).get("exptime_key")
                exptime = get_exptime_from_header(hdr, preferred_key=preferred)
            # If a 'dark' directory contains mixed calibration frames, ignore BIAS here.
            if is_bias_frame(hdr, exptime, max_exptime=float(args.bias_max_exptime)):
                continue
            if abs(exptime - EXPTIME_TARGET) < EXPTIME_TOL:
                dark_selected.append(f)
        except Exception as e:
            logger.warning("Skipping dark %s: %s", f, e)

    if not dark_selected:
        # Print available dark exposure times to help debugging, then pick the closest
        available_exps = []
        for f in dark_files:
            try:
                with fits.open(f, memmap=False) as hdul:
                    exp = get_exptime_from_header(hdul[0].header)
                if math.isfinite(exp):
                    available_exps.append(round(exp, 3))
            except Exception:
                continue
        if not available_exps:
            raise RuntimeError("No dark frames had a readable exposure keyword (EXPTIME/EXPOSURE/etc.).")
        uniq = sorted(set(available_exps))
        logger.info("Available dark exposure times (s): %s", uniq)
        closest = min(uniq, key=lambda x: abs(x-EXPTIME_TARGET))
        logger.warning(
            "No darks within ±%.2fs of %.3fs. Using closest=%.3fs instead.",
            EXPTIME_TOL, EXPTIME_TARGET, closest,
        )
        EXPTIME_TARGET = closest
        dark_selected = []
        for f in dark_files:
            try:
                with fits.open(f, memmap=False) as hdul:
                    exp = get_exptime_from_header(hdul[0].header)
                if math.isfinite(exp) and abs(exp-EXPTIME_TARGET) <= EXPTIME_TOL:
                    dark_selected.append(f)
            except Exception as e:
                logger.warning("Skipping dark %s: %s", f, e)
        if not dark_selected:
            raise RuntimeError("Still no matching darks found after choosing closest exposure time.")

    logger.info("Found %d darks matching EXPTIME=%.3fs", len(dark_selected), EXPTIME_TARGET)

    dark_stack = []
    for f in tqdm(dark_selected, desc="Loading darks", ncols=90):
        with fits.open(f, memmap=False) as hdul:
            d = hdul[0].data.astype(np.float32)
        if master_bias is not None:
            if d.shape != master_bias.shape:
                raise RuntimeError(f"Dark frame {f.name} shape {d.shape} != bias shape {master_bias.shape}")
            d = ne.evaluate("d - b", local_dict={"d": d, "b": master_bias})
        dark_stack.append(d)
    master_dark = np.nanmedian(np.stack(dark_stack, axis=0), axis=0).astype(np.float32)

    # 1.) Select science frames by exposure time
    fits_files = sorted(science_root.rglob("*.fit*"))
    selected: list[Path] = []

    meta_rows = []  # for manifest/csv, aligned with selected order
    '''# --- OPTIONAL sky-geometry inputs (may be None) ---
    target_ra = cfg.get("target", {}).get("ra")
    target_dec = cfg.get("target", {}).get("dec")
    
    site = cfg.get("site", {})
    obs_lat = site.get("lat_deg")
    obs_lon = site.get("lon_deg")
    obs_height = site.get("height_m")'''

    # - Sky-geometry inputs (optional; may be None) -
    # target_ra, target_dec, obs_lat, obs_lon, obs_height already resolved above.
    for f in fits_files:
        try:
            with fits.open(f) as hdul:
                hdr = hdul[0].header
                exptime = get_exptime_from_header(hdr)
                if abs(exptime - EXPTIME_TARGET) >= EXPTIME_TOL:
                    continue
                dateobs = hdr.get("DATE-OBS") or hdr.get("DATEOBS")
                
                airmass = float("nan")
                # Prefer header keyword if present
                if "AIRMASS" in hdr:
                    try:
                        airmass = float(hdr["AIRMASS"])
                    except Exception:
                        airmass = float("nan")
                
                # If not available, try compute
                if (not np.isfinite(airmass)) and dateobs and target_ra and target_dec and (obs_lat is not None) and (obs_lon is not None) and (obs_height is not None):
                    airmass = compute_airmass(
                        date_obs_utc=dateobs,
                        ra_str=target_ra,
                        dec_str=target_dec,
                        lat_deg=float(obs_lat),
                        lon_deg=float(obs_lon),
                        height_m=float(obs_height),
                    )
            selected.append(f)
            
            # Time conversions
            jd_utc = float("nan")
            bjd_tdb = float("nan")
            if dateobs:
                try:
                    t = Time(dateobs, format="isot", scale="utc")
                    jd_utc = float(t.jd)
                except Exception:
                    jd_utc = float("nan")
                bjd_tdb = compute_bjd_tdb(dateobs, target=target, location=location)

            meta_rows.append({
                "FILE": f.name,
                "PATH": str(f),
                "DATE_OBS": dateobs,
                "EXPTIME": exptime,
                "AIRMASS": float(airmass),
                "JD_UTC": jd_utc,
                "BJD_TDB": bjd_tdb,
            })
        except Exception as e:
            logger.warning("Skipping science %s: %s", f, e)

    if not selected:
        raise RuntimeError("No matching science frames found for the requested exposure time.")

    logger.info("Selected %d science frames matching EXPTIME=%.3fs", len(selected), EXPTIME_TARGET)


    # 2.) Create calibrated cube: (science - master_dark)
    # Use first frame to set shape/header
    with fits.open(selected[0]) as hdul0:
        first_data_raw = hdul0[0].data.astype(np.float32)
        base_header = hdul0[0].header.copy()

    if first_data_raw.shape != master_dark.shape:
        raise RuntimeError(
            f"Science frame shape {first_data_raw.shape} does not match master dark shape {master_dark.shape}"
        )

    n_frames = len(selected)
    ny, nx = first_data_raw.shape
    cube = np.empty((n_frames, ny, nx), dtype=np.float32)

    for i, f in enumerate(tqdm(selected, desc="Building cube", ncols=90)):
        with fits.open(f) as hdul:
            sci = hdul[0].data.astype(np.float32)
        if sci.shape != master_dark.shape:
            raise RuntimeError(f"Frame {f.name} has shape {sci.shape} != dark shape {master_dark.shape}")
        if master_bias is not None:
            if sci.shape != master_bias.shape:
                raise RuntimeError(f"Frame {f.name} has shape {sci.shape} != bias shape {master_bias.shape}")
            cube[i] = ne.evaluate("sci - dark - bias", local_dict={"sci": sci, "dark": master_dark, "bias": master_bias})
        else:
            cube[i] = ne.evaluate("sci - dark", local_dict={"sci": sci, "dark": master_dark})


    # 3.) Write cube with metadata table (FITS) + CSV manifests
    primary_hdu = fits.PrimaryHDU(cube, header=base_header)

    # FITS metadata table (keep minimal)
    from astropy.table import Table
    tab = Table(rows=[
        (r["FILE"], r["DATE_OBS"], r["JD_UTC"], r["BJD_TDB"], r["AIRMASS"], r["EXPTIME"])
        for r in meta_rows
    ], names=["FILE", "DATE_OBS", "JD_UTC", "BJD_TDB", "AIRMASS", "EXPTIME"])
    table_hdu = fits.BinTableHDU(tab, name="METADATA")

    fits.HDUList([primary_hdu, table_hdu]).writeto(output_cube, overwrite=True)
    logger.info("Written cube: %s", output_cube)

    # CSV manifest aligned with cube order (useful for downstream scripts)
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        w.writeheader()
        w.writerows(meta_rows)
    logger.info("Written manifest: %s", manifest_csv)

    # Compatibility time CSV for photometry script
    # Keep just the commonly-used columns
    cols = ["FILE", "DATE_OBS", "JD_UTC", "BJD_TDB", "AIRMASS", "EXPTIME"]
    with open(header_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in meta_rows:
            w.writerow({k: r.get(k, "") for k in cols})
    logger.info("Written time CSV: %s", header_csv)

    # 4.) Write products.json for downstream steps (runner/photometry)
    products = {
        "cube_fits": str(Path(output_cube).resolve()),
        "manifest_csv": str(Path(manifest_csv).resolve()),
        "time_csv": str(Path(header_csv).resolve()),
        "read_noise_e": float(read_noise_e),
        "gain_e_per_adu": float(gain),
    }
    products_path = Path(outdir) / "products.json"
    with open(products_path, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2)
    logger.info("Written products: %s", products_path)
    logger.info("✅ Data Reduction Finished")


if __name__ == "__main__":
    main()
