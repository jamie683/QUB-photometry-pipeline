# INT Pipeline



# --- OVERVIEW ---
'''
End-to-end INT/WFC transit photometry pipeline in a single script.

Workflow:
1) Recursively scan a data directory for FITS frames, classify OBSTYPE (with bias/flat rescue),
   and build a master header table including BJD_TDB and airmass.
2) Interactively select target/filter/night, then perform calibration: overscan+trim, master bias,
   master flats per filter (illumination + pixel flat), and parallel reduction of science frames
   into per-filter calibrated FITS cubes + CSV manifests.
3) For each reduced cube, load or auto-generate a DS9 region file (target + comparison stars),
   track drift using Gaussian centroiding/FWHM, run sky-subtracted aperture photometry, build an
   ensemble comparison light curve, and fit a BATMAN transit model with MCMC.
Outputs are written to ./outputs and ./photometry_outputs (CSVs, FITS products, diagnostic plots,
and fit summaries).
'''  



# --- STANDARD LIBRARY IMPORTS ---
import argparse
import warnings
import json
import os
import sys
import math
import heapq
import logging
import platform
import importlib.metadata
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import re



# --- THIRD PARTY IMPORTS ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, solar_system_ephemeris
import astropy.units as u

from photutils.aperture import aperture_photometry, CircularAnnulus, CircularAperture
from photutils.detection import DAOStarFinder

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

import batman
import emcee
import corner
try:
    # Evaluates array expressions using multithreading and fewer temporaries than NumPy
    import numexpr as ne                         
    from multiprocessing import cpu_count
    
    # Let numexpr use most cores (apart from 1 for system function)
    ne.set_num_threads(max(1, cpu_count() - 1))  
    USE_NUMEXPR = True
except Exception:                                
    USE_NUMEXPR = False



# --- CLI ENTRYPOINT / WARNINGS / LOGGING ---
def parse_args():
    p = argparse.ArgumentParser(description="INT/WFC transit photometry pipeline (monolithic).")
    p.add_argument("--data-dir", type=str, default=None, help="Root directory containing FITS data.")
    p.add_argument("--targets", type=str, default=None, help="Path to targets.json (defaults to ./targets.json).")
    p.add_argument("--instrument", type=str, default=None, help="Path to instrument.json (defaults to ./instrument.json).")
    p.add_argument("--target", type=str, default=None, help="Target key/name to run (skips interactive selection if given).")
    p.add_argument("--band", type=str, default=None, help="Filter/band key (e.g. r, g, i).")
    p.add_argument("--night", type=str, default=None, help="Night/date selector (your existing format).")
    p.add_argument("--outdir", type=str, default=None, help="Output directory.")
    p.add_argument("--verbose", action="store_true", help="More console output.")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility.")
    return p.parse_args()

# - WARNINGS -
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)      
warnings.filterwarnings("ignore", message="partition.*MaskedArray", category=UserWarning)            
warnings.filterwarnings("ignore", message="The fit may not have converged", category=UserWarning)    
warnings.filterwarnings("ignore", message="Warning: 'partition' will ignore the 'mask' of the MaskedArray.", category=UserWarning)
warnings.filterwarnings("ignore", message="Format strings passed to MaskedConstant are ignored", category=FutureWarning)
warnings.filterwarnings("ignore", message="This figure includes Axes that are not", category=UserWarning)    

# - LOGGING -  
'''
Can replace some print(...) with logger.info(...) and warnings with logger.warning(...)
'''
def setup_logging(outdir: Path | None = None, *, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("INT_PIPELINE")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(outdir / "pipeline.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

def write_run_metadata(outdir: Path, *, seed: int, data_root: Path, target: str | None = None):
    pkgs = ["numpy", "scipy", "astropy", "photutils", "matplotlib", "emcee", "batman-package"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = importlib.metadata.version(p)
        except Exception:
            versions[p] = None

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": seed,
        "target": target,
        "package_versions": versions,
    }

    path = outdir / "run_metadata.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"📝 Wrote metadata: {path.name}")
    
    
    
# --- CONFIGURATION & GLOBALS ---

# True = auto-generate DS9 regions from cube
AUTO_REG = False          

# True = overwrite existing reg file (debug)
AUTO_REG_FORCE = False   

# Globals for worker cache
g_master_bias = None

# dict: filt_key -> (pixflat_array, illumflat_array)
g_flats = None           

# dict: cleaned -> original key
g_flat_keys_clean = None 
g_inst = None



# --- CONFIG LOADERS ---
def load_instrument_config(inst_path, instrument_name="INT_WFC"):
    """
    Load instrument parameters. Supports nested instrument.json like:
    { "INT_WFC": { "hdu_index": ..., "overscan": {...}, "noise": {...}, "photometry": {...} } }
    """
    inst = {
        "hdu_index": 4,

        # Overscan
        "horizontal_overscan": 53,
        "vertical_overscan": 100,
        "measure_side": "left",
        "subtract_overscan": True,
        "trim_left": True,
        "trim_right": True,
        "trim_top": True,
        "trim_bottom": True,

        # Noise / gain
        "gain_e_per_adu": 2.9,
        "auto_readnoise": True,
        "read_noise_e": float("nan"),
        "beta_factor": 1.0,

        # Parallelism
        "max_workers": None,

        # Photometry defaults
        "start_frame": 0,
        "min_snr_target": 3.0,
    }

    cfg_path = Path(inst_path)
    if not cfg_path.is_file():
        print("ℹ️ No instrument.json found, using code defaults.")
        return inst

    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("instrument.json must contain a JSON object (dict).")

        # If file is instrument-keyed (like your current one), select the block:
        user_cfg = raw.get(instrument_name, raw)

        if not isinstance(user_cfg, dict):
            raise ValueError(f"instrument.json[{instrument_name}] must be a JSON object (dict).")

        # Map nested schema -> flat keys used by pipeline
        if "hdu_index" in user_cfg:
            inst["hdu_index"] = user_cfg["hdu_index"]

        overscan = user_cfg.get("overscan", {})
        if isinstance(overscan, dict):
            inst["horizontal_overscan"] = overscan.get("horizontal", inst["horizontal_overscan"])
            inst["vertical_overscan"]   = overscan.get("vertical", inst["vertical_overscan"])
            inst["measure_side"]        = overscan.get("measure_side", inst["measure_side"])
            inst["subtract_overscan"]   = overscan.get("subtract", inst["subtract_overscan"])
            inst["trim_left"]           = overscan.get("trim_left", inst["trim_left"])
            inst["trim_right"]          = overscan.get("trim_right", inst["trim_right"])
            inst["trim_top"]            = overscan.get("trim_top", inst["trim_top"])
            inst["trim_bottom"]         = overscan.get("trim_bottom", inst["trim_bottom"])

        noise = user_cfg.get("noise", {})
        if isinstance(noise, dict):
            inst["gain_e_per_adu"] = noise.get("gain_e_per_adu", inst["gain_e_per_adu"])
            inst["read_noise_e"]   = noise.get("read_noise_e", inst["read_noise_e"])
            inst["auto_readnoise"] = noise.get("auto_readnoise", inst["auto_readnoise"])
            inst["beta_factor"]    = float(noise.get("beta_factor", 1.0))
            
        phot = user_cfg.get("photometry", {})
        if isinstance(phot, dict):
            inst["start_frame"]    = phot.get("start_frame", inst["start_frame"])
            inst["min_snr_target"] = phot.get("min_snr_target", inst["min_snr_target"])

            

        print(f"✅ Loaded instrument config '{instrument_name}' from {cfg_path.name}")
        return inst

    except Exception as e:
        print(f"⚠️ Failed to read instrument.json ({cfg_path}): {e}")
        return inst

def load_target_config(path: Path, target_name: str) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))

    if target_name not in cfg:
        raise KeyError(f"Target '{target_name}' not found in {path.name}. Available: {list(cfg.keys())}")

    targ = cfg[target_name]
    targ = dict(targ)
    targ["name"] = target_name
    return targ

def resolve_target_name_from_object(targets_db: dict, object_name: str) -> str:
    if not object_name:
        raise ValueError("Empty OBJECT string; can't resolve target config.")

    obj_clean = _clean_name(object_name)

    # 1) direct key match
    for key in targets_db.keys():
        if object_name == key or obj_clean == _clean_name(key):
            return key

    # 2) alias match
    for key, block in targets_db.items():
        for a in block.get("aliases", []):
            if object_name == a or obj_clean == _clean_name(a):
                return key

    raise KeyError(
        f"OBJECT='{object_name}' not found in targets DB. Available keys: {list(targets_db.keys())}"
    )
    
    
    
# --- UTILITY HELPERS ---
def _to_float(x):
    try: 
        return float(x)
    except Exception: 
        return np.nan

def _is_missing(x) -> bool:
    if x is None:
        return True
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False
    
def _clean_name(s: str) -> str:
    s = "" if s is None else str(s)
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def _clean(s: str) -> str:
    return _clean_name(s)

def file_stem_key(path: str) -> str:
    """
    int20120502_00895155.fits.fz -> int20120502_00895155
    """
    name = Path(path).name
    for _ in range(3):
        new = re.sub(r"(\.fz|\.gz|\.fits|\.fit)$", "", name, flags=re.IGNORECASE)
        if new == name:
            break
        name = new
    return name

def _slug(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s if s else "unknown"

def get_ld_params(targ: dict, band: str):
    ld = targ.get("limb_darkening", {})
    if band not in ld:
        raise KeyError(f"No limb_darkening entry for band='{band}'. Available: {list(ld.keys())}")
    d = ld[band]
    return float(d["U1_0"]), float(d["U2_0"]), float(d.get("SIG_U", 0.10))



# --- FITS I/O helpers ---
def read_fits(path, hdu_index=None):
    if hdu_index is None:
        hdu_index = 0
    with fits.open(path, memmap=False) as hdul:
        return hdul[hdu_index].data.astype(np.float64)

def write_fits(path, data, header=None, overwrite=True, history=None):
    h = fits.Header() if header is None else header.copy()                 
    if history:
        for line in history:
            h.add_history(line)                                            
    fits.PrimaryHDU(data=data, header=h).writeto(path, overwrite=overwrite)
    
    
    
# --- CALIBRATION PRIMITIVES ---
def overscan_correct(image: np.ndarray, inst: dict):
    img = np.asarray(image, dtype=np.float64).copy()
    ny, nx = img.shape

    # Pull from config dict
    h_ov = int(inst["horizontal_overscan"])
    v_ov = int(inst["vertical_overscan"])
    side = str(inst["measure_side"]).lower().strip()

    subtract = bool(inst["subtract_overscan"])
    trim_left   = bool(inst["trim_left"])
    trim_right  = bool(inst["trim_right"])
    trim_top    = bool(inst["trim_top"])
    trim_bottom = bool(inst["trim_bottom"])

    # 1.) Measure strip
    strip = None
    if side == "left" and h_ov > 0:
        w = min(h_ov, nx)
        strip = img[:, :w]
    elif side == "right" and h_ov > 0:
        w = min(h_ov, nx)
        strip = img[:, nx-w:nx]
    elif side == "top" and v_ov > 0:
        h = min(v_ov, ny)
        strip = img[:h, :]
    elif side == "bottom" and v_ov > 0:
        h = min(v_ov, ny)
        strip = img[ny-h:ny, :]
    level = float(np.nanmedian(strip)) if (strip is not None and strip.size) else 0.0

    # 2.) Subtract
    if subtract:
        img -= level

    # 3.) Trim bounds
    x0, x1 = 0, nx
    y0, y1 = 0, ny

    if h_ov > 0:
        if trim_left:
            x0 = min(h_ov, nx)
        if trim_right:
            x1 = max(x0, nx - h_ov)

    if v_ov > 0:
        if trim_top:
            y0 = min(v_ov, ny)
        if trim_bottom:
            y1 = max(y0, ny - v_ov)

    if (x1 - x0) < 10 or (y1 - y0) < 10:
        raise ValueError(
            f"Overscan trim produced tiny/empty image: orig=({ny},{nx}) "
            f"x0={x0},x1={x1} y0={y0},y1={y1} h_ov={h_ov} v_ov={v_ov}"
        )

    return img[y0:y1, x0:x1], level

# - MEDIAN COMBINING -
def median_combine(images, use_sigma_clip=True, clip_sigma=3.0):
    """Median combine 2D frames with sigma clipping."""
    stack = np.stack(images, axis=0)
    if use_sigma_clip:
        
        # Astropy function replacing outliers with NaN
        clipped = sigma_clip(stack, sigma=clip_sigma, axis=0, masked=True)
        return np.nanmedian(clipped.filled(np.nan), axis=0)
    
    # If use_sigma_clip is False, this line runs instead.
    return np.median(stack, axis=0)

# - MASTER BIAS -
def build_master_bias(bias_files, inst: dict):
    images = []                                     
    for f in bias_files:
        image = read_fits(f, hdu_index=inst["hdu_index"])
        image_corr, level = overscan_correct(image, inst)
        images.append(image_corr)
    return median_combine(images)

# - MASTER FLAT -
def build_master_flat(flat_files, master_bias, inst: dict):
    """Build a normalised master flat from a list of flat frames."""
    images = []
    for f in flat_files:
         # Loads 'raw' fit files
        image = read_fits(f, hdu_index=inst["hdu_index"])

        # Overscan correction          
        image_corr, level = overscan_correct(image, inst)
        
        # Subtract master bias
        image_corr -= master_bias        

        # Add corrected flats to list
        images.append(image_corr)                   

    if not images:
        print('⚠️ No flat files found')
        return None, None
        
    # Median combine all corrected flats
    mflat = median_combine(images, use_sigma_clip=True, clip_sigma=3.0)

    # split into illumination & pixel flats
    med = np.nanmedian(mflat[np.isfinite(mflat)])
    if not np.isfinite(med) or med == 0:
        raise RuntimeError("⚠️ Flat normalisation failed (median is zero or non-finite).")

    mflat_norm = mflat / med     # ✅ define it before use

    # 1.) Large-scale illumination: heavy Gaussian smoothing
    illum = gaussian_filter(mflat_norm, sigma=40.0)   # tune sigma if needed
    illum_med = np.nanmedian(illum[np.isfinite(illum)])
    if not np.isfinite(illum_med) or illum_med == 0:
        illum_norm = np.ones_like(illum, dtype=float)
    else:
        illum_norm = illum / illum_med

    # 2.) Pixel-scale flat = master / illumination
    pixflat = mflat_norm / illum_norm
    pix_med = np.nanmedian(pixflat[np.isfinite(pixflat)])
    if not np.isfinite(pix_med) or pix_med == 0:
        pixflat_norm = pixflat
    else:
        pixflat_norm = pixflat / pix_med
        
    # 3.) Guard check
    mflat = np.asarray(mflat, dtype=float)
    finite = np.isfinite(mflat)
    if finite.sum() < 0.99 * mflat.size:
        print(f"⚠️ Master flat has many non-finite pixels: {(~finite).sum()} / {mflat.size}")
    
    p1, p50, p99 = np.nanpercentile(mflat, [1, 50, 99])
    
    # Guard against nonsense flats
    if not np.isfinite(p50) or p50 == 0:
        print("⚠️ Master flat median is invalid (0 or non-finite). This flat will break calibration.")
        
    return pixflat_norm, illum_norm

def build_master_flats_by_filter(flat_files, master_bias, outdir: Path, inst: dict):
    """
    flat_files: list[Path] of flat frames (same night/date window already chosen)
    Returns flat_map: {filter_key: (pix_path, illum_path)}
    """
    groups = defaultdict(list)

    for f in flat_files:
        try:
            with fits.open(f, memmap=False) as hdul:
                H0 = hdul[0].header
            filt = filter_from_header(H0)
            if not filt:
                continue
            groups[filt].append(f)
        except Exception:
            continue

    if not groups:
        raise SystemExit("⚠️ No usable flats (no filter detected).")

    flat_map = {}
    for filt, files in sorted(groups.items(), key=lambda kv: kv[0].lower()):
        pixflat, illumflat = build_master_flat(files, master_bias, inst)
        if pixflat is None or illumflat is None:
            continue

        out_pix   = outdir / f"pix_flat_{filt}.fits"
        out_illum = outdir / f"illum_flat_{filt}.fits"

        write_fits(out_pix, pixflat, history=[f"PIXFLAT for {filt}", f"Built from {len(files)} flats"])
        write_fits(out_illum, illumflat, history=[f"ILLUMFLAT for {filt}", f"Built from {len(files)} flats"])

        flat_map[filt] = (out_pix, out_illum)

    if not flat_map:
        raise SystemExit("⚠️ No master flats created.")
    return flat_map

def divide_by_flat(image_corr: np.ndarray, mflat: np.ndarray) -> np.ndarray:
    """Fast, safe divide (multithreaded with numexpr if available)."""
    safe = np.where(np.isfinite(mflat) & (mflat > 0), mflat, 1.0)

    if USE_NUMEXPR:
        # Note: string expression, variables from local_dict
        return ne.evaluate("image_corr / safe",
                           local_dict={"image_corr": image_corr, "safe": safe})
    else:
        return image_corr / safe

# - READ NOISE -
def bias_read_noise(bias_files, inst: dict, box_size=200):
    """
    Estimate read noise from a stack of bias frames.
    """
    gain = float(inst["gain_e_per_adu"])
    bias_imgs = []

    for f in bias_files:
        image = read_fits(f, hdu_index=inst["hdu_index"])
        img_corr, level = overscan_correct(image, inst)

        # Crop to central box_size x box_size region
        ny, nx = img_corr.shape
        b = min(box_size, ny, nx)
        x0 = (nx - b) // 2
        y0 = (ny - b) // 2
        img_sub = img_corr[y0:y0 + b, x0:x0 + b]

        bias_imgs.append(img_sub)

    # Stack cropped bias frames into cube
    bias_cube = np.stack(bias_imgs, axis=0)

    # Per-pixel std across the cube (ADU)
    sigma_map_adu = np.std(bias_cube, axis=0, ddof=1)

    # Use median 
    sigma_bias_adu = np.median(sigma_map_adu)

    # Convert to electrons
    read_noise_e = sigma_bias_adu * gain

    return sigma_bias_adu, read_noise_e



# --- HEADER & MASTER TABLE ---
def collect_headers(root: Path, *, obstype_keep=("TARGET", "FLAT", "BIAS")):
    """
    Scan FITS files, classify frames (incl. bias rescue), then keep only OBSTYPE(s),
    extract JD/AIRMASS/OBJECT/FILTER/coords.
    Adds join_key for later merging with cube manifests.
    Also returns skip stats.
    """
    rows = []
    skip_counter = {}
    total = 0
    keep_set = {str(x).strip().upper() for x in (obstype_keep or [])}
    fits_files = sorted(root.rglob("*.fit*"))
    print(f"Found {len(fits_files)} FITS files under: {root}")

    for fp in fits_files:
        total += 1
        try:
            with fits.open(fp, memmap=False) as hdul:
                H0 = hdul[0].header

            obstype_raw = str(H0.get("OBSTYPE", "")).strip().upper()
            exptime = _to_float(H0.get("EXPTIME", H0.get("EXP_TIME", np.nan)))

            # classify / rescue BEFORE filtering 
            obstype = obstype_raw

            # If OBSTYPE missing/blank, attempt simple rescue rules
            if not obstype:
                obstype = ""

            # Bias: exposure time basically zero
            if obstype in ("", "UNKNOWN") and np.isfinite(exptime) and exptime <= 0.1:
                obstype = "BIAS"

           # Flat rescue
            obj_u = str(H0.get("OBJECT", "")).strip().upper()
            if obstype in ("", "UNKNOWN") and ("FLAT" in obj_u or "DOME" in obj_u or "TWIL" in obj_u):
                obstype = "FLAT"

            # NOW filter by keep_set
            if keep_set and obstype not in keep_set:
                skip_counter["OBSTYPE_not_kept"] = skip_counter.get("OBSTYPE_not_kept", 0) + 1
                continue

            date_obs = str(H0.get("DATE-OBS", "")).strip()
            night = date_obs.split("T")[0] if "T" in date_obs else date_obs[:10]

            jd = _to_float(H0.get("JD", np.nan))
            am = _to_float(H0.get("AIRMASS", np.nan))

            obj = str(H0.get("OBJECT", "")).strip()
            filt = H0.get("WFFBAND", H0.get("FILTER", H0.get("INSFLNAM", "unknown")))
            filt = str(filt).strip()

            # Coords
            ra_deg = np.nan
            dec_deg = np.nan

            cat_ra = H0.get("CAT-RA", None)
            cat_dec = H0.get("CAT-DEC", None)

            if cat_ra is not None and cat_dec is not None:
                try:
                    c = SkyCoord(str(cat_ra).strip(), str(cat_dec).strip(), unit=(u.hourangle, u.deg))
                    ra_deg, dec_deg = c.ra.deg, c.dec.deg
                except Exception:
                    pass

            if (not np.isfinite(ra_deg)) and ("RA" in H0) and ("DEC" in H0):
                try:
                    c = SkyCoord(str(H0["RA"]).strip(), str(H0["DEC"]).strip(), unit=(u.hourangle, u.deg))
                    ra_deg, dec_deg = c.ra.deg, c.dec.deg
                except Exception:
                    pass

            rows.append({
                "filename": fp.name,
                "filepath": str(fp),
                "join_key": file_stem_key(fp.name),

                "JD_UTC": jd,
                "AIRMASS": am,

                "DATE_OBS": date_obs,
                "NIGHT": night,

                "OBJECT": obj,
                "OBSTYPE": obstype,
                "FILTER": filt,

                "RA_DEG": float(ra_deg) if np.isfinite(ra_deg) else np.nan,
                "DEC_DEG": float(dec_deg) if np.isfinite(dec_deg) else np.nan,
            })

        except Exception as e:
            skip_counter["open_or_parse_failed"] = skip_counter.get("open_or_parse_failed", 0) + 1
            print(f"⚠️ Skipped {fp.name}: {e}")

    print("\n📊 Header scan summary:")
    print(f"  Total files scanned: {total}")
    print(f"  Rows kept:           {len(rows)}")
    skipped = total - len(rows)
    print(f"  Skipped:             {skipped}")
    if skip_counter:
        for k, v in sorted(skip_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    - {k}: {v}")

    return rows

# - BJD + MASTER WRITE -
def make_master_table(rows, outdir: Path, basename: str, *, write_master=True, ephemeris="de432s"):
    if not rows:
        raise SystemExit("No frames kept. Check OBSTYPE_KEEP and your root folder.")

    df = pd.DataFrame(rows)

    # Numeric
    df["JD_UTC"]  = pd.to_numeric(df["JD_UTC"], errors="coerce")
    df["AIRMASS"] = pd.to_numeric(df["AIRMASS"], errors="coerce")
    df["RA_DEG"]  = pd.to_numeric(df["RA_DEG"], errors="coerce")
    df["DEC_DEG"] = pd.to_numeric(df["DEC_DEG"], errors="coerce")

    # Sort for stable order
    df = df.sort_values(["OBJECT", "FILTER", "JD_UTC"], kind="mergesort").reset_index(drop=True)
    df["order_idx"] = np.arange(len(df), dtype=int)

    # Site (INT)
    try:
        site = EarthLocation.of_site("Roque de los Muchachos")
    except Exception:
        site = EarthLocation.from_geodetic(lon=-17.8792*u.deg, lat=28.7606*u.deg, height=2326*u.m)

    # Ephemeris (safe)
    try:
        solar_system_ephemeris.set(ephemeris)
    except Exception:
        solar_system_ephemeris.set("builtin")

    # Compute BJD_TDB per (OBJECT, FILTER) — safer if you ever mix targets/filters
    bjd = np.full(len(df), np.nan, dtype=float)

    for (obj, filt), g in df.groupby(["OBJECT", "FILTER"], dropna=False):
        idx = g.index.to_numpy()
        jd_arr  = g["JD_UTC"].to_numpy(float)
        ra_arr  = g["RA_DEG"].to_numpy(float)
        dec_arr = g["DEC_DEG"].to_numpy(float)

        good_time = np.isfinite(jd_arr)
        good_coord = np.isfinite(ra_arr) & np.isfinite(dec_arr)

        if good_time.sum() == 0 or good_coord.sum() == 0:
            continue

        ra0 = float(ra_arr[good_coord][0])
        dec0 = float(dec_arr[good_coord][0])
        coord = SkyCoord(ra0*u.deg, dec0*u.deg, frame="icrs")

        t_utc = Time(jd_arr[good_time], format="jd", scale="utc", location=site)
        t_tdb = t_utc.tdb
        ltt   = t_tdb.light_travel_time(coord, location=site)
        bjd[idx[good_time]] = (t_tdb + ltt).value

    df["BJD_TDB"] = bjd

    if write_master:
        out_csv = outdir / f"{basename}.csv"
        df.to_csv(out_csv, index=False)
        print(f"✅ Wrote {out_csv.name}")

    return df

def rescue_flats_if_needed(df: pd.DataFrame, min_flats=5) -> pd.DataFrame:
    # if we already have enough flats, do nothing
    n_flats = (df["OBSTYPE"] == "FLAT").sum()
    if n_flats >= min_flats:
        return df

    print(f"⚠️ Only found {n_flats} flats via OBSTYPE. Running fallback flat detection...")

    df2 = df.copy()
    obj_u = df2["OBJECT"].astype(str).str.upper()

    # Heuristics
    looks_flat = (
        obj_u.str.contains("FLAT", na=False) |
        obj_u.str.contains("DOME", na=False) |
        obj_u.str.contains("SKY",  na=False) |
        obj_u.str.contains("TWIL", na=False)
    )

    # Only relabel if it isn't already classified
    mask = (df2["OBSTYPE"].isin(["", "UNKNOWN"])) & looks_flat
    df2.loc[mask, "OBSTYPE"] = "FLAT"

    print(f"✅ After rescue: flats={(df2['OBSTYPE']=='FLAT').sum()}")
    return df2

# - FILTER AUTOMATIC DETECTION (FOR SCIENCE FRAMES) -
def filter_from_header(H0) -> str | None:
    raw = H0.get("WFFBAND", H0.get("FILTER", H0.get("INSFLNAM", "")))
    raw = str(raw).strip()
    return normalize_filter_name(raw) if raw else None

def normalize_filter_name(raw: str) -> str:
    
    # Removes leading/trailing whitespace; keeps the original spelling
    s = raw.strip()
    
    # Builds 'clean' version: no spaces/hyphens/underscores, all lowercase
    s_clean = s.replace(" ", "").replace("-", "").replace("_", "").lower()
    
    # Commonly used filters
    table = {
        "b":"B", "g":"G", "u":"U", "v":"V", "r":"r",
        "halpha":"Halpha", "ha":"Halpha",
        "hbeta":"Hbeta_Broad", "hb":"Hbeta_Broad",
    }
    if s_clean in table:
        return table[s_clean]

    if "halpha" in s_clean or s_clean.startswith("ha"):
        return "Halpha"     # The following are for looser matches
    if "hbeta" in s_clean or "hbroad" in s_clean:
        return "Hbeta_Broad"
    if s_clean.startswith("b"):
        return "B"
    if s_clean.startswith("g"):
        return "G"
    if s_clean.startswith("u"):
        return "U"
    if s_clean.startswith("v"):
        return "V"
    if s_clean.startswith("r"):
        return "r"
    # If nothing matched, return the original (trimmed) string.
    return s



# --- DATA REDUCTION (PARALLEL CUBE BUILDING) ---

# - GLOBAL WORKER -
def init_worker(master_bias, flat_map, inst):
    """
    Runs once per process. Cache master bias + all flats in RAM for speed.
    """
    global g_master_bias, g_flats, g_flat_keys_clean, g_inst
    g_inst = inst
    g_master_bias = master_bias

    # Flat_map: {key: (pix_path, illum_path)}
    g_flats = {}
    for key, (pix_path, illum_path) in flat_map.items():
        g_flats[key] = (fits.getdata(pix_path), fits.getdata(illum_path))

    g_flat_keys_clean = {_clean(k): k for k in flat_map.keys()}
    
def choose_workers(n_tasks: int, *, max_cap: int | None = None) -> int:
    """
    Choose a sensible number of worker processes across platforms.

    - Leaves 1 core free
    - Caps aggressively on Windows (spawn overhead)
    - Caps by number of tasks
    - Never returns < 1
    """

    # Defensive: no tasks → no workers needed, but return 1 so caller doesn't explode
    if n_tasks <= 0:
        return 1

    cores = os.cpu_count()
    if cores is None or cores < 1:
        cores = 1

    # Leave one core free for OS / UI
    base = max(1, cores - 1)

    # Platform cap
    if sys.platform.startswith("win"):
        cap = 6
    else:
        cap = 12

    # User override (if provided and sane)
    if max_cap is not None:
        try:
            max_cap = int(max_cap)
            if max_cap > 0:
                cap = min(cap, max_cap)
        except Exception:
            pass  # ignore bad user input silently

    # Final clamp
    workers = min(base, cap, n_tasks)

    return max(1, workers)
    
def get_available_ram_bytes():
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        # Fallback: unknown
        return None

def cap_workers_for_ram(n_workers, est_bytes_per_worker, *, frac=0.5):
    avail = get_available_ram_bytes()
    if avail is None:
        return n_workers
    max_workers = max(1, int((avail * frac) // max(1, est_bytes_per_worker)))
    return max(1, min(n_workers, max_workers))

def science_frame(args):
    """
    Process one science frame using cached bias/flats.
    Returns (src_idx, filt_key, calibrated_image, meta) or (src_idx, None, None, None).
    """
    src_idx, f = args

    try:
        with fits.open(f, memmap=False) as hdul:
            H0 = hdul[0].header
            image = hdul[int(g_inst["hdu_index"])].data.astype(np.float64)
    except Exception:
        return src_idx, None, None, None, "exception"

    # Overscan + trim
    image_corr, _ = overscan_correct(image, g_inst)

    # Bias subtract (cached)
    image_corr -= g_master_bias

    # Filter detect from primary header (fast)
    filt = filter_from_header(H0)
    if not filt:
        return src_idx, None, None, None, "no_filter"

    # Match to flat key
    fclean = _clean(filt)
    key = g_flat_keys_clean.get(fclean)

    if key is None:
        # Substring fallback
        for kclean, korig in g_flat_keys_clean.items():
            if kclean in fclean or fclean in kclean:
                key = korig
                break
            
    if key is None:
        return src_idx, None, None, None, "no_matching_flat"

    # Cached flats (no disk IO here)
    pixflat, illumflat = g_flats[key]
    image_corr = divide_by_flat(image_corr, pixflat)
    image_corr = divide_by_flat(image_corr, illumflat)

    meta = {
        "filename": Path(f).name,
        "filepath": str(f),
        "join_key": file_stem_key(f),
        "JD_UTC": _to_float(H0.get("JD", np.nan)),
        "OBJECT": str(H0.get("OBJECT", "")).strip(),
        "OBSTYPE": str(H0.get("OBSTYPE", "")).strip().upper(),
        "FILTER": str(H0.get("WFFBAND", H0.get("FILTER", ""))).strip(),
        "filter_key": key,
    }

    return src_idx, key, image_corr, meta, None

# - DATA CUBE -
def build_data_cube(sci_files, master_bias, flat_map, outdir, inst: dict):
    """
    Build calibrated data cubes per filter.
    Returns {filter_key: cube ndarray (N, Y, X)} and writes cube_manifest_<filter>.csv.

    science_frame must return:
      (src_idx, key, image_corr, meta, skip_reason)
    """

    total = len(sci_files)
    if total == 0:
        raise SystemExit("No science files provided to build_data_cube().")

    # Choose workers
    ncpu = choose_workers(total, max_cap=inst.get("max_workers"))
    print(f"Using {ncpu} CPU cores...")

    ny, nx = fits.getdata(sci_files[0], ext=inst["hdu_index"]).shape
    
    # Bias + pix+illum per flat
    est_bytes_worker = (ny*nx*8) * (1 + 2*len(flat_map))  
    ncpu = cap_workers_for_ram(ncpu, est_bytes_worker, frac=0.5)
    print(f"Workers after RAM guard: {ncpu}")

    # Safe chunksize
    chunksize = inst.get("mp_chunksize", 30)
    try:
        chunksize = int(chunksize)
    except Exception:
        chunksize = 30
    chunksize = max(1, min(chunksize, total))

    img_by_filter  = defaultdict(dict)   # filt -> {src_idx: image}
    meta_by_filter = defaultdict(dict)   # filt -> {src_idx: meta}
    skip_counter   = Counter()

    # Streamed args (avoid huge list in RAM)
    args_iter = ((idx, str(f)) for idx, f in enumerate(sci_files))

    def _accumulate_result(result):
        """Shared accumulator for both parallel + serial fallback."""
        try:
            src_idx, key, image_corr, meta, skip_reason = result
        except Exception:
            skip_counter["bad_result_unpack"] += 1
            return

        if skip_reason is not None:
            skip_counter[str(skip_reason)] += 1
            return

        if key is None or image_corr is None or meta is None:
            skip_counter["bad_return"] += 1
            return

        img_by_filter[key][src_idx] = image_corr
        meta_by_filter[key][src_idx] = meta

    # Run parallel, fallback to serial if it blows up 
    try:
        with ProcessPoolExecutor(
            max_workers=ncpu,
            initializer=init_worker,
            initargs=(master_bias, flat_map, inst)
        ) as exe:

            results_iter = exe.map(science_frame, args_iter, chunksize=chunksize)

            for result in tqdm(results_iter, total=total, desc="Parallel calibration", ncols=90):
                _accumulate_result(result)

    except Exception as e:
        print(f"⚠️ Multiprocessing failed ({e}); falling back to single-core calibration...")
        try:
            init_worker(master_bias, flat_map, inst)
        except Exception as e2:
            raise SystemExit(f"Fallback init_worker() failed: {e2}")

        # serial loop
        for args in tqdm(((i, str(f)) for i, f in enumerate(sci_files)),
                         total=total, desc="Serial calibration", ncols=90):
            _accumulate_result(science_frame(args))

    # Summary + guard for "kept==0"
    kept = sum(len(v) for v in img_by_filter.values())
    skipped = total - kept

    print("\n📊 Reduction summary:")
    print(f"  Total science frames:       {total}")
    print(f"  Successfully calibrated:    {kept}")
    print(f"  Skipped:                    {skipped}")

    if skipped > 0:
        print("  Skip reasons:")
        for reason, count in skip_counter.most_common():
            print(f"    - {reason}: {count}")

    if kept == 0:
        raise SystemExit("No calibrated science frames produced (all frames skipped).")

    # Stack per filter + write manifests 
    cubes_dict = {}
    for filt, idx_map in img_by_filter.items():
        
        # Preserve original sci_files order
        idx_sorted = sorted(idx_map.keys())         
        imgs = [idx_map[i] for i in idx_sorted]
        cube = np.stack(imgs, axis=0)
        cubes_dict[filt] = cube

        meta_rows = []
        for cube_idx, src_idx in enumerate(idx_sorted):
            row = dict(meta_by_filter[filt][src_idx])  # copy
            row["cube_idx"]   = int(cube_idx)
            row["src_idx"]    = int(src_idx)
            row["filter_key"] = str(filt)
            row["cube_file"]  = f"cube_{filt}.fits"
            row.setdefault("join_key", "")
            meta_rows.append(row)

        mdf = pd.DataFrame(meta_rows)
        if cube.shape[0] != len(meta_rows):
            raise RuntimeError(
                f"Cube/manifest mismatch for {filt}: cube has {cube.shape[0]} frames, "
                f"manifest has {len(meta_rows)} rows."
            )
        mpath = outdir / f"cube_manifest_{filt}.csv"
        mdf.to_csv(mpath, index=False)

        print(f"✅ Built cube for {filt}: {cube.shape[0]} frames + manifest ({mpath.name})")

    return cubes_dict

def Data_Reduction(sci_files, flat_files, bias_files, outdir: Path, inst: dict):
    outdir.mkdir(parents=True, exist_ok=True)

    master_bias = build_master_bias(bias_files, inst)
    
    if inst.get("auto_readnoise", True):
        try:
            _, rn_e = bias_read_noise(bias_files, inst=inst)
            inst["read_noise_e"] = float(rn_e)
            print(f"✅ Estimated read noise: {inst['read_noise_e']:.3f} e-")
        except Exception as e:
            print(f"⚠️ Read-noise estimate failed ({e}); read_noise_e remains {inst['read_noise_e']}")
    else:
        if not np.isfinite(inst.get("read_noise_e", float("nan"))):
            print("⚠️ auto_readnoise=False but read_noise_e not set. Noise model will be degraded.")

    write_fits(outdir / "master_bias.fits", master_bias)

    flat_map = build_master_flats_by_filter(flat_files, master_bias, outdir, inst)
    if not flat_map:
        raise SystemExit("No master flats available after flat construction.")

    cubes_dict = build_data_cube(sci_files, master_bias, flat_map, outdir, inst)
    if not cubes_dict:
        raise SystemExit("No calibrated science frames produced.")

    for filt, cube in cubes_dict.items():
        write_fits(outdir / f"cube_{filt}.fits", cube)

    return cubes_dict, master_bias, flat_map



# --- REGION HANDLING (DS9) ---
def load_ds9_circles(path: Path, subtract_one: bool=True):
    circles = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith('#') or s.lower().startswith(('global','fk5','icrs','galactic')):
            continue
        if s.startswith('circle(') and ')' in s:
            parts = [p.strip() for p in s[s.find('(')+1:s.find(')')].split(',')]
            if len(parts) >= 3:
                x, y, r = map(float, parts[:3])
                
                # Convert DS9 1-based to 0-based
                if subtract_one:
                    x -= 1.0
                    y -= 1.0
                circles.append((x, y, r))
    return circles

def write_ds9_circles(path: Path, circles_xy_r, add_header=True):
    lines = []
    if add_header:
        lines += ["# Region file format: DS9 version 4.1", "image", "global color=green"]
    for x, y, r in circles_xy_r:
        lines.append(f"circle({x+1:.2f},{y+1:.2f},{r:.2f})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    
# - REG FILE DIAGNOSTICS -
def prompt_for_reg_file(project_root: Path, band: str) -> Path:
    default = project_root / f"{band}_targets.reg"
    if default.is_file():
        print(f"✅ Using region file: {default}")
        return default

    while True:
        s = input(f"\nPaste path to DS9 .reg file for band '{band}' (default would be {default.name}):\n> ").strip().strip('"')
        p = Path(s).expanduser()
        if p.is_file() and p.suffix.lower() == ".reg":
            return p.resolve()
        print("❌ Not a valid .reg file. Try again.")
        
def build_reference_image_from_cube(cube_path: Path, n_frames=10):
    cube = fits.getdata(cube_path).astype(np.float64)
    N = cube.shape[0]
    k = min(max(1, int(n_frames)), N)
    ref = np.nanmedian(cube[:k], axis=0)
    return ref

def detect_stars_daofinder(ref_img: np.ndarray, fwhm_guess=4.0, threshold_sigma=5.0):
    mean, med, std = sigma_clipped_stats(ref_img, sigma=3.0, maxiters=5)
    data = ref_img - med
    finder = DAOStarFinder(fwhm=float(fwhm_guess), threshold=float(threshold_sigma) * float(std))
    tab = finder(data)

    if tab is None or len(tab) == 0:
        return None

    # Convert to a simple dataframe-like structure
    # columns typically include: xcentroid, ycentroid, flux, peak, sharpness, roundness1, roundness2
    return tab

def save_detection_plot(ref_img, sources, out_png: Path, max_to_plot=80):
    """
    Saves a diagnostic plot with numbered detections.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 7))
    v1, v2 = np.nanpercentile(ref_img, [5, 99.5])
    plt.imshow(ref_img, origin="lower", vmin=v1, vmax=v2)
    plt.title("Auto star detection (IDs overplotted)")
    plt.colorbar(label="ADU")

    n = min(len(sources), int(max_to_plot))
    xs = sources["xcentroid"][:n]
    ys = sources["ycentroid"][:n]

    plt.scatter(xs, ys, s=20, marker="o")

    for i in range(n):
        x = float(xs[i])
        y = float(ys[i])
        plt.text(x + 6, y + 6, str(i), fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"✅ Wrote detection diagnostic: {out_png}")

def _edge_ok(x, y, nx, ny, edge_margin):
    return (x > edge_margin) and (x < nx - edge_margin) and (y > edge_margin) and (y < ny - edge_margin)

def _far_enough(x, y, chosen_xy, min_sep):
    for (cx, cy) in chosen_xy:
        if (x - cx)**2 + (y - cy)**2 < min_sep**2:
            return False
    return True

def auto_make_ds9_reg_from_cube(
    cube_path: Path,
    out_reg_path: Path,
    *,
    n_ref_frames=10,
    fwhm_guess=4.0,
    threshold_sigma=5.0,
    edge_margin=60,
    n_comps=12,
    min_sep=35.0,
    flux_ratio_range=(0.2, 5.0),
    circle_radius=12.0,
    max_candidates_plot=80,
):
    """
    Auto-generate a DS9 region file from a reduced cube:
      - detects stars on a median reference image
      - user selects target by ID
      - auto-selects comparison stars
      - writes DS9 .reg circles in IMAGE pixel coords (1-based in file, but your loader subtracts 1)
    """

    print("\n🤖 AUTO_REG: building reference image...")
    ref = build_reference_image_from_cube(cube_path, n_frames=n_ref_frames)
    ny, nx = ref.shape

    print("🤖 AUTO_REG: detecting stars...")
    sources = detect_stars_daofinder(ref, fwhm_guess=fwhm_guess, threshold_sigma=threshold_sigma)
    if sources is None:
        raise SystemExit("❌ AUTO_REG: No stars detected. Try lowering threshold_sigma or changing fwhm_guess.")

    # Sort brightest-first by flux (DAOStarFinder provides 'flux' typically)
    # Some photutils versions use 'flux' or 'mag'. We try flux.
    if "flux" in sources.colnames:
        sources.sort("flux")
        sources.reverse()
    elif "peak" in sources.colnames:
        sources.sort("peak")
        sources.reverse()

    # Save plot + print a short list
    out_png = out_reg_path.with_suffix(".png")
    save_detection_plot(ref, sources, out_png, max_to_plot=max_candidates_plot)

    print("\nTop candidates (first ~20):")
    topn = min(20, len(sources))
    for i in range(topn):
        x = float(sources["xcentroid"][i])
        y = float(sources["ycentroid"][i])
        fl = float(sources["flux"][i]) if "flux" in sources.colnames else np.nan
        pk = float(sources["peak"][i]) if "peak" in sources.colnames else np.nan
        print(f"  ID {i:2d}: x={x:8.2f} y={y:8.2f}  flux={fl: .3e}  peak={pk: .3e}")

    # Ask user which ID is the target
    while True:
        s = input("\nEnter the TARGET star ID from the plot (e.g. 0, 1, 2, ...): ").strip()
        if s.isdigit():
            tid = int(s)
            if 0 <= tid < len(sources):
                break
        print("❌ Invalid ID. Try again.")

    tx = float(sources["xcentroid"][tid])
    ty = float(sources["ycentroid"][tid])
    tflux = float(sources["flux"][tid]) if "flux" in sources.colnames else None

    # Now pick comparison stars
    chosen = [(tx, ty)]
    circles = [(tx, ty, float(circle_radius))]

    lo, hi = flux_ratio_range

    for i in range(len(sources)):
        if i == tid:
            continue

        x = float(sources["xcentroid"][i])
        y = float(sources["ycentroid"][i])

        if not _edge_ok(x, y, nx, ny, edge_margin):
            continue
        if not _far_enough(x, y, chosen, min_sep):
            continue

        if tflux is not None and "flux" in sources.colnames:
            f = float(sources["flux"][i])
            r = f / tflux if tflux > 0 else np.inf
            if not (lo <= r <= hi):
                continue

        chosen.append((x, y))
        circles.append((x, y, float(circle_radius)))

        if len(circles) >= 1 + int(n_comps):
            break

    if len(circles) < 2:
        raise SystemExit("❌ AUTO_REG: Couldn’t find any comparison stars. Relax edge/sep/flux ratio constraints.")

    if len(circles) < 1 + int(n_comps):
        print(f"⚠️ AUTO_REG: Only found {len(circles)-1} comps (requested {n_comps}). "
              f"Consider lowering min_sep/edge_margin or widening flux_ratio_range.")

    # Write DS9 region file (your write_ds9_circles writes 1-based, which is correct for DS9)
    write_ds9_circles(out_reg_path, circles, add_header=True)
    print(f"✅ AUTO_REG: wrote region file: {out_reg_path}")
    print("   (Target is the first circle; comparison stars follow.)")

    return out_reg_path    



# --- TRACKING / CENTROIDING ---
def _circle_bbox(xc, yc, r, nx, ny):
    x1 = int(max(0, np.floor(xc - r)))
    x2 = int(min(nx, np.ceil(xc + r) + 1))
    y1 = int(max(0, np.floor(yc - r)))
    y2 = int(min(ny, np.ceil(yc + r) + 1))
    return x1, x2, y1, y2

def _gauss2d_const(coords, A, mux, muy, sx, sy, C):
    x, y = coords
    
    #Curvefit Function
    gx = (x - mux)**2 / (2*sx*sx)
    gy = (y - muy)**2 / (2*sy*sy)
    
    # Flatten to 1D for curve_fit
    return (C + A*np.exp(-(gx + gy))).ravel()

def gauss_centroid_and_fwhm(image, x0, y0, r):
    """
    Fit centroid + FWHM with guards against pathological frames.
    Returns (mux, muy, fwhm, peak, sky_rms, ok_flag).
    """
    ny, nx = image.shape
    r_fit = max(7.0, 2.5 * r)
    x1, x2, y1, y2 = _circle_bbox(x0, y0, r_fit, nx, ny)
    sub = image[y1:y2, x1:x2]

    yy, xx = np.mgrid[y1:y2, x1:x2]
    msk = (xx - x0) ** 2 + (yy - y0) ** 2 <= r_fit * r_fit

    x_data = xx[msk].astype(float)
    y_data = yy[msk].astype(float)
    z_data = sub[msk].astype(float)

    # Quick sanity checks
    n_pix = z_data.size
    if n_pix < 20:
        # Too few points to fit
        return float(x0), float(y0), np.nan, np.nan, np.nan, False

    if not np.isfinite(z_data).any():
        # All NaN/Inf
        return float(x0), float(y0), np.nan, np.nan, np.nan, False

    # Hard cap on pixels to avoid insane curve_fit cost
    MAX_PIX = 8000
    if n_pix > MAX_PIX:
        # Shrink the fit radius until below threshold
        factor = np.sqrt(MAX_PIX / n_pix)
        r_fit_new = max(7.0, r_fit * factor)
        x1, x2, y1, y2 = _circle_bbox(x0, y0, r_fit_new, nx, ny)
        sub = image[y1:y2, x1:x2]
        yy, xx = np.mgrid[y1:y2, x1:x2]
        msk = (xx - x0) ** 2 + (yy - y0) ** 2 <= r_fit_new * r_fit_new
        x_data = xx[msk].astype(float)
        y_data = yy[msk].astype(float)
        z_data = sub[msk].astype(float)
        n_pix = z_data.size
        if n_pix < 20 or not np.isfinite(z_data).any():
            return float(x0), float(y0), np.nan, np.nan, np.nan, False
        r_fit = r_fit_new

    # Background ring
    rin  = r_fit * 1.05
    rout = r_fit * 1.25
    rmask = ((xx - x0) ** 2 + (yy - y0) ** 2 >= rin * rin) & \
            ((xx - x0) ** 2 + (yy - y0) ** 2 <= rout * rout)

    if rmask.any():
        ring = sigma_clip(sub[rmask], 5.0, masked=True).filled(np.nan)
        sky_rms = float(np.nanstd(ring))
    else:
        sky_rms = np.nan

    C0 = float(np.nanmedian(z_data))
    A0 = float(np.nanmax(z_data) - C0)
    if not np.isfinite(A0) or A0 <= 0:
        
        # Basically flat, nothing to fit
        return float(x0), float(y0), np.nan, np.nan, sky_rms, False

    p0 = [A0, float(x0), float(y0),
          max(1.5, r_fit / 3), max(1.5, r_fit / 3), C0]

    bounds = (
        [0, x0 - r_fit, y0 - r_fit, 0.5, 0.5, -np.inf],
        [np.inf, x0 + r_fit, y0 + r_fit, 2 * r_fit, 2 * r_fit, np.inf]
    )

    try:
        popt, _ = curve_fit(
            _gauss2d_const,
            (x_data, y_data),
            z_data,
            p0=p0,
            bounds=bounds,
            maxfev=5000 
        )
    except Exception:
        return float(x0), float(y0), np.nan, np.nan, sky_rms, False

    A, mux, muy, sx, sy, C = popt
    sigma_eff = np.sqrt(0.5*(sx*sx + sy*sy))
    fwhm = 2.355 * sigma_eff
    peak = float(A + C)
    
    # Return fit values and success
    return float(mux), float(muy), fwhm, peak, sky_rms, True
                            


# --- PHOTOMETRY ---
def optimise_aperture_grid(
    aper_file: Path,
    *,
    cube_path: Path,
    reg_path: Path,
    data_root: Path | None = None,
    outdir: Path,
    inst: dict,
    targ: dict,
    r_ap_min: float = 8.0,
    r_ap_max: float = 25.0,
    r_ap_step: float = 1.0,
    r_in_scale: float = 1.5,
    r_out_scale: float = 2.5,
):
    """
    Grid-search aperture radii. Saves best to aper_file (.npz).
    Returns (rms_best, (r_ap, r_in, r_out)).
    """

    # Defensive: sane stepping
    if r_ap_step <= 0:
        raise ValueError("r_ap_step must be > 0")
    if r_ap_max <= r_ap_min:
        raise ValueError("r_ap_max must be > r_ap_min")

    best = None

    # Inclusive grid
    r_grid = np.arange(r_ap_min, r_ap_max + 0.5 * r_ap_step, r_ap_step)

    for r_ap in r_grid:
        r_in  = r_in_scale * r_ap
        r_out = r_out_scale * r_ap

        rms = main(
            cube_path, reg_path, outdir,
            fixed_radii=(float(r_ap), float(r_in), float(r_out)),
            grid_only=True,
            data_root=data_root,
            inst=inst,
            targ=targ,
            band=cube_path.stem.replace("cube_", "")
        )

        # Guard: main() can return inf
        if not np.isfinite(rms):
            continue

        if (best is None) or (rms < best[0]):
            best = (float(rms), (float(r_ap), float(r_in), float(r_out)))

    if best is None:
        raise RuntimeError("Aperture grid search produced no finite RMS values.")

    rms_best, best_radii = best
    np.savez(
        aper_file,
        r_ap=best_radii[0],
        r_in=best_radii[1],
        r_out=best_radii[2],
        rms=rms_best,
        r_ap_min=r_ap_min,
        r_ap_max=r_ap_max,
        r_ap_step=r_ap_step,
        r_in_scale=r_in_scale,
        r_out_scale=r_out_scale,
    )
    return rms_best, best_radii

def effective_area(aperture, image, *, method="exact"):
    """
    Return the effective (weighted) area of an aperture/annulus
    consistent with photutils aperture_photometry, excluding NaN/Inf pixels.
    """
    ny, nx = image.shape

    m = aperture.to_mask(method=method)
    m = m[0] if isinstance(m, (list, tuple)) else m

    bbox = m.bbox
    
    ysl = slice(int(bbox.iymin), int(bbox.iymax))
    xsl = slice(int(bbox.ixmin), int(bbox.ixmax))

    # Clip bbox to image bounds
    y0 = max(0, ysl.start)
    y1 = min(ny, ysl.stop)
    x0 = max(0, xsl.start)
    x1 = min(nx, xsl.stop)

    if y0 >= y1 or x0 >= x1:
        return 0.0

    w = np.asarray(m.data, dtype=float)
    w = w[(y0 - ysl.start):(y1 - ysl.start),
          (x0 - xsl.start):(x1 - xsl.start)]

    cut = np.asarray(image[y0:y1, x0:x1], dtype=float)
    finite = np.isfinite(cut)

    if not finite.any():
        return 0.0

    return float(np.sum(w[finite]))

def photometry(image, xc, yc, fwhm=None, radii=None, *,
               k_ap=10, k_in=15, k_out=25,
               method="exact",
               require_full_annulus=True):
    """
    Sky-subtracted aperture photometry for one star.
    Uses fixed radii if provided, otherwise scales by FWHM.

    Returns: (flux, sky_pp, r_ap, r_in, r_out, ok_flag)
    """
    ny, nx = image.shape

    # Choose radii
    if radii is not None:
        r_ap, r_in, r_out = radii
    else:
        if fwhm is None or not np.isfinite(fwhm):
            return np.nan, np.nan, np.nan, np.nan, np.nan, False
        r_ap  = max(2.5, k_ap * fwhm)
        r_in  = max(r_ap, k_in * fwhm)
        r_out = max(r_in, k_out * fwhm)

    # Boundary guard (full annulus must be on-frame)
    if require_full_annulus:
        if (xc - r_out) < 0 or (xc + r_out) >= nx or (yc - r_out) < 0 or (yc + r_out) >= ny:
            return np.nan, np.nan, float(r_ap), float(r_in), float(r_out), False

    ap  = CircularAperture((xc, yc), r=r_ap)
    ann = CircularAnnulus((xc, yc), r_in=r_in, r_out=r_out)

    # Weighted sums from photutils (handles partial edge pixels)
    phot = aperture_photometry(image, [ap, ann], method=method)
    ap_sum  = float(phot["aperture_sum_0"][0])
    ann_sum = float(phot["aperture_sum_1"][0])

    # Effective areas (sum of mask weights over finite pixels)
    ap_eff  = effective_area(ap,  image, method=method)
    ann_eff = effective_area(ann, image, method=method)

    if (ap_eff <= 0) or (ann_eff <= 0) or (not np.isfinite(ap_sum)) or (not np.isfinite(ann_sum)):
        return np.nan, np.nan, float(r_ap), float(r_in), float(r_out), False

    # Sky per effective pixel and sky-subtracted flux (consistent weighting)
    sky_pp = ann_sum / ann_eff
    flux   = ap_sum - sky_pp * ap_eff

    ok = np.isfinite(flux) and np.isfinite(sky_pp)
    return float(flux), float(sky_pp), float(r_ap), float(r_in), float(r_out), bool(ok)

# Build ensemble from comps                
def oot_mask_from_geometry(phase_centered, P, a_rs, inc_deg, *, margin_frac=0.75):
    """
    Build an OOT mask using only P, a/Rs, inc (assumes e=0).
    Uses a k=0 duration proxy and inflates by margin_frac for safety.
    Returns (oot_mask, half_phase_excluded).
    """
    inc = np.deg2rad(float(inc_deg))
    a_rs = float(a_rs)
    P = float(P)

    b = a_rs * np.cos(inc)

    # k=0 duration proxy
    num = np.sqrt(max(1.0 - b*b, 0.0))
    den = a_rs * np.sin(inc) + 1e-12
    arg = np.clip(num / den, 0.0, 1.0)

    T14 = (P / np.pi) * np.arcsin(arg)   # days (approx)
    half_phase = 0.5 * (T14 / P)         # phase units

    # conservative buffer
    half_phase *= (1.0 + float(margin_frac))

    oot = np.abs(np.asarray(phase_centered, float)) > half_phase
    return oot, float(half_phase)


def build_ensemble_weighted(comp_array, oot_mask, *, floor=1e-6):
    """
    OOT-weighted ensemble:
      1) Normalize each comparison by its OOT median.
      2) Compute OOT scatter per comp; weights w_j = 1/sigma_j^2.
      3) Ensemble is weighted mean across comps per frame.

    Returns: ens (N,), rel_flux (N,ncomp), weights (ncomp,)
    """
    comp = np.asarray(comp_array, float)
    oot_mask = np.asarray(oot_mask, bool)

    med_oot = np.nanmedian(comp[oot_mask, :], axis=0)
    med_oot = np.where(np.isfinite(med_oot) & (med_oot > 0), med_oot, np.nan)

    rel = comp / med_oot[None, :]

    sig = np.nanstd(rel[oot_mask, :], axis=0)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nan)

    w = 1.0 / np.clip(sig, floor, np.inf) ** 2
    w[~np.isfinite(w)] = 0.0

    num = np.nansum(rel * w[None, :], axis=1)
    den = np.nansum(np.isfinite(rel) * w[None, :], axis=1)
    ens = num / den

    return ens, rel, w

def build_ensemble(comp_array):           

    # per-star scaling constants          
    M_j = np.nanmedian(comp_array, axis=0)   

    # Relative flux per comp       
    Rel_flux  = comp_array / M_j[None, :]   

    # Per-frame median across comps              
    ens = np.nanmedian(Rel_flux, axis=1)                      
    return ens, Rel_flux, M_j
    
  
    
# --- TRANSIT MODELLING ---
def batman_flux(rp, dt_days, c0, c1, x_phase, period, a_rs, inc_deg, u1, u2):
    """
    Baseline + BATMAN transit evaluated on x_phase (phase array).

    Parameters 
    ----------
    rp : float
        Planet-to-star radius ratio (Rp/Rs).
    dt_days : float
        Mid-transit time offset *in days* relative to nominal epoch.
    c0, c1 : float
        Linear baseline coefficients: flux = c0 + c1 * phase + transit_model - 1.
    x_phase : array
        Orbital phase array (centered around 0), e.g. [-0.5, 0.5].
    period : float
        Orbital period in days.
    a_rs : float
        Scaled semi-major axis a/Rs.
    inc_deg : float
        Inclination in degrees.
    u1, u2 : float
        Quadratic limb darkening coefficients.
    """
    # Convert phase -> days
    t = x_phase * float(period)

    p = batman.TransitParams()
    p.t0 = float(dt_days)
    p.per = float(period)
    p.rp  = float(rp)
    p.a   = float(a_rs)
    p.inc = float(inc_deg)
    p.ecc = 0.0
    p.w   = 90.0
    p.u   = [float(u1), float(u2)]
    p.limb_dark = "quadratic"

    tm = batman.TransitModel(p, t)
    
    # Normalized around 1
    mod = tm.light_curve(p)  

    return c0 + c1 * x_phase + (mod - 1.0)

def log_prior(theta, u1_0, u2_0, sig_u):
    rp, dt, c0, c1, u1, u2, log10_sj = theta

    lp  = -0.5 * ((c0 - 1.0) / 0.05) ** 2
    lp += -0.5 * (c1 / 0.5) ** 2

    lp += -0.5 * ((u1 - u1_0) / sig_u) ** 2
    lp += -0.5 * ((u2 - u2_0) / sig_u) ** 2

    if not (-6.0 < log10_sj < -1.0):
        return -np.inf

    return lp

def log_likelihood(theta, x_phase, y_flux, y_sigma, period, a_rs, inc):
    rp, dt, c0, c1, u1, u2, log10_sj = theta
    sj = 10 ** log10_sj
    mu = batman_flux(rp, dt, c0, c1, x_phase, period, a_rs, inc, u1, u2)
    var = y_sigma**2 + sj**2
    r = y_flux - mu
    return -0.5 * np.sum(r*r/var + np.log(2*np.pi*var))

def log_posterior(theta, x_phase, y_flux, y_sigma, period, a_rs, inc, u1_0, u2_0, sig_u):
    lp = log_prior(theta, u1_0, u2_0, sig_u)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x_phase, y_flux, y_sigma, period, a_rs, inc)



# --- USER INTERFACE ---
def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """
    Simple Y/n prompt. default=True means Enter -> Yes.
    """
    s = input(prompt).strip().lower()
    if not s:
        return default
    if s in ("y", "yes"):
        return True
    if s in ("n", "no"):
        return False
    print("❌ Please answer y or n.")
    return prompt_yes_no(prompt, default=default)

def prompt_for_data_dir() -> Path:
    while True:
        s = input("\nPaste the path to your INT data folder (either the folder named 'data' or the project folder containing it):\n> ").strip().strip('"')

        if not s:
            print("❌ Empty path. Try again.")
            continue

        p = Path(s).expanduser()

        # If user pasted the project root, use root/data
        if p.is_dir() and (p / "data").is_dir():
            return (p / "data").resolve()

        # If user pasted the data dir itself
        if p.is_dir() and p.name.lower() == "data":
            return p.resolve()

        print("❌ I couldn't find a 'data' folder there. Paste either:\n"
              "   - the folder that IS named 'data'\n"
              "   - OR the parent folder that CONTAINS 'data'\n")
        
def choose_from_list(title, items):
    items = list(items)
    if not items:
        raise SystemExit(f"No options available for {title}.")
    print(f"\n{title}:")
    for i, x in enumerate(items, start=1):
        print(f"  [{i}] {x}")
    while True:
        s = input("Select number: ").strip()
        if s.isdigit() and 1 <= int(s) <= len(items):
            return items[int(s)-1]

def select_files_from_master(
    master_csv: Path,
    *,
    obj_override: str | None = None,
    filt_override: str | None = None,
    night_override: str | None = None,
):
    df = pd.read_csv(master_csv)

    # Normalize key columns
    df["OBSTYPE"] = df["OBSTYPE"].astype(str).str.upper().str.strip()
    df["OBJECT"]  = df["OBJECT"].astype(str).str.strip()
    df["FILTER"]  = df["FILTER"].astype(str).str.strip()
    df["NIGHT"]   = df["NIGHT"].astype(str).str.strip()

    # Restrict to things we care about
    df = df[df["OBSTYPE"].isin(["TARGET", "FLAT", "BIAS"])].copy()

    # - OBJECT -
    objects = sorted(df.loc[df["OBSTYPE"] == "TARGET", "OBJECT"].unique())
    if obj_override is not None:
        if obj_override not in objects:
            raise SystemExit(f"OBJECT '{obj_override}' not found. Available: {objects}")
        obj = obj_override
        print(f"🧭 Using OBJECT from CLI: {obj}")
    else:
        obj = choose_from_list("Choose OBJECT", objects)

    df_obj = df[df["OBJECT"] == obj]

    # - FILTER -
    filters = sorted(df_obj.loc[df_obj["OBSTYPE"] == "TARGET", "FILTER"].unique())
    if filt_override is not None:
        if filt_override not in filters:
            raise SystemExit(f"FILTER '{filt_override}' not found for OBJECT={obj}. Available: {filters}")
        filt = filt_override
        print(f"🧭 Using FILTER from CLI: {filt}")
    else:
        filt = choose_from_list("Choose FILTER", filters)

    df_of = df_obj[df_obj["FILTER"] == filt]

    # - NIGHT -
    nights = sorted(df_of.loc[df_of["OBSTYPE"] == "TARGET", "NIGHT"].unique())
    if night_override is not None:
        if night_override not in nights:
            raise SystemExit(f"NIGHT '{night_override}' not found for OBJECT={obj}, FILTER={filt}. Available: {nights}")
        night = night_override
        print(f"🧭 Using NIGHT from CLI: {night}")
    else:
        night = choose_from_list("Choose NIGHT", nights)

    # - Final selection -
    sci = df[(df["OBSTYPE"]=="TARGET") & (df["OBJECT"]==obj) & (df["FILTER"]==filt) & (df["NIGHT"]==night)]
    flats = df[(df["OBSTYPE"]=="FLAT") & (df["FILTER"]==filt) & (df["NIGHT"]==night)]
    bias  = df[(df["OBSTYPE"]=="BIAS") & (df["NIGHT"]==night)]

    sci_files  = [Path(p) for p in sci["filepath"].tolist()]
    flat_files = [Path(p) for p in flats["filepath"].tolist()]
    bias_files = [Path(p) for p in bias["filepath"].tolist()]

    print("\nSelection summary:")
    print(f"  OBJECT={obj}")
    print(f"  FILTER={filt}")
    print(f"  NIGHT ={night}")
    print(f"  science={len(sci_files)}  flats={len(flat_files)}  bias={len(bias_files)}")

    if not sci_files:
        raise SystemExit("No science files matched.")
    if not flat_files:
        raise SystemExit("No flats matched.")
    if not bias_files:
        raise SystemExit("No bias matched.")

    return obj, filt, night, sci_files, flat_files, bias_files

def ensure_target_cfg_complete(targ: dict) -> dict:
    def ask_float(prompt):
        while True:
            s = input(prompt).strip()
            try:
                return float(s)
            except Exception:
                print("❌ Please enter a number.")

    # Use .get so missing keys don't crash
    if _is_missing(targ.get("T0_BJD")):
        targ["T0_BJD"] = ask_float("Enter T0_BJD (BJD_TDB) for this target: ")

    if _is_missing(targ.get("Period")):
        targ["Period"] = ask_float("Enter Period (days) for this target: ")

    if _is_missing(targ.get("A_RS")):
        targ["A_RS"] = ask_float("Enter a/Rs for this target: ")

    if _is_missing(targ.get("INC")):
        targ["INC"] = ask_float("Enter inclination (deg) for this target: ")

    return targ

def prompt_aperture_grid_defaults():
    # Default
    return 4.0, 11.0, 0.2

def prompt_aperture_grid_custom(*, hard_max_evals: int = 200):
    """
    Ask user for custom aperture grid.
    Returns (r_ap_min, r_ap_max, r_ap_step)
    Clamps step to keep evaluations <= hard_max_evals.
    """
    def ask_float(msg, *, lo=None, hi=None):
        while True:
            s = input(msg).strip()
            try:
                v = float(s)
            except Exception:
                print("❌ Enter a number.")
                continue
            if lo is not None and v < lo:
                print(f"❌ Must be >= {lo}")
                continue
            if hi is not None and v > hi:
                print(f"❌ Must be <= {hi}")
                continue
            return v

    print("\nCustom aperture grid setup:")
    print("  Sensible starting points: r_ap min ~ 3–5, max ~ 10–16, step ~ 0.1–0.3")

    rmin = ask_float("  r_ap min (px): ", lo=1.5, hi=80.0)
    rmax = ask_float("  r_ap max (px): ", lo=rmin + 0.1, hi=120.0)
    step = ask_float("  step (px): ", lo=0.05, hi=5.0)

    # Clamp to avoid insane grids
    n_eval = int(np.floor((rmax - rmin) / step)) + 1
    if n_eval > hard_max_evals:
        new_step = (rmax - rmin) / max(1, (hard_max_evals - 1))
        print(f"⚠️ Grid would evaluate {n_eval} apertures; clamping step to {new_step:.4f} "
              f"to keep <= {hard_max_evals}.")
        step = new_step

    return float(rmin), float(rmax), float(step)




# --- HIGH-LEVEL ORCHESTRATION ---
def merge_photometry_with_headers(data_root: Path, band: str):
    outputs = data_root / "outputs"
    photdir = data_root / "photometry_outputs" / band

    phot_csv = photdir / f"photometry_raw_cube_{band}.csv"
    man_csv  = outputs / f"cube_manifest_{band}.csv"
    hdr_csv  = data_root / "photometry_outputs" / "Header_Master.csv"

    df_phot = pd.read_csv(phot_csv)
    df_man  = pd.read_csv(man_csv)
    df_hdr  = pd.read_csv(hdr_csv)

    # Photometry ↔ manifest (cube_idx)
    df = df_phot.merge(df_man, on="cube_idx", how="left", validate="one_to_one")

    # Manifest ↔ header master (join_key)
    df = df.merge(df_hdr, on="join_key", how="left", suffixes=("", "_hdr"))

    out = photdir / f"final_merged_{band}.csv"
    df.to_csv(out, index=False)

def load_time_airmass_from_master(project_root: Path, cube_path: Path):
    outputs = project_root / "outputs"
    master_csv = outputs / "Header_Master.csv"

    band = cube_path.stem.replace("cube_", "")
    man_csv = outputs / f"cube_manifest_{band}.csv"

    if not man_csv.is_file():
        raise FileNotFoundError(f"Missing manifest: {man_csv}")
    if not master_csv.is_file():
        raise FileNotFoundError(f"Missing Header_Master: {master_csv}")

    df_man = pd.read_csv(man_csv)
    df_hdr = pd.read_csv(master_csv)

    # Normalize join_key
    df_man["join_key"] = df_man["join_key"].astype(str)
    df_hdr["join_key"] = df_hdr["join_key"].astype(str)

    # Only pull the needed columns so names stay stable
    needed = ["join_key", "BJD_TDB", "AIRMASS"]
    missing = [c for c in needed if c not in df_hdr.columns]
    if missing:
        raise KeyError(f"Header_Master missing columns: {missing}. Has: {list(df_hdr.columns)}")

    df = df_man.merge(
        df_hdr[needed],
        on="join_key",
        how="left",
        validate="many_to_one"
    )

    # Align by cube_idx
    if "cube_idx" not in df.columns:
        raise KeyError(f"{man_csv.name} missing cube_idx column.")
    df["cube_idx"] = pd.to_numeric(df["cube_idx"], errors="coerce")
    df = df.sort_values("cube_idx").reset_index(drop=True)

    t_bjd = pd.to_numeric(df["BJD_TDB"], errors="coerce").to_numpy(float)
    airmass = pd.to_numeric(df["AIRMASS"], errors="coerce").to_numpy(float)

    return t_bjd, airmass

def run_pipeline(args=None):
    if args is None:
        class Dummy: pass
        args = Dummy()
        args.data_dir = None
        args.targets = None
        args.instrument = None
        args.target = None
        args.band = None
        args.night = None
        args.outdir = None
        args.verbose = False
        args.seed = None

    script_root = Path(__file__).resolve().parent

    # Config paths (CLI overrides, else defaults beside script)
    targets_path = Path(args.targets) if args.targets else (script_root / "targets.json")
    inst_path    = Path(args.instrument) if args.instrument else (script_root / "instrument.json")

    # Data root: CLI override else existing prompt
    data_root = Path(args.data_dir) if args.data_dir else prompt_for_data_dir()
    project_root = Path(data_root).parent

    # Outdir: CLI override else default under project_root
    outdir = Path(args.outdir) if args.outdir else (project_root / "outputs")
    outdir.mkdir(exist_ok=True, parents=True)

    # Seed: CLI override else default 
    seed = int(args.seed) if args.seed is not None else 42

    # 1.) Header scan + master table
    print("\n🔎 Searching FITS files for options...")
    rows = collect_headers(data_root, obstype_keep=("TARGET", "FLAT", "BIAS"))

    df_master = make_master_table(rows, outdir, "Header_Master", write_master=False)
    df_master = rescue_flats_if_needed(df_master, min_flats=3)

    master_csv = outdir / "Header_Master.csv"
    df_master.to_csv(master_csv, index=False)

    # 2.) Run selection (CLI overrides optional, otherwise your interactive selection)
    obj, filt, night, sci_files, flat_files, bias_files = select_files_from_master(
        master_csv,
        obj_override=args.target,
        filt_override=args.band,
        night_override=args.night
    )

    # 3.) Load configs (instrument + target DB)
    inst = load_instrument_config(inst_path)

    # Targets DB keyed by target name
    targets_db = json.loads(Path(targets_path).read_text(encoding="utf-8"))

    target_name = resolve_target_name_from_object(targets_db, obj)
    targ = dict(targets_db[target_name])
    targ["name"] = target_name
    targ = ensure_target_cfg_complete(targ)

    # 4.) Run metadata
    write_run_metadata(outdir, seed=seed, data_root=data_root, target=target_name)

    # 5.) Data reduction
    print("\n🤖 Running Data Reduction...")
    cubes_dict, master_bias, flat_map = Data_Reduction(sci_files, flat_files, bias_files, outdir, inst)

    # 6.) Photometry + merge per produced band
    for band in sorted(cubes_dict.keys()):
        reg_path = prompt_for_reg_file(project_root, band)
        run_photometry_for_band(project_root, band, reg_path=reg_path, inst=inst, targ=targ)
        merge_photometry_with_headers(project_root, band)
        
def run_photometry_for_band(data_root: Path, band: str, reg_path: Path, *, inst: dict, targ: dict):
    outputs = data_root / "outputs"
    photdir = data_root / "photometry_outputs" / band
    photdir.mkdir(parents=True, exist_ok=True)

    cube_path = outputs / f"cube_{band}.fits"

    # Default reg name if user doesn't supply one
    if reg_path is None:
        reg_path = data_root / f"{band}_targets.reg"

    # AUTO-REG block
    if AUTO_REG and (AUTO_REG_FORCE or (not reg_path.is_file())):
        print(f"\n🤖 AUTO_REG enabled for band={band}")
        auto_make_ds9_reg_from_cube(
            cube_path=cube_path,
            out_reg_path=reg_path,
            n_ref_frames=10,
            fwhm_guess=4.0,
            threshold_sigma=5.0,
            edge_margin=60,
            n_comps=12,
            min_sep=35.0,
            flux_ratio_range=(0.2, 5.0),
            circle_radius=12.0,
        )
    else:
        if not reg_path.is_file():
            raise SystemExit(f"❌ No reg file found at {reg_path}. Set AUTO_REG=True or create one manually.")
        print(f"✅ Using reg file: {reg_path}")

    outdir = photdir

    aper_file = outdir / f"best_aperture_{band}.npz"
    print("Looking for best-aperture file at:", aper_file)

    if aper_file.is_file():
        data = np.load(aper_file)
        best_radii = (float(data["r_ap"]), float(data["r_in"]), float(data["r_out"]))
        print("Loaded best aperture from file:", best_radii)

    else:
        print("🔍 No aperture file found — need to run grid search.")

        use_default = prompt_yes_no(
            "Use default aperture grid (r_ap 4..11 step 0.2)? [Y/n]: ",
            default=True
        )

        if use_default:
            r_ap_min, r_ap_max, r_ap_step = 4.0, 11.0, 0.2
        else:
            r_ap_min, r_ap_max, r_ap_step = prompt_aperture_grid_custom(hard_max_evals=200)

        print(f"🔧 Grid: r_ap in [{r_ap_min:.3f}, {r_ap_max:.3f}] step {r_ap_step:.3f} "
              f"(r_in=1.5*r_ap, r_out=2.5*r_ap)")

        rms_best, best_radii = optimise_aperture_grid(
            aper_file,
            cube_path=cube_path,
            reg_path=reg_path,
            data_root=data_root,
            outdir=outdir,
            inst=inst,
            targ=targ,
            r_ap_min=r_ap_min,
            r_ap_max=r_ap_max,
            r_ap_step=r_ap_step,
            r_in_scale=1.5,
            r_out_scale=2.5,
        )
        print(f"✅ Grid search best RMS={rms_best:.6e}, radii={best_radii}")

    print("\n🚀 Running main photometry...")
    main(
        cube_path, reg_path, outdir,
        fixed_radii=best_radii, grid_only=False,
        data_root=data_root, band=band,
        inst=inst, targ=targ
    )

def main(cube_path: Path, reg_path: Path, outdir: Path,
         fixed_radii=None, grid_only=False,
         data_root: Path | None = None,
         band: str | None = None,
         inst: dict | None = None,
         targ: dict | None = None,
         ):
    photdir = outdir
    photdir.mkdir(parents=True, exist_ok=True)

    with fits.open(cube_path, memmap=True) as hdul:
        cube = hdul[0].data       
        
    if band is None:
        band = cube_path.stem.replace("cube_", "")
    band = str(band).strip()

    if inst is None or targ is None:
        raise ValueError("main() requires inst and targ configs.")
        
    gain = inst["gain_e_per_adu"]
    read_noise = float(inst.get("read_noise_e", float("nan")))
    if not np.isfinite(read_noise):
        print("⚠️ read_noise_e missing/NaN. Setting read_noise_e=0 for noise model (photometry still runs).")
        read_noise = 0.0
    start_frame = int(inst.get("start_frame", 0))
    
    T0_BJD = float(targ["T0_BJD"])
    Period = float(targ["Period"])
    A_RS   = float(targ["A_RS"])
    INC    = float(targ["INC"])
    
    ld = targ.get("limb_darkening", {})
    if not isinstance(ld, dict) or not ld:
        raise KeyError(f"Target config missing 'limb_darkening'. targ keys={list(targ.keys())}")
    
    if band not in ld:
        raise KeyError(f"No LD entry for band='{band}'. Available: {list(ld.keys())}")
    
    U1_0  = float(ld[band]["U1_0"])
    U2_0  = float(ld[band]["U2_0"])

    N, ny, nx = cube.shape                              
    
    # Optional fixed aperture override
    if fixed_radii is not None:
        r_ap, r_in, r_out = fixed_radii
        print(f"⚙️  Using fixed aperture: r_ap={r_ap:.2f}, r_in={r_in:.2f}, r_out={r_out:.2f}")
    else:
        r_ap = r_in = r_out = None

    regs = load_ds9_circles(reg_path, True)
    if len(regs) < 2:
        raise SystemExit("❌ Region file must contain target + at least 1 comparison star circle.")
    
    target_pos0 = (regs[0][0], regs[0][1])
    comps0      = regs[1:]
    comps_pos0  = [(x, y) for (x, y, _r) in comps0]
    comps_r     = [r for (_x, _y, r) in comps0]

    # Minimum number of comparison stars required for a frame to be considered usable.
    n_comps_total = len(comps_pos0)
    min_good_comps = int(inst.get("min_good_comps", 5))
    min_good_comps = max(1, min(min_good_comps, n_comps_total))
    if min_good_comps < min(3, n_comps_total):
        min_good_comps = min(3, n_comps_total)
        
    ref_pos0 = comps_pos0[0]
    ref_r    = comps_r[0]
        
    # Working positions (set per frame)
    target_pos = target_pos0
    comps_pos  = list(comps_pos0)
    ref_pos    = ref_pos0
                                                           
    fwhm_prev = np.nan
    dx_prev = 0.0
    dy_prev = 0.0
        
    all_target, all_comps, target_noise, target_flux_e, fwhm_list, sky_list, comp_fluxes = [], [], [], [], [], [], []
    frames = np.arange(start_frame, N, dtype=int)

    # Photometry Loop >>>>>> 
    for i in tqdm(frames, desc="Aperture photometry", ncols=90):
        img = cube[i]
        ny, nx = img.shape
        
        # 1.) Fit reference star
        rx, ry, rfwhm, rpk, rrms, rok = gauss_centroid_and_fwhm(img, ref_pos[0], ref_pos[1], ref_r)  
        
        # 2.) Guard: if reference fit failed, don't update drift; reuse last drift and last good fwhm
        if (not rok) or (not np.isfinite(rx)) or (not np.isfinite(ry)):
            dx, dy = dx_prev, dy_prev
            rfwhm_use = fwhm_prev
        else:
            dx = rx - ref_pos[0]
            dy = ry - ref_pos[1]
            ref_pos = (rx, ry)
            dx_prev, dy_prev = dx, dy
            rfwhm_use = rfwhm
        
        # 3.) Guard: if fwhm is bad, reuse previous
        if (rfwhm_use is None) or (not np.isfinite(rfwhm_use)) or (rfwhm_use < 0.5) or (rfwhm_use > 15):
            rfwhm_use = fwhm_prev
        else:
            fwhm_prev = float(rfwhm_use)
        
        # 4.) Apply drift to target and comparison stars
        if rok and np.isfinite(rx) and np.isfinite(ry):
            dx = rx - ref_pos0[0]
            dy = ry - ref_pos0[1]
            dx_prev, dy_prev = dx, dy
        else:
            dx, dy = dx_prev, dy_prev
        
        target_pos = (target_pos0[0] + dx, target_pos0[1] + dy)
        comps_pos  = [(px + dx, py + dy) for (px, py) in comps_pos0]                            
        
        # 5.) Target/comp photometry
        radii = fixed_radii if (fixed_radii is not None) else None

        # Target
        t_flux, sky_pp, r_ap, r_in, r_out, tok = photometry(
            img, target_pos[0], target_pos[1],
            fwhm=rfwhm_use, radii=radii,
            require_full_annulus=True
        )

        
        # Comps (allow partial failures; store NaN for failed comps)
        comp_fluxes = []
        comp_ok_flags = []
        for px, py in comps_pos:
            flux, _, _, _, _, ok = photometry(
                img, px, py,
                fwhm=rfwhm_use, radii=radii,
                require_full_annulus=True
            )
            comp_fluxes.append(float(flux) if (ok and np.isfinite(flux)) else np.nan)
            comp_ok_flags.append(bool(ok))
        
        n_good_comps = int(np.sum(np.isfinite(comp_fluxes)))
        
        # 6.) Frame-level validity
        # Keep the frame if the target is OK and we have enough usable comps.
        frame_ok = bool(tok) and np.isfinite(t_flux) and np.isfinite(sky_pp) and np.isfinite(r_ap) and (n_good_comps >= min_good_comps)

        if not frame_ok:
            # Keep lengths consistent but do NOT contaminate ensemble
            target_noise.append(np.nan)
            target_flux_e.append(np.nan)

            all_target.append(np.nan)
            sky_list.append(np.nan)
            all_comps.append([np.nan] * len(comps_pos))

            fwhm_list.append(float(rfwhm_use) if np.isfinite(rfwhm_use) else np.nan)
            continue

        # 7.) Noise (target only, electrons)
        t_flux_e = float(t_flux) * gain
        sky_pp_e = float(sky_pp) * gain
        n_pix = float(np.pi * (r_ap ** 2))

        noise = np.sqrt(max(t_flux_e + n_pix * (sky_pp_e + read_noise**2), 0.0))
        target_noise.append(float(noise))
        target_flux_e.append(float(t_flux_e))

        # 8.) Store fluxes
        all_target.append(float(t_flux))
        sky_list.append(float(sky_pp))
        all_comps.append([float(v) if np.isfinite(v) else np.nan for v in comp_fluxes])

        fwhm_list.append(float(rfwhm_use) if np.isfinite(rfwhm_use) else np.nan)

  
    # 9.) Length Guard
    L = min(len(all_target), len(fwhm_list), len(sky_list))
    if L == 0:
        raise SystemExit("❌ No photometry outputs recorded.")
    
    # Trim everything to same length
    all_target = all_target[:L]
    fwhm_list  = fwhm_list[:L]
    sky_list   = sky_list[:L]
    
    # 10.) Save details
    res = pd.DataFrame({
            "cube_idx": np.arange(len(all_target), dtype=int),
            "t_flux": np.array(all_target, float),
            "fwhm": np.array(fwhm_list, float),
            "sky_pp": np.array(sky_list, float),
    })
    res.to_csv(outdir / f"photometry_raw_{cube_path.stem}.csv", index=False)# - Lightcurve construction -
    target_array = np.array(all_target, float)
    comp_array   = np.array(all_comps,  float)
    
    # Time/airmass from Header_Master via cube_manifest
    t_bjd, airmass = load_time_airmass_from_master(data_root, cube_path)
    
    # Compute orbital phase (centered around 0)
    tmid  = np.nanmedian(t_bjd)
    n_near = np.round((tmid - T0_BJD) / Period)
    Tref  = T0_BJD + n_near * Period
    phase = ((t_bjd - Tref) / Period) % 1.0
    phase_centered = ((phase + 0.5) % 1.0) - 0.5
    
    # Dataset-specific OOT mask (geometry-based; no Rp used)
    half_phase_excl = None
    for _margin in (0.75, 0.50, 0.25, 0.10):
        OOT_MASK, half_phase_excl = oot_mask_from_geometry(
            phase_centered, Period, A_RS, INC, margin_frac=_margin
        )
        if np.sum(OOT_MASK) >= 30:
            break
    if np.sum(OOT_MASK) < 20:
        print("⚠️ Very few OOT points available; ensemble weighting/ranking may be unreliable.")
    else:
        print(f"✅ OOT mask: excluding |phase| <= {half_phase_excl:.5f} (conservative)")
    
    # Build OOT-weighted ensemble
    ens, Cn_kept, w = build_ensemble_weighted(comp_array, OOT_MASK)
    
    # Differential light curve in absolute units (ADU / dimensionless ensemble)
    lc_abs = target_array / ens
    
    beta = float(inst.get("beta_factor", 1.0))
    
    lc_err_abs = np.abs(lc_abs) * np.sqrt(
        (np.array(target_noise) / np.array(target_flux_e))**2
    )
    
    # Comparison-star margin
    lc_err_abs *= 1.2
    
    # Time-correlated noise inflation
    lc_err_abs *= beta
    
    # Normalise light curve
    scale = np.nanmedian(lc_abs)
    lc_abs     = lc_abs / scale
    lc_err_abs = lc_err_abs / scale
    
    lc_abs      = np.array(lc_abs, float)
    lc_err_abs  = np.array(lc_err_abs, float)
    lc_err_abs  = np.abs(lc_err_abs)
    
    # -- Apply finite mask + ordering (now everything exists) --
    g = np.isfinite(phase_centered) & np.isfinite(lc_abs) & np.isfinite(lc_err_abs) & np.isfinite(airmass)
    order = np.argsort(phase_centered[g])
    # Diagnostics
    total_frames = len(phase_centered)
    used_frames = np.sum(g)
    rejected_frames = total_frames - used_frames
    print("\nFrame diagnostic:")
    print(f"  Total frames:   {total_frames}")
    print(f"  Used (finite):  {used_frames}")
    print(f"  Rejected:       {rejected_frames} ({100*rejected_frames/total_frames:.1f}%)")
    
    summary_path = photdir / "photometry_frame_quality.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Total frames: {total_frames}\n")
        f.write(f"Used (finite): {used_frames}\n")
        f.write(f"Rejected: {rejected_frames} ({100*rejected_frames/total_frames:.2f}%)\n")
    print(f"📝 Wrote frame-quality summary: {summary_path.name}")

    airmass = np.asarray(airmass, dtype=float)

    # 3.) Define data for plotting
    x  = phase_centered[g][order]
    y  = lc_abs[g][order]
    yerr = lc_err_abs[g][order]
    air = airmass[g][order]

    # 4.) Plot
    plt.figure(figsize=(10,4))
    plt.errorbar(x, y, yerr=yerr, fmt='.', ecolor='black', elinewidth=1, capsize=2, label="Data + errors")
    plt.scatter(x, y, s=8, color='blue', alpha=0.6)
    plt.xlabel("Orbital phase", fontsize=14)
    plt.ylabel("Relative flux", fontsize=14)
    plt.title("Phase-folded light curve", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"lightcurve_phase_{cube_path.stem}.png", dpi=150)
    plt.close()

    # - FWHM and sky background vs time plot -
    fwhm_arr = np.array(fwhm_list, float)
    sky_arr  = np.array(sky_list,  float)

    n_fwhm = len(fwhm_arr)
    n_sky  = len(sky_arr)
    n_time = len(t_bjd)

    # Align lengths defensively (should all match N, but just in case)
    n = min(n_fwhm, n_sky, n_time)
    fwhm_arr = fwhm_arr[:n]
    sky_arr  = sky_arr[:n]
    t_plot   = t_bjd[:n]
    air_plot = airmass[:n]

    # Time relative to mid-run in hours for nicer axis
    t_mid_run = np.nanmedian(t_plot)
    t_rel_hrs = (t_plot - t_mid_run) * 24.0

    # FWHM vs time
    plt.figure(figsize=(8,3))
    plt.plot(t_rel_hrs, fwhm_arr, marker='.', lw=1)
    plt.xlabel("Time from mid-run [hours]")
    plt.ylabel("FWHM [pixels]")
    plt.title("Seeing (FWHM) vs time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"diagnostic_fwhm_vs_time_{cube_path.stem}.png", dpi=150)
    plt.close()

    # Background vs time
    plt.figure(figsize=(8,3))
    plt.plot(t_rel_hrs, sky_arr, marker='.', lw=1)
    plt.xlabel("Time from mid-run [hours]")
    plt.ylabel("Sky background [ADU / pixel]")
    plt.title("Sky background vs time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"diagnostic_sky_vs_time_{cube_path.stem}.png", dpi=150)
    plt.close()

    # Airmass vs time to compare
    plt.figure(figsize=(8,3))
    plt.plot(t_rel_hrs, air_plot[:n], marker='.', lw=1)
    plt.xlabel("Time from mid-run [hours]")
    plt.ylabel("Airmass")
    plt.title("Airmass vs time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"diagnostic_airmass_vs_time_{cube_path.stem}.png", dpi=150)
    plt.close()

    # -- Batman Plot -- 
    # 1.) Quick curve_fit to get starting values
    u1_0, u2_0, sig_u = get_ld_params(targ, band)
    def _f_curvefit(x_phase, rp, dt_days, c0, c1):
        return batman_flux(rp, dt_days, c0, c1, x_phase, Period, A_RS, INC, u1_0, u2_0)

    p0_cf = [0.12, 0.0, 1.0, 0.0]  # rp, dt_days, c0, c1
    bounds_cf = (
        [0.03, -0.02 * Period, 0.95, -1.0],   # lower
        [0.25,  0.02 * Period, 1.05,  1.0]    # upper
    )

    popt_cf, pcov_cf = curve_fit( _f_curvefit, x, y, p0=p0_cf, bounds=bounds_cf,
                                 sigma=yerr, absolute_sigma=True, maxfev=30000)

    rp_init, dt_init, c0_init, c1_init = popt_cf
    '''print(f"[curve_fit] rp={rp_init:.5f}, depth≈{100*rp_init**2:.2f}%  "
          f"dt={dt_init*24*60:.2f} min  c0={c0_init:.5f} c1={c1_init:.5f}")'''

    # 2.) MCMC setup: (rp, dt, c0, c1, u1, u2, log10_jitter)
    ndim, nwalkers = 7, 32
    start = np.array([rp_init, dt_init, c0_init, c1_init, U1_0, U2_0, -3.0])
    p0_walkers = start + 1e-4 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior,
        args=(x, y, yerr, Period, A_RS, INC, u1_0, u2_0, sig_u)
    )

    # Burn-in
    state = sampler.run_mcmc(p0_walkers, 1500, progress=False)
    sampler.reset()

    # Production
    sampler.run_mcmc(state, 3000, progress=False)

    chain = sampler.get_chain(flat=True)
    labels = ["rp", "dt [d]", "c0", "c1", "u1", "u2", "log10_sj"]

    med = np.median(chain, axis=0)
    lo  = np.percentile(chain, 16, axis=0)
    hi  = np.percentile(chain, 84, axis=0)


    # 3.) Median model evaluated on your actual x
    rp_m, dt_m, c0_m, c1_m, u1_m, u2_m, _ = med
    y_med = batman_flux(rp_m, dt_m, c0_m, c1_m,
                        x, Period, A_RS, INC, u1_m, u2_m)
    
    # 4.) Plot data + model + residuals
    resid = y - y_med  # residuals: data - model
    
    # Chi-squared metric for this aperture set
    valid = np.isfinite(resid) & np.isfinite(yerr) & (yerr > 0)
    chi2 = np.sum((resid[valid] / yerr[valid])**2)
    ndof = valid.sum() - len(popt_cf)
    chi2_red = chi2 / ndof
    if grid_only:
        # - OOT RMS metric for aperture optimisation -
    
        # Use SAME phase window as normal
        phase_min = -0.04
        phase_max =  0.04
        sel = (x >= phase_min) & (x <= phase_max)
        xg = x[sel]
        yg = y[sel]
    
        # OOT mask
        oot = (xg < -0.02) | (xg > 0.02)
        if np.sum(oot) < 10:
            
            # Reject bad apertures safely
            return np.inf   
    
        # Linear baseline using OOT only
        A = np.vstack([np.ones_like(xg[oot]), xg[oot]]).T
        coef, *_ = np.linalg.lstsq(A, yg[oot], rcond=None)
        base = (np.vstack([np.ones_like(xg), xg]).T) @ coef
        yg_detrended = yg / base * np.nanmedian(base)
        rms = np.nanstd(yg_detrended[oot])
        return rms


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    
    # Top panel: light curve + model 
    ax1.errorbar(x, y, yerr=yerr, fmt='.', ecolor='gray', elinewidth=1, capsize=2, label="Data")
    ax1.plot(x, y_med, 'r-', lw=2, label="Median BATMAN+MCMC model")
    ax1.set_ylabel("Relative flux", fontsize=14)
    ax1.set_title("Phase-folded light curve with MCMC transit fit", fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Bottom panel: residuals
    ax2.errorbar(x, resid, yerr=yerr, fmt='.', ecolor='gray', elinewidth=1, capsize=2)
    ax2.axhline(0.0, color='k', lw=1, linestyle='--')
    ax2.set_xlabel("Orbital phase", fontsize=14)
    ax2.set_ylabel("Residuals", fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / f"lightcurve_phase_mcmc_{cube_path.stem}.png", dpi=200)
    plt.close()




    # -- COMPARISON STAR ANALYSIS --
    print("\n--- Comparison Star Analysis (random subsets with per-subset model) ---")
    rng = np.random.default_rng(42)

    target_full       = target_array.astype(float)
    target_noise_arr  = np.array(target_noise,   dtype=float)
    target_flux_e_arr = np.array(target_flux_e, dtype=float)
    n_frames, n_comps = comp_array.shape

    print(f"Total comparison stars: {n_comps}")

    # Choose frames to use for ranking (limit for speed) 
    # Only frames where target & all comps are finite and positive.
    good_frame_mask = (
        np.isfinite(phase_centered) &
        np.isfinite(target_full) &
        (target_full > 0) &
        (np.sum(np.isfinite(comp_array), axis=1) >= 3)
    )
    idx_good = np.where(good_frame_mask)[0]

    if idx_good.size == 0:
        print("No good frames available for comparison-star analysis; skipping.")
        top_list = []
    else:
        # Use OOT-only frames for ranking (prevents transit leakage bias)
        idx_eval = idx_good[OOT_MASK[idx_good]]
    
    # Downsample for speed (OOT is safe, so you can use more than 60)
    N_limit = 300
    if idx_eval.size > N_limit:
        idx_eval = np.sort(rng.choice(idx_eval, size=N_limit, replace=False))
    
    if idx_eval.size < 20:
        print("⚠️ Not enough OOT frames for reliable comparison-star ranking; skipping.")
        top_list = []
    else:
        print(f"Using {idx_eval.size} OOT frames out of {n_frames} total for subset ranking.")
    
        phase_eval   = phase_centered[idx_eval]
        target_eval  = target_full[idx_eval]
        noise_eval   = target_noise_arr[idx_eval]
        tfe_eval     = target_flux_e_arr[idx_eval]
        comps_eval   = comp_array[idx_eval, :]
    
        # Phase window for evaluation (same as main LC)
        phase_min = -0.04
        phase_max =  0.04
        
        # - Helper: build LC and fit transit for a given subset -
        def build_and_fit_subset(js):
            """
            Subset score using OOT-only baseline:
              - OOT-normalize each comp
              - OOT-weight the ensemble (1/sigma^2 from OOT scatter)
              - build differential LC (target/ensemble)
              - detrend using OOT-only linear baseline
              - score = RMS of detrended OOT residuals (lower is better)
            Returns (x, y, yerr, None, score, None)
            """
            js = list(js)
            subset_flux = comps_eval[:, js]
        
            # OOT mask in the eval window
            oot_eval = OOT_MASK[idx_eval]
        
            # Guard: need enough OOT samples to normalize + weight
            if np.sum(oot_eval) < 10:
                return None
        
            # OOT-only normalization per star
            norm = np.nanmedian(subset_flux[oot_eval, :], axis=0)
            norm = np.where(np.isfinite(norm) & (norm > 0), norm, np.nan)
            rel = subset_flux / norm[None, :]
        
            # OOT scatter weights
            sig = np.nanstd(rel[oot_eval, :], axis=0)
            sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nan)
            w = 1.0 / np.clip(sig, 1e-6, np.inf) ** 2
            w[~np.isfinite(w)] = 0.0
        
            # Weighted ensemble per frame
            num = np.nansum(rel * w[None, :], axis=1)
            den = np.nansum(np.isfinite(rel) * w[None, :], axis=1)
            ens = num / den
        
            if (not np.all(np.isfinite(ens))) or np.any(ens <= 0):
                return None
        
            # Differential LC (absolute units)
            lc_abs = target_eval / ens
            if not np.all(np.isfinite(lc_abs)):
                return None
        
            # Error propagation (target noise only; keep your existing formula)
            lc_err_abs = lc_abs * np.sqrt((noise_eval / tfe_eval) ** 2)
            lc_err_abs *= 1.2  # comparison-star margin
            lc_err_abs *= beta
        
            # Normalize
            scale = np.nanmedian(lc_abs)
            if not np.isfinite(scale) or scale <= 0:
                return None
            y = lc_abs / scale
            yerr = lc_err_abs / scale
        
            # Sort by phase and restrict to window
            order_sub = np.argsort(phase_eval)
            x = phase_eval[order_sub]
            y = y[order_sub]
            yerr = yerr[order_sub]
        
            sel = (x >= phase_min) & (x <= phase_max)
            x = x[sel]
            y = y[sel]
            yerr = yerr[sel]
        
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
            x = x[valid]
            y = y[valid]
            yerr = yerr[valid]
        
            if x.size < 20:
                return None
        
            # OOT in this restricted window (use geometry-based half-width if available)
            if half_phase_excl is not None and np.isfinite(half_phase_excl):
                oot = (np.abs(x) > half_phase_excl)
            else:
                oot = (x < -0.02) | (x > 0.02)
        
            if np.sum(oot) < 10:
                return None
        
            # Linear baseline using OOT only
            y_raw = y.copy()
            A_oot = np.vstack([np.ones_like(x[oot]), x[oot]]).T
            coef, *_ = np.linalg.lstsq(A_oot, y_raw[oot], rcond=None)
            base_all = (np.vstack([np.ones_like(x), x]).T) @ coef
            base_med = np.nanmedian(base_all)
            y_detr = y_raw / base_all * base_med
        
            score = float(np.nanstd(y_detr[oot]))
            if not np.isfinite(score):
                return None
        
            return x, y_detr, yerr, None, score, None
        
        # - Random-subset search over comparison stars -
        min_comps = 9
        max_comps = min(12, n_comps)
        top_k     = 2
        
        # Random subsets per k
        max_eval_per_k = 100000               

        # Store (oot_rms, js_tuple, popt)
        top_heap = []   

        def maybe_add_to_top(oot_rms, js_tuple, popt_unused=None):
            """Maintain heap of best top_k (lowest OOT RMS) subsets."""
            item = (oot_rms, js_tuple, popt_unused)
        
            if len(top_heap) < top_k:
                heapq.heappush(top_heap, item)
            else:
                # top_heap[0] is currently the smallest RMS (best).
                # We want to keep only the best top_k; easiest: replace the worst.
                # Since heapq is a min-heap, we find the current worst among kept items.
                worst_item = max(top_heap, key=lambda t: t[0])
                if oot_rms < worst_item[0]:
                    top_heap.remove(worst_item)
                    heapq.heapify(top_heap)
                    heapq.heappush(top_heap, item)

        for k in range(min_comps, max_comps + 1):
            n_k = math.comb(n_comps, k)
            print(f"  Subset size k={k}: C({n_comps},{k}) = {n_k}")

            if n_k <= max_eval_per_k:
                iter_js = combinations(range(n_comps), k)
                total_iter = n_k
                mode = "all"
            else:
                print(f"    Sampling {max_eval_per_k} random subsets (out of {n_k}) for k={k}")
                seen = set()

                def random_js_gen():
                    while len(seen) < max_eval_per_k:
                        js = tuple(sorted(rng.choice(n_comps, size=k, replace=False)))
                        if js in seen:
                            continue
                        seen.add(js)
                        yield js

                iter_js = random_js_gen()
                total_iter = max_eval_per_k
                mode = "random"

            for js in tqdm(iter_js, total=total_iter,
                           desc=f"k={k} ({mode})", ncols=90):
                result = build_and_fit_subset(js)
                if result is None:
                    continue
                x_sub, y_sub, yerr_sub, y_model_sub, oot_rms, popt_sub = result
                maybe_add_to_top(oot_rms, tuple(js), popt_sub)


        # Extract and sort best subsets (ranked by OOT RMS; lower is better)
        top_list_raw = [(oot_rms, js, None) for (oot_rms, js, _unused) in top_heap]
        top_list_raw.sort(key=lambda t: t[0])  # sort by OOT RMS ascending

        top_list = []
        if top_list_raw:
            for rank, (oot_rms, js, _popt_unused) in enumerate(top_list_raw, start=1):
                js_list = list(js)
                size = len(js_list)
                top_list.append((oot_rms, js_list, size))
        else:
            top_list = []
                    
    # Convert noise lists to arrays once
    target_noise_arr   = np.array(target_noise, dtype=float)
    target_flux_e_arr  = np.array(target_flux_e, dtype=float)

    subset_mcmc_results = []

    for rank, (oot_rms, js, size) in enumerate(top_list, start=1):
        # ensure plain Python ints, not np.int64
        js = [int(j) for j in js]
    
        # 1.) Rebuild ensemble from this subset (all frames)
        subset_flux_full = comp_array[:, js]
        
        # OOT-weighted ensemble (consistent with ranking)
        oot_full = OOT_MASK
        norm_full = np.nanmedian(subset_flux_full[oot_full, :], axis=0)
        norm_full = np.where(np.isfinite(norm_full) & (norm_full > 0), norm_full, np.nan)
        rel_full = subset_flux_full / norm_full[None, :]
        
        sig_full = np.nanstd(rel_full[oot_full, :], axis=0)
        sig_full = np.where(np.isfinite(sig_full) & (sig_full > 0), sig_full, np.nan)
        w_full = 1.0 / np.clip(sig_full, 1e-6, np.inf) ** 2
        w_full[~np.isfinite(w_full)] = 0.0
        
        num_full = np.nansum(rel_full * w_full[None, :], axis=1)
        den_full = np.nansum(np.isfinite(rel_full) * w_full[None, :], axis=1)
        ens_subset_full = num_full / den_full
    
        # 2.) Build differential LC: target / ensemble
        target_full = target_array.astype(float)
        good = (np.isfinite(phase_centered) &
                np.isfinite(target_full) &
                np.isfinite(ens_subset_full) &
                (ens_subset_full > 0))

        if np.sum(good) < 20:
            print("  Too few good points for this subset, skipping.")
            continue

        phase_sub = phase_centered[good]
        tgt_sub   = target_full[good]
        ens_sub   = ens_subset_full[good]

        # Absolute differential flux
        lc_abs_sub = tgt_sub / ens_sub

        # Error propagation
        tn_sub  = target_noise_arr[good]
        tfe_sub = target_flux_e_arr[good]

        lc_err_abs_sub = lc_abs_sub * np.sqrt((tn_sub / tfe_sub)**2)
        lc_err_abs_sub *= 1.2 # Comparison star error margin
        lc_err_abs_sub *= beta
        
        # Normalise
        scale_sub = np.nanmedian(lc_abs_sub)
        y_sub     = lc_abs_sub / scale_sub
        yerr_sub  = lc_err_abs_sub / scale_sub

        # Sort by phase
        order_sub = np.argsort(phase_sub)
        x_sub     = phase_sub[order_sub]
        y_sub     = y_sub[order_sub]
        yerr_sub  = yerr_sub[order_sub]
        
        # 3.) Apply SAME OOT linear baseline -
        y_raw_sub = y_sub.copy()
        oot_sub = (x_sub < -0.02) | (x_sub > 0.02)

        if np.sum(oot_sub) >= 4:
            A_oot_sub = np.vstack([np.ones_like(x_sub[oot_sub]), x_sub[oot_sub]]).T
            coef_sub, *_ = np.linalg.lstsq(A_oot_sub, y_raw_sub[oot_sub], rcond=None)
            base_all_sub = (np.vstack([np.ones_like(x_sub), x_sub]).T) @ coef_sub
            base_med_sub = np.nanmedian(base_all_sub)
            y_sub = y_raw_sub / base_all_sub * base_med_sub

        # 4.) curve_fit to get starting values (rp, dt, c0, c1)
        p0_cf = [0.11, 0.0, 1.0, 0.0]
        bounds_cf = ([0.03, -0.02*Period, 0.95, -1.0],
                     [0.25,  0.02*Period, 1.05,  1.0])

        popt_sub, pcov_sub = curve_fit(
                _f_curvefit, x_sub, y_sub,
                p0=p0_cf,
                bounds=bounds_cf,
                sigma=yerr_sub,
                absolute_sigma=True,
                maxfev=10000)

        rp_init, dt_init, c0_init, c1_init = popt_sub

        # 5.) MCMC for this subset (rp, dt, c0, c1, u1, u2, log10_sj)
        ndim, nwalkers = 7, 32
        start = np.array([rp_init, dt_init, c0_init, c1_init, U1_0, U2_0, -3.0])
        p0_walkers = start + 1e-4 * np.random.randn(nwalkers, ndim)

        sampler_sub = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior,
            args=(x_sub, y_sub, yerr_sub, Period, A_RS, INC, u1_0, u2_0, sig_u)
        )

        # Burn-in
        state_sub = sampler_sub.run_mcmc(p0_walkers, 1500, progress=False)
        sampler_sub.reset()

        # Production
        sampler_sub.run_mcmc(state_sub, 3000, progress=False)

        chain_sub = sampler_sub.get_chain(flat=True)
        labels_sub = ["rp", "dt [d]", "c0", "c1", "u1", "u2", "log10_sj"]

        med_sub = np.median(chain_sub, axis=0)
        lo_sub  = np.percentile(chain_sub, 16, axis=0)
        hi_sub  = np.percentile(chain_sub, 84, axis=0)

        # 6.) Compute model and residuals for plotting
        rp_m, dt_m, c0_m, c1_m, u1_m, u2_m, _ = med_sub
        y_med_sub = batman_flux(
            rp_m, dt_m, c0_m, c1_m,
            x_sub, Period, A_RS, INC, u1_m, u2_m
        )
        resid_sub = y_sub - y_med_sub

        # Store results
        subset_mcmc_results.append({
            "rank": rank,
            "comps": js,
            "oot_rms": oot_rms,
            "chi2_red": chi2_red,
            "x": x_sub,
            "y": y_sub,
            "yerr": yerr_sub,
            "y_med": y_med_sub,
            "resid": resid_sub,
            "chain": chain_sub,
            "med": med_sub,
            "lo":     lo_sub,
            "hi":     hi_sub,
            "labels": labels_sub,
        })

        # 7.) Plot data + model + residuals for this subset
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(6, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [5, 1]}
        )

        ax1.errorbar(x_sub, y_sub, yerr=yerr_sub,
                     fmt='.', ecolor='gray', elinewidth=1, capsize=2,
                     label="Data")
        ax1.plot(x_sub, y_med_sub, 'r-', lw=2, label="Median model")
        ax1.set_ylabel("Relative flux")
        ax1.set_title("MCMC transit fit")
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        ax2.axhline(0.0, color='k', lw=1)
        ax2.errorbar(x_sub, resid_sub, yerr=yerr_sub,
                     fmt='.', ecolor='gray', elinewidth=1, capsize=2)
        ax2.set_xlabel("Orbital phase")
        ax2.set_ylabel("Residuals")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(outdir / f"lightcurve_phase_mcmc_subset{rank}_{cube_path.stem}.png",
                    dpi=150)
        plt.close()
        
        
    # -- Corner plot for BEST comparison-star subset --
    if subset_mcmc_results:
        best = min(subset_mcmc_results, key=lambda d: d["oot_rms"])
    
        chain_best  = best["chain"]
        labels_best = best["labels"]   # ["rp","dt [d]","c0","c1","u1","u2","log10_sj"]
    
        # Optional: choose subset of parameters for readability
        use_idx = [0, 1, 2, 3]  # rp, dt, c0, c1 (or keep all 7)
        chain_plot  = chain_best[:, use_idx]
        labels_plot = [labels_best[i] for i in use_idx]
    
        corner.corner(
            chain_plot,
            labels=labels_plot,
            show_titles=True,
            title_fmt=".4f",
            quantiles=[0.16, 0.5, 0.84],
        )
    
        rank = best["rank"]
        oot  = best["oot_rms"]
        plt.suptitle(f"Best subset corner (rank={rank}, OOT RMS={oot:.3e})", fontsize=11)
        plt.tight_layout()
        plt.savefig(outdir / f"mcmc_corner_best_subset_rank{rank}_{cube_path.stem}.png", dpi=150)
        plt.close()
    else:
        print("⚠️ No subset MCMC results available; skipping best-subset corner plot.")


    # - EXTRA PLOTS -
    
    # Residuals vs airmass
    mask = (
    np.isfinite(resid) &
    np.isfinite(air) &
    (air > 0)
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    ax1.scatter(air[mask], resid[mask], s=10, alpha=0.7)
    ax1.axhline(0.0, color='k', lw=1, linestyle='--')
    ax1.set_xlabel("Airmass", fontsize=13)
    ax1.set_ylabel("Residuals", fontsize=13)
    ax1.set_title("Residuals vs airmass", fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=11)
        
    # Residuals vs FWHM
    fwhm_list_arr = np.array(fwhm_list)
    ax2.scatter(fwhm_list_arr, resid, s=10, alpha=0.7)
    ax2.axhline(0.0, color='k', lw=1, linestyle='--')
    ax2.set_xlabel("FWHM (pixels)", fontsize=13)
    ax2.set_ylabel("Residuals", fontsize=13)
    ax2.set_title("Residuals vs FWHM", fontsize=14)
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=11)
        
    plt.tight_layout()
    plt.savefig(outdir / f"analysis_resid_airmass_fwhm_{cube_path.stem}.png", dpi=200)
    plt.close()
        
    def binned_rms(resid, bin_sizes, min_nbins=10):
        """
        Compute RMS of binned residuals (std of bin means), only for bin sizes
        that yield at least `min_nbins` bins.
        """
        resid = resid[np.isfinite(resid)]
        n = len(resid)
    
        out_bins = []
        out_rms  = []
        out_nbins = []
    
        for bs in bin_sizes:
            if bs <= 0 or bs >= n:
                continue
    
            nbins = n // bs
            if nbins < min_nbins:
                continue
    
            trimmed = resid[:nbins * bs].reshape(nbins, bs)
            bin_means = np.nanmean(trimmed, axis=1)
    
            # standard choice: scatter of binned means about their mean
            rms = np.nanstd(bin_means, ddof=1)
    
            out_bins.append(bs)
            out_rms.append(rms)
            out_nbins.append(nbins)
    
        return np.array(out_bins), np.array(out_rms), np.array(out_nbins)
    
    
    # Choose candidate bin sizes (log-ish), but let the function cull large ones
    candidate_bins = np.array([1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32])
    
    resid_use = resid[np.isfinite(resid)]
    bin_sizes, rms_vals, nbins_each = binned_rms(resid_use, candidate_bins, min_nbins=10)
    
    plt.figure(figsize=(6,4))
    plt.loglog(bin_sizes, rms_vals, 'o-', label="Data")
    
    # White-noise expectation anchored at the unbinned RMS (standard)
    plt.loglog(bin_sizes, rms_vals[0] / np.sqrt(bin_sizes), 'k--',
               label=r"White noise $\propto 1/\sqrt{N}$")
    
    plt.xlabel("Bin size (number of points)", fontsize=13)
    plt.ylabel("RMS of binned residuals", fontsize=13)
    plt.title("RMS vs bin size (noise test)", fontsize=15)
    plt.grid(alpha=0.3, which='both')
    plt.legend(fontsize=11)
    plt.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(outdir / f"analysis_rms_bins_{cube_path.stem}.png", dpi=200)
    plt.close()


    # --- Beta factor from binned residual RMS (Pont+ style) ---
    rms1 = float(rms_vals[0])  # unbinned RMS (bin=1)
    beta_vals = rms_vals / (rms1 / np.sqrt(bin_sizes))
    
    beta_median_measured = float(np.nanmedian(beta_vals[1:])) if len(beta_vals) > 1 else float("nan")
    beta_max_measured    = float(np.nanmax(beta_vals[1:]))    if len(beta_vals) > 1 else float("nan")

    beta_used = float(beta)  # this is your config beta (default 1.0)
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n--- Noise inflation (beta factor) ---\n")
        f.write(f"beta_used (config): {beta_used:.3f}\n")
        f.write(f"beta_measured_median (excl N=1): {beta_median_measured:.3f}\n")
        f.write(f"beta_measured_max (excl N=1): {beta_max_measured:.3f}\n")

    # - Save best-fit system parameters to a text file (for pRT etc.) -
    summary_path = outdir / f"system_fit_summary_{cube_path.stem}.txt"
    
    with open(summary_path, "w") as f:
        f.write("# BATMAN + MCMC system-fit summary\n")
        f.write(f"# Cube: {cube_path}\n")
        f.write(f"# Band: {band}\n\n")

        # - Main fit (using full ensemble) -
        f.write("=== Main ensemble fit (all comparison stars used in build_ensemble) ===\n")
        f.write(f"oot_rms_main = {oot_rms:.5f}\n\n")
    
        f.write("Parameters (median +/-1sigma):\n")
        for lab, m, l, h in zip(labels, med, lo, hi):
            plus  = h - m
            minus = m - l
            f.write(f"  {lab:10s} = {m:.8f}  +{plus:.8f}  -{minus:.8f}\n")
    
        # Convenience lines for pRT
        rp_main     = med[0]
        rp_main_lo  = med[0] - lo[0]
        rp_main_hi  = hi[0]  - med[0]
        depth_main  = 100.0 * rp_main**2
    
        f.write("\nDerived (main fit):\n")
        f.write(f"  depth_main_percent = {depth_main:.6f}\n")
        f.write(f"  rp_main            = {rp_main:.8f}  +{rp_main_hi:.8f}  -{rp_main_lo:.8f}\n")
        f.write(f"  dt_main_minutes    = {med[1]*24*60:.4f}\n\n")
    
        # -- Best comparison-star subset --
        if subset_mcmc_results:
            best_subset = min(subset_mcmc_results, key=lambda d: d["oot_rms"])
    
            rank_b   = best_subset["rank"]
            chi2_b   = best_subset["chi2_red"]
            comps_b  = best_subset["comps"]
            med_b    = best_subset["med"]
            lo_b     = best_subset["lo"]
            hi_b     = best_subset["hi"]
            labels_b = best_subset["labels"]
    
            f.write("=== Best comparison-star subset (selected by OOT RMS; parameters from MCMC) ===\n")
            f.write(f"oot_rms_score      = {best_subset['oot_rms']:.6e}\n")
            f.write(f"rank              = {rank_b}\n")
            f.write(f"chi2_red_subset   = {chi2_b:.5f}\n")
            f.write(f"comps_indices     = {comps_b}\n\n")
    
            f.write("Parameters (median +/-1sigma):\n")
            for lab, m, lo_i, hi_i in zip(labels_b, med_b, lo_b, hi_b):
                plus  = hi_i - m
                minus = m - lo_i
                f.write(f"  {lab:10s} = {m:.8f}  +{plus:.8f}  -{minus:.8f}\n")
    
            rp_b     = med_b[0]
            rp_b_lo  = med_b[0] - lo_b[0]
            rp_b_hi  = hi_b[0]  - med_b[0]
            depth_b  = 100.0 * rp_b**2
    
            f.write("\nDerived (best subset):\n")
            f.write(f"  depth_subset_percent = {depth_b:.6f}\n")
            f.write(f"  rp_subset            = {rp_b:.8f}  +{rp_b_hi:.8f}  -{rp_b_lo:.8f}\n")
            f.write(f"  dt_subset_minutes    = {med_b[1]*24*60:.4f}\n")
        else:
            f.write("No subset MCMC results available (subset_mcmc_results empty).\n")
    
    print(f"📄 Saved system-parameter summary to: {summary_path}")

    # - Grid of comparison-star normalised fluxes -
    
    # Use time relative to midpoint (for nicer axis values)
    t_mid = np.nanmedian(t_bjd)
    time_rel = (t_bjd - t_mid) * 24  # hours
    
    n_frames, n_comps = comp_array.shape
    n_show = min(12, n_comps)
    kept_idx = np.arange(n_show)   # first 12 comps
    
    # Setup grid
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), sharex=True)
    
    for plot_i, comp_idx in enumerate(kept_idx):
        ax = axes.flat[plot_i]
        flux = comp_array[:, comp_idx]
    
        # Normalise
        norm_flux = flux / np.nanmedian(flux)
    
        # Plot as line
        ax.plot(time_rel, norm_flux, lw=0.7, color="tab:blue")
    
        # Formatting
        ax.set_title(f"Comp {comp_idx+1}", fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(alpha=0.3, lw=0.3)
    
    # Hide unused panels
    for k in range(len(kept_idx), nrows * ncols):
        axes.flat[k].axis("off")
    
    # Label edges
    for ax in axes[-1, :]:
        ax.set_xlabel("Time - mid [hours]", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalised flux", fontsize=9)
    
    fig.suptitle("Normalised Raw Flux of Comparison Stars", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save high-res
    fig.savefig(outdir / "comparison_star_normalised_flux_grid.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # - Summary -
    print(f"📂 Outputs in: {outdir}")


# --- ENTRY POINT ---
if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
    
    