#!/usr/bin/env python
#"%USERPROFILE%\pyenvs\astro\Scripts\activate"
"""QUB transit photometry pipeline.

Takes a calibrated FITS cube plus a DS9 region file (target first, then comparison stars),
produces a differential light curve, diagnostics, and optional transit model fits.

This script is intended to be called by scripts/run_pipeline.py.
"""


# --- IMPORTS ---
import warnings
import json                                                                                    
import logging
from pathlib import Path                                                                             
import numpy as np                                                                                  
import matplotlib.pyplot as plt                                                                      
import pandas as pd
from astropy.io import fits
from scipy.optimize import curve_fit
from astropy.stats import sigma_clip
from photutils.aperture import aperture_photometry, CircularAnnulus, CircularAperture
from tqdm import tqdm                                                                                
import batman, emcee, corner
from itertools import combinations
import math
import sys as _sys
from qub_pipeline.utils import (
    deep_update,
    load_config,
    setup_logging,
)



# --- WARNINGS ---
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)      
warnings.filterwarnings("ignore", message="partition.*MaskedArray", category=UserWarning)            
warnings.filterwarnings("ignore", message="The fit may not have converged", category=UserWarning)    
warnings.filterwarnings("ignore", message="Warning: 'partition' will ignore the 'mask' of the MaskedArray.", category=UserWarning)
warnings.filterwarnings("ignore", message="Format strings passed to MaskedConstant are ignored", category=FutureWarning)
warnings.filterwarnings("ignore", message="This figure includes Axes that are not", category=UserWarning)    


# --- CONFIG (defaults) ---

# Default configuration values (intentionally minimal). Provide a config.json for real runs.
DEFAULTS: dict = {}

_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
def log_print(*args) -> None:
    """Drop-in replacement for log_print() that logs to qub_photometry."""
    logging.getLogger("qub_photometry").info(" ".join(str(a) for a in args))

    
    
def _segments_from_legacy(*, start_frame: int, reg_path: Path, flip_frame: int | None, postflip_reg_path: Path | None):
    """Build segments from legacy flip-frame style inputs."""
    if flip_frame is None or postflip_reg_path is None:
        return [{"start": int(start_frame), "end": None, "region_path": str(reg_path)}]
    return [
        {"start": int(start_frame), "end": int(flip_frame), "region_path": str(reg_path)},
        {"start": int(flip_frame), "end": None, "region_path": str(postflip_reg_path)},
    ]

def _parse_segments_from_cfg(cfg):
    if not isinstance(cfg, dict):
        return None

    # allow either photometry.segments or segmentation.segments
    phot = cfg.get("photometry", {}) if isinstance(cfg.get("photometry", {}), dict) else {}
    segc = cfg.get("segmentation", {}) if isinstance(cfg.get("segmentation", {}), dict) else {}

    segs = phot.get("segments") or segc.get("segments")
    if not segs:
        return None
    if not isinstance(segs, list):
        raise ValueError("segments must be a list of dicts")

    # minimal validation
    out = []
    for s in segs:
        if not isinstance(s, dict):
            continue
        out.append({
            "start": int(s.get("start", 0)),
            "end": s.get("end", None),
            "region_path": s.get("region_path") or s.get("regions") or s.get("reg_path"),
        })
    return out if out else None

def resolve_segments(
    cfg: dict | None,
    *,
    reg_path_default: Path,
    postflip_default: Path | None,
    start_frame: int,
    legacy_flip_frame: int | None = None,
    legacy_postflip_reg_path: Path | None = None,
):
    """Resolve segmentation regions.

    Priority:
      1) Explicit segments in config (photometry.segments or segmentation.segments).
      2) Legacy flip-frame style segmentation (config segmentation.flip_frame + paths.postflip_reg_path
         or CLI legacy_* overrides).
      3) Single segment using reg_path_default.
    """
    if cfg:
        segs = _parse_segments_from_cfg(cfg)
        if segs:
            return segs

        seg_cfg = cfg.get("segmentation", {}) if isinstance(cfg.get("segmentation", {}), dict) else {}
        paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
        flip_frame = legacy_flip_frame if legacy_flip_frame is not None else seg_cfg.get("flip_frame")
        postflip = legacy_postflip_reg_path
        if postflip is None:
            pf = paths.get("postflip_reg_path") or paths.get("postflip_regions") or paths.get("postflip_reg")
            if pf:
                postflip = Path(pf).expanduser().resolve()
        if postflip is None and postflip_default is not None:
            postflip = Path(postflip_default).expanduser().resolve()

        return _segments_from_legacy(
            start_frame=int(start_frame),
            reg_path=Path(reg_path_default).expanduser().resolve(),
            flip_frame=None if flip_frame is None else int(flip_frame),
            postflip_reg_path=postflip,
        )

    # No cfg: pure legacy/single segment
    return _segments_from_legacy(
        start_frame=int(start_frame),
        reg_path=Path(reg_path_default).expanduser().resolve(),
        flip_frame=None if legacy_flip_frame is None else int(legacy_flip_frame),
        postflip_reg_path=legacy_postflip_reg_path or postflip_default,
    )

def _segment_index_for_frame(i: int, seg_infos):
    for s in seg_infos:
        st = s["start"]
        en = s["end"]
        if i < st:
            continue
        if en is None:
            return s["idx"]
        if st <= i < en:
            return s["idx"]
    return seg_infos[-1]["idx"]

def load_time_and_airmass(N: int, start_frame: int = 0, *, cube_path: Path | None = None, outdir: Path | None = None, time_priority_cols):
    t  = np.full(N, np.nan, dtype=float)
    am = np.full(N, np.nan, dtype=float)

    def _try_df(df: pd.DataFrame, label: str):
        chosen = None

        # time column
        for col in time_priority_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
                L = min(N - start_frame, len(vals))
                if L <= 0:
                    continue
                v = vals[:L]
                nfin = int(np.isfinite(v).sum())
                min_need = min(10, max(3, int(0.2 * L)))
                if nfin >= min_need:
                    t[start_frame:start_frame+L] = v
                    chosen = col
                    log_print(f"⏱ Using time column '{col}' from {label} (finite {nfin}/{L})")
                    break
                else:
                    log_print(f"⚠️ Time column '{col}' in {label} has only {nfin}/{L} finite values; trying next.")

        # airmass optional
        if "AIRMASS" in df.columns:
            vals = pd.to_numeric(df["AIRMASS"], errors="coerce").to_numpy()
            L = min(N - start_frame, len(vals))
            if L > 0:
                am[start_frame:start_frame+L] = vals[:L]
                nfin = int(np.isfinite(am[start_frame:start_frame+L]).sum())
                log_print(f"🌫 Using AIRMASS from {label} (finite {nfin}/{L})")

        return chosen

    # 1.) Cube_manifest.csv (preferred)
    candidates = []
    if outdir is not None:
        candidates.append(Path(outdir) / "cube_manifest.csv")
    if cube_path is not None:
        candidates.append(Path(cube_path).with_name("cube_manifest.csv"))

    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            if _try_df(df, p.name) is not None:
                return t, am

    # 2.) Header_r.csv fallback
        candidates = []
        if outdir is not None:
            candidates.append(Path(outdir) / "Header_r.csv")
        if cube_path is not None:
            candidates.append(Path(cube_path).with_name("Header_r.csv"))
    
        for p in candidates:
            if p.exists():
                df = pd.read_csv(p)
                if _try_df(df, p.name) is not None:
                    return t, am

    raise RuntimeError(
        "No usable time column found in cube_manifest.csv or Header_r.csv. "
        "Need one of: " + ", ".join(time_priority_cols)
    )


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
                if subtract_one: x -= 1.0; y -= 1.0
                circles.append((x, y, r))
    return circles

def write_ds9_circles(path: Path, circles_xy_r, add_header=True):
    lines = []
    if add_header: lines += ["# Region file format: DS9 version 4.1", "image", "global color=green"]
    for x, y, r in circles_xy_r: lines.append(f"circle({x+1:.2f},{y+1:.2f},{r:.2f})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")      
    
    
# --- TRACKING / FWHM ---
def _circle_bbox(xc, yc, r, nx, ny):
    x1 = int(max(0, np.floor(xc - r))); x2 = int(min(nx, np.ceil(xc + r) + 1))
    y1 = int(max(0, np.floor(yc - r))); y2 = int(min(ny, np.ceil(yc + r) + 1))
    return x1, x2, y1, y2

def _gauss2d_const(coords, A, mux, muy, sx, sy, C):
    x, y = coords
    
    # Curvefit Function
    gx = (x - mux)**2 / (2*sx*sx)
    gy = (y - muy)**2 / (2*sy*sy)                       
    return (C + A*np.exp(-(gx + gy))).ravel()

def gauss_centroid_and_fwhm(image, x0, y0, r, jump_px=None,
                            min_snr=5.0, maxfev_good=4000):
    ny, nx = image.shape
    r_fit = max(7.0, 2.5*r)
    x1, x2, y1, y2 = _circle_bbox(x0, y0, r_fit, nx, ny)
    sub = image[y1:y2, x1:x2]
    yy, xx = np.mgrid[y1:y2, x1:x2]
    msk = (xx - x0)**2 + (yy - y0)**2 <= r_fit*r_fit

    x_data = xx[msk].astype(float)
    y_data = yy[msk].astype(float)
    z_data = sub[msk].astype(float)

    # Background ring
    rin  = r_fit*1.05
    rout = r_fit*1.25
    rmask = ((xx - x0)**2 + (yy - y0)**2 >= rin*rin) & \
            ((xx - x0)**2 + (yy - y0)**2 <= rout*rout)

    if rmask.any():
        ring = sigma_clip(sub[rmask], 5.0, masked=True).filled(np.nan)
        sky_rms = float(np.nanstd(ring))
    else:
        sky_rms = np.nan

    if not np.isfinite(sky_rms) or sky_rms <= 0:
        
        # Fallback to avoid division by zero
        sky_rms = 1.0  

    C0 = float(np.nanmedian(z_data))
    A0 = float(np.nanmax(z_data) - C0)

    # - FAST REJECTION: low SNR or negative peak -
    snr0 = A0 / sky_rms if sky_rms > 0 else 0.0
    if (A0 <= 0) or (snr0 < min_snr):
        
        # Star basically invisible: don't waste time on curve_fit
        return float(x0), float(y0), np.nan, np.nan, sky_rms, False

    # Initial params & bounds
    p0 = [A0, float(x0), float(y0),
          max(1.5, r_fit/3), max(1.5, r_fit/3), C0]

    bounds = ([0,   x0-r_fit, y0-r_fit, 0.5,   0.5,   -np.inf],
              [np.inf, x0+r_fit, y0+r_fit, 2*r_fit, 2*r_fit,  np.inf])

    try:
        A, mux, muy, sx, sy, C = curve_fit(
            _gauss2d_const,
            (x_data, y_data),
            z_data,
            p0=p0,
            bounds=bounds,
            maxfev=maxfev_good,
        )[0]
    except RuntimeError:
        return float(x0), float(y0), np.nan, np.nan, sky_rms, False

    # Optional: reject huge jumps (lost lock)
    if jump_px is not None:
        if (mux - x0)**2 + (muy - y0)**2 > jump_px**2:
            return float(x0), float(y0), np.nan, np.nan, sky_rms, False

    fwhm = 2.355 * float(np.sqrt(sx*sx + sy*sy))
    peak = float(A + C)
    return float(mux), float(muy), fwhm, peak, sky_rms, True

    

# --- PHOTOMETRY ---
def photometry_batch(image, xys, fwhm, *, k_ap=2.5, k_in=3.5, k_out=6.0,
                     method="exact", require_full_annulus=True):
    """
    Same maths as photometry(), but for many stars at once.
    xys: list/array of (x, y) positions, length N.

    Returns:
      flux (N,), sky_pp (N,), ok (N,), r_ap, r_in, r_out
    """
    ny, nx = image.shape

    # Match your single-star geometry exactly
    r_ap  = max(2.5, k_ap * fwhm)
    r_in  = max(r_ap * 1.1, k_in * fwhm)
    r_out = max(r_in + 1.0, k_out * fwhm)

    xys = np.asarray(xys, dtype=float)
    xs = xys[:, 0]
    ys = xys[:, 1]

    # Validity mask
    ok = np.isfinite(xs) & np.isfinite(ys)

    # Optional: require the full annulus to be on-frame
    if require_full_annulus:
        ok &= (xs - r_out >= 0) & (xs + r_out < nx) & (ys - r_out >= 0) & (ys + r_out < ny)

    N = len(xys)
    flux = np.full(N, np.nan, dtype=float)
    sky_pp = np.full(N, np.nan, dtype=float)

    if not np.any(ok):
        return flux, sky_pp, ok, r_ap, r_in, r_out

    positions = np.column_stack([xs[ok], ys[ok]])

    ap = CircularAperture(positions, r=r_ap)
    ann = CircularAnnulus(positions, r_in=r_in, r_out=r_out)

    tab = aperture_photometry(image, [ap, ann], method=method)
    ap_sum  = np.asarray(tab["aperture_sum_0"], dtype=float)

    # Sky subtraction
    sky_pp_ok  = np.full_like(ap_sum, np.nan, dtype=float)
    sky_std_ok = np.full_like(ap_sum, np.nan, dtype=float)
    n_sky_ok   = np.zeros_like(ap_sum, dtype=int)
    
    # Loop sky stats (correct > fast)
    for i in range(len(ap_sum)):
        sky_pp_ok[i], sky_std_ok[i], n_sky_ok[i] = sky_stats_sigma_clip(
            image, CircularAnnulus(positions[i], r_in=r_in, r_out=r_out),
            sigma=3.0, maxiters=5
        )
    
    flux_ok = ap_sum - (sky_pp_ok * ap.area)
    
    # Allocate outputs at top:
    sky_std = np.full(N, np.nan, dtype=float)
    n_sky   = np.zeros(N, dtype=int)
    
    # Fill:
    flux[ok]     = flux_ok
    sky_pp[ok]   = sky_pp_ok
    sky_std[ok]  = sky_std_ok
    n_sky[ok]    = n_sky_ok
    
    return flux, sky_pp, sky_std, n_sky, ok, r_ap, r_in, r_out

def sky_stats_sigma_clip(image, annulus, *, sigma=3.0, maxiters=5):
    
    # Annulus.to_mask gives you a cutout + mask
    m = annulus.to_mask(method="center")
    cut = m.cutout(image)
    if cut is None:
        return np.nan, np.nan, 0
    
    data = cut[m.data.astype(bool)]
    data = data[np.isfinite(data)]
    if data.size < 20:
        return np.nan, np.nan, int(data.size)

    clipped = sigma_clip(data, sigma=sigma, maxiters=maxiters, masked=True)
    good = clipped.data[~clipped.mask]
    if good.size < 10:
        return np.nan, np.nan, int(good.size)

    sky_pp = np.median(good)
    sky_std = 1.4826 * np.median(np.abs(good - sky_pp))  # robust sigma (MAD)
    return float(sky_pp), float(sky_std), int(good.size)



# --- COMBINE COMPARISON STARS ---
def oot_mask_from_geometry(phase_centered, *, period, a_rs, inc_deg, k0=0.0, safety=1.15):
    """Return boolean mask selecting out-of-transit (OOT) points.

    Uses a simple circular-orbit transit-duration geometry with k=k0 (default 0)
    to estimate the half-duration in orbital phase. Points within +/- half-duration
    (inflated by `safety`) are treated as in-transit and excluded.

    This is intentionally conservative; it is used only for comparison-star weighting
    and baseline estimation, not for the final transit inference.
    """
    ph = np.asarray(phase_centered, dtype=float)
    oot = np.isfinite(ph)

    try:
        a = float(a_rs)
        inc = np.deg2rad(float(inc_deg))
        # Impact parameter b = a cos i (Rp/Rs neglected if k0=0)
        b = a * np.cos(inc)

        # Guardrails
        if not (np.isfinite(a) and np.isfinite(b) and a > 1.0 and 0.0 <= b < (1.0 + k0)):
            
            # Fall back: everything finite is OOT
            return oot

        # Eqn for total duration T14 for circular orbit (Winn 2010 style form)
        # T14/P = (1/pi) * arcsin( (1/a) * sqrt((1+k)^2 - b^2) / sin i )
        numer = np.sqrt(max((1.0 + k0)**2 - b**2, 0.0))
        denom = a * np.sin(inc)
        arg = numer / denom if denom > 0 else np.nan
        if not np.isfinite(arg) or arg <= 0:
            return oot

        arg = min(1.0, max(0.0, arg))
        T14_over_P = (1.0 / np.pi) * np.arcsin(arg)
        half_phase = 0.5 * T14_over_P * safety

        oot &= (np.abs(ph) > half_phase)
        return oot
    except Exception:
        return oot


def build_ensemble(comp_array, *, min_valid_comps=8):
    """Legacy unweighted ensemble (kept for compatibility)."""
    
    # Per-star scaling constants
    M_j = np.nanmedian(comp_array, axis=0)    

    # Relative flux per comp
    Rel_flux = comp_array / M_j[None, :]    

    # Per-frame median across comps
    ens = np.nanmedian(Rel_flux, axis=1)                   
    return ens, Rel_flux, M_j


def build_ensemble_weighted(comp_array, oot_mask, *, min_valid_comps=8, sigma_floor=1e-6):
    """Build a weighted ensemble using OOT-only statistics.

    Steps:
      1) OOT-normalise each comparison star by its OOT median.
      2) Compute each star's OOT scatter (robust MAD->sigma).
      3) Weight stars as w_j ∝ 1/sigma_j^2 and form weighted mean per frame.

    Returns
    -------
    ens : (N,) weighted ensemble (dimensionless, ~1)
    Rel_flux : (N,C) relative flux per comp (normalised by OOT median)
    M_j : (C,) OOT medians
    w : (C_kept,) weights for kept comps (sum to 1)
    kept : list[int] indices of comps used
    sigma_j : (C_kept,) OOT scatter estimates
    """
    comp = np.asarray(comp_array, dtype=float)
    oot = np.asarray(oot_mask, dtype=bool)
    if comp.ndim != 2:
        raise ValueError("comp_array must be 2D (Nframes, Ncomps)")
    n, c = comp.shape

    # OOT median per comp
    M_j = np.full(c, np.nan, dtype=float)
    for j in range(c):
        v = comp[:, j]
        ok = oot & np.isfinite(v)
        if np.sum(ok) >= 5:
            M_j[j] = np.nanmedian(v[ok])

    Rel_flux = comp / M_j[None, :]

    # OOT scatter per comp
    def _robust_sigma(a):
        a = np.asarray(a, dtype=float)
        med = np.nanmedian(a)
        mad = np.nanmedian(np.abs(a - med))
        sig = 1.4826 * mad
        return sig

    sigma = np.full(c, np.nan, dtype=float)
    for j in range(c):
        v = Rel_flux[:, j]
        ok = oot & np.isfinite(v)
        if np.sum(ok) >= 8:
            sigma[j] = _robust_sigma(v[ok])

    # Keep stars with finite sigma and median
    kept = [j for j in range(c) if np.isfinite(M_j[j]) and np.isfinite(sigma[j]) and sigma[j] > 0]
    if len(kept) < min_valid_comps:
        
        # Fall back to legacy ensemble if too few
        ens, Rel_flux2, M2 = build_ensemble(comp, min_valid_comps=min_valid_comps)
        return ens, Rel_flux2, M2, None, list(range(c)), None

    sigma_kept = np.array([sigma[j] for j in kept], dtype=float)
    w_raw = 1.0 / np.maximum(sigma_kept, sigma_floor)**2
    w = w_raw / np.nansum(w_raw)

    # Weighted mean per frame
    rel_kept = Rel_flux[:, kept]  # (N, C_kept)
    ens = np.nansum(rel_kept * w[None, :], axis=1)

    return ens, Rel_flux, M_j, w, kept, sigma_kept


# --- BATMAN FUNCTIONS 5D ---
def batman_flux_fixedLD(rp, dt_days, c0, c1, x_phase,
                        *, period, a_rs, inc_deg, u1, u2):
    """
    Baseline + BATMAN transit evaluated on x_phase (phase array).
    Limb darkening fixed to (u1,u2).
    """
    t = np.asarray(x_phase, dtype=float) * float(period)

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

    # Baseline in phase (centered)
    return c0 + c1 * np.asarray(x_phase, dtype=float) + (mod - 1.0)

def log_prior_5d(theta, *, Period):
    rp, dt, c0, c1, log10_sj = theta

    # Hard bounds (tight for stability)
    if not (0.01 < rp < 0.5):
        return -np.inf
    if not (-0.05 * Period < dt < 0.05 * Period):
        return -np.inf
    if not (0.8 < c0 < 1.2):
        return -np.inf
    if not (-2.0 < c1 < 2.0):
        return -np.inf
    if not (-6.0 < log10_sj < -1.0):
        return -np.inf

    # Soft priors (Gaussian)
    lp  = -0.5 * ((c0 - 1.0) / 0.05) ** 2
    lp += -0.5 * (c1 / 0.5) ** 2
    return lp

def log_likelihood_5d(theta, x_phase, y_flux, y_sigma,
                      *, Period, A_RS, INC, U1_0, U2_0):
    rp, dt, c0, c1, log10_sj = theta
    sj = 10.0 ** log10_sj

    mu = batman_flux_fixedLD(
        rp, dt, c0, c1, x_phase,
        period=Period, a_rs=A_RS, inc_deg=INC, u1=U1_0, u2=U2_0
    )

    var = np.asarray(y_sigma, dtype=float) ** 2 + sj ** 2
    r = np.asarray(y_flux, dtype=float) - mu
    return -0.5 * np.sum(r * r / var + np.log(2.0 * np.pi * var))

def log_posterior_5d(theta, x_phase, y_flux, y_sigma,
                     *, Period, A_RS, INC, U1_0, U2_0):
    lp = log_prior_5d(theta, Period=Period)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_5d(theta, x_phase, y_flux, y_sigma,
                                  Period=Period, A_RS=A_RS, INC=INC, U1_0=U1_0, U2_0=U2_0)



# --- MAIN ---
def binned_rms(resid, nbins_list):
    rms_vals = []
    n = len(resid)
    for nb in nbins_list:
        if nb >= n:
            rms_vals.append(np.nan)
            continue
        m = n // nb   # number of bins
        r = resid[:m*nb].reshape(m, nb)
        bin_means = np.nanmean(r, axis=1)
        rms_vals.append(np.sqrt(np.nanmean(bin_means**2)))
    return np.array(rms_vals)

def _load_segment_regions(segments, *, base_dir: Path):
    """
    Load DS9 regions for each segment and return a list of segment info dicts:
      {
        "idx": int,
        "start": int,
        "end": int|None,
        "region_path": str,
        "target_pos": (x,y),
        "target_r": r,
        "comps_pos": [(x,y), ...],
        "comps_r": [r, ...],
      }
    Assumes DS9 region file has circles, first circle = target, rest = comps.
    """
    seg_infos = []
    for idx, s in enumerate(segments):
        start = int(s.get("start", 0))
        end = s.get("end", None)
        end = None if end in (None, "None") else int(end)

        rp = Path(s.get("region_path", "")).expanduser()
        if not rp.is_absolute():
            rp = (base_dir / rp).resolve()
        if not rp.exists():
            raise SystemExit(f"❌ Segment region file not found: {rp}")

        circles = load_ds9_circles(rp, subtract_one=True)
        if len(circles) < 2:
            raise SystemExit(f"❌ Need target+≥1 comp circles in {rp} (found {len(circles)})")

        (tx, ty, tr) = circles[0]
        comps = circles[1:]
        comp_pos = [(x, y) for (x, y, r) in comps]
        comp_r   = [r for (x, y, r) in comps]

        seg_infos.append({
            "idx": idx,
            "start": start,
            "end": end,
            "region_path": str(rp),
            "target_pos": (tx, ty),
            "target_r": tr,
            "comps_pos": comp_pos,
            "comps_r": comp_r,
        })
    return seg_infos

def run_photometry(cube_path: Path, reg_path: Path, outdir: Path, *, cfg: dict | None = None):
    phot = cfg["photometry"]
    inst = cfg["instrument"]
    ephem = cfg["ephemeris"]
    geom = cfg["transit_geom"]
    ens_cfg = cfg["ensemble"]
    time_cfg = cfg["time"]
    ld = cfg["limb_darkening"]
    err_cfg = cfg.get("error_model", {}) if isinstance(cfg.get("error_model", {}), dict) else {}
    beta = float(err_cfg.get("beta", 1.0))
    if (not np.isfinite(beta)) or beta <= 0:
        beta = 1.0

    # Locals used everywhere below
    k_ap = float(phot["k_ap"]); k_in = float(phot["k_in"]); k_out = float(phot["k_out"])
    MIN_SNR_REF  = float(phot["min_snr_ref"])
    MIN_SNR_TARG = float(phot["min_snr_targ"])
    MIN_SNR_COMP = float(phot["min_snr_comp"])
    MIN_VALID_COMPS = int(phot.get("min_valid_comps", phot.get("min_comps", 8)))
    FWHM_MIN = float(phot["fwhm_min"]); FWHM_MAX = float(phot["fwhm_max"])
    WRITE_TRACKED_REG_EVERY = int(phot["write_tracked_reg_every"])
    Period = float(ephem["period_days"])
    T0_BJD = float(ephem["t0_bjd"])
    A_RS = float(geom["a_rs"])
    INC  = float(geom["inc_deg"])
    U1_0 = float(ld["u1_0"]); U2_0 = float(ld["u2_0"])
    time_priority_cols = list(time_cfg["priority_cols"])
    gain = float(inst["gain_e_per_adu"])
    
    # --- Load read noise from reduction products.json ---
    read_noise_e = float("nan")
    
    cube_path = Path(cube_path).expanduser().resolve()
    outdir    = Path(outdir).expanduser().resolve()
    
    # The cube is in the reduction folder, so that's the most reliable anchor
    red_dir = cube_path.parent
    
    candidate_products = [
        red_dir / "products.json",                 # ✅ correct: alongside cube in reduction/
        outdir / "products.json",                  # optional/fallback
        outdir.parent / "products.json",           # legacy fallback
    ]
    
    products_path = next((p for p in candidate_products if p.exists()), None)
    if products_path is None:
        raise FileNotFoundError(
            "products.json not found. Looked in:\n  - " +
            "\n  - ".join(str(p) for p in candidate_products)
        )
    
    with open(products_path, "r", encoding="utf-8") as f:
        prod = json.load(f)
    
    read_noise_e = float(prod.get("read_noise_e", float("nan")))
    if not np.isfinite(read_noise_e):
        raise RuntimeError(f"read_noise_e missing/NaN in {products_path}")
    log_print(f"✅ Loaded read_noise_e from {products_path}: {read_noise_e:.3f} e-")
    run_summary = {}
    
    
    # -- Post flip only hard crop --
    
    # Pull config
    phot = (cfg.get("photometry", {}) if isinstance(cfg, dict) else {})
    start_frame_cfg = int(phot.get("start_frame", 0))   # set to 144 in JSON to start post-flip
    
    # Resolve paths early
    cube_path = Path(cube_path).expanduser().resolve()
    reg_path  = Path(reg_path).expanduser().resolve()
    outdir    = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load cube (optionally memmapped) then crop
    io_cfg = cfg.get('io', {}) if isinstance(cfg.get('io', {}), dict) else {}
    use_memmap = bool(io_cfg.get('memmap', True))

    hdul = None
    if use_memmap:
        hdul = fits.open(cube_path, memmap=True)
        import atexit
        atexit.register(lambda: hdul.close())
        cube_full = hdul[0].data
    else:
        cube_full = fits.getdata(cube_path)
    N_full = int(cube_full.shape[0])
    
    if not (0 <= start_frame_cfg < N_full):
        raise SystemExit(f"❌ photometry.start_frame={start_frame_cfg} out of range for cube with N={N_full}")
    
    # This becomes working cube: "frame 0" now corresponds to original frame start_frame_cfg
    cube = cube_full[start_frame_cfg:]
    N, ny, nx = cube.shape
    log_print(f"N_full={N_full}, start_frame={start_frame_cfg} -> using N={N} cropped frames; cube shape={cube.shape}")
    
    # Load time/airmass, then crop to match cube
    t_full, airmass_full = load_time_and_airmass(
        N_full, start_frame=0, cube_path=cube_path, outdir=outdir,  time_priority_cols= time_priority_cols
    )
    t_bjd   = t_full[start_frame_cfg:]
    airmass = airmass_full[start_frame_cfg:]
    
    # Resolve segments & pick region file of ORIGINAL start frame
    segments = resolve_segments(
        cfg,
        reg_path_default=reg_path,
        postflip_default=None,
        start_frame=0
    )
    seg_infos = _load_segment_regions(segments, base_dir=reg_path.parent)
    
    # Find which segment covers the ORIGINAL frame index = start_frame_cfg
    seg0_idx = _segment_index_for_frame(int(start_frame_cfg), seg_infos)
    s = seg_infos[seg0_idx]
    log_print(f"Using regions from segment {seg0_idx}: {s['region_path']} (covers original frame {start_frame_cfg})")
    
    # Freeze region definitions for entire run
    target_pos = s["target_pos"]
    target_r   = s["target_r"]
    comps_pos  = list(s["comps_pos"])
    comps_r    = list(s["comps_r"])
    if len(comps_pos) < 1:
        raise SystemExit(f"❌ Segment {seg0_idx} has no comparison stars.")
    
    ref_pos = comps_pos[0]
    ref_r   = comps_r[0]
    frames = np.arange(0, N, dtype=int)
    
    # Sanity debug
    log_print("Time finite:", int(np.isfinite(t_bjd).sum()), "/", len(t_bjd),
          " | Airmass finite:", int(np.isfinite(airmass).sum()), "/", len(airmass))



    # -- Photometry loop --
    
    # - Config -
    fwhm_prev = 3.0  
    Jmax = len(comps_pos)
    all_target, all_comps, target_noise, target_flux_e, fwhm_list = [], [], [], [], []  
    
    # Initialise containers
    frames   = np.arange(0, N, dtype=int)
    n_frames = len(frames)
    
    # Per-frame scalars
    all_target    = np.full(n_frames, np.nan, dtype=float)
    target_noise  = np.full(n_frames, np.nan, dtype=float)
    target_flux_e = np.full(n_frames, np.nan, dtype=float)
    fwhm_list     = np.full(n_frames, np.nan, dtype=float)
    
    # Per-frame, per-comparison-star fluxes
    all_comps = np.full((n_frames, Jmax), np.nan, dtype=float)
    flag_ref_ok     = np.zeros(n_frames, dtype=bool)
    flag_targ_ok    = np.zeros(n_frames, dtype=bool)
    flag_fwhm_ok    = np.zeros(n_frames, dtype=bool)
    flag_mincomps   = np.zeros(n_frames, dtype=bool)
    n_valid_comps   = np.zeros(n_frames, dtype=int)
    
    # Store fitted centroids
    ref_x = np.full(n_frames, np.nan)
    ref_y = np.full(n_frames, np.nan)
    targ_x = np.full(n_frames, np.nan)
    targ_y = np.full(n_frames, np.nan)
    
    # For aperture analysis
    targ_xy = np.full((n_frames, 2), np.nan)
    comps_xy = np.full((n_frames, Jmax, 2), np.nan)
    sky_pp_list = np.full(n_frames, np.nan, dtype=float)

    # Photometry Loop >>>>>>
    for k, i in enumerate(tqdm(frames, desc="Aperture photometry", ncols=90)):
        img = cube[i]
        ny, nx = img.shape
    
        # Jump limits
        Jref  = max(0.6 * ref_r,    6.0)
        Jtarg = max(0.6 * target_r, 6.0)
    
        # 1.) Fit reference star
        rx, ry, rfwhm, rpk, rrms, rok = gauss_centroid_and_fwhm(
            img, ref_pos[0], ref_pos[1], ref_r,
            jump_px=Jref, min_snr=MIN_SNR_REF
        )
    
        ref_x[k], ref_y[k] = rx, ry  
        fwhm_list[k] = rfwhm
    
        # If reference failed, do NOT apply drift this frame.
        if (not rok) or (not np.isfinite(rx)) or (not np.isfinite(ry)):
            
            # Leave NaNs (frame rejected) and continue
            continue
    
        flag_ref_ok[k] = True
    
        # 2.) Drift update (from ref)
        dx = rx - ref_pos[0]
        dy = ry - ref_pos[1]
        ref_pos = (rx, ry)
    
        # 3.) FWHM update + sanity
        if np.isfinite(rfwhm) and (FWHM_MIN <= rfwhm <= FWHM_MAX):
            fwhm_prev = float(rfwhm)
            flag_fwhm_ok[k] = True
        fwhm = fwhm_prev
    
        # 4.) Apply drift to target + comps
        target_pred = (target_pos[0] + dx, target_pos[1] + dy)
        comps_pred  = [(px + dx, py + dy) for (px, py) in comps_pos]
    
        # 5.) Fit target around prediction
        tx, ty, tfw, tpk, trms, tok = gauss_centroid_and_fwhm(
            img, target_pred[0], target_pred[1], target_r,
            jump_px=Jtarg, min_snr=MIN_SNR_TARG
        )
    
        targ_x[k], targ_y[k] = tx, ty
        
        # Aperture analysis arrays
        targ_xy[k] = (tx, ty)
        comps_xy[k, :, 0] = [p[0] for p in comps_pos]
        comps_xy[k, :, 1] = [p[1] for p in comps_pos]

        if (not tok) or (not np.isfinite(tfw)):
            
            # Target failed -> reject frame
            continue
    
        flag_targ_ok[k] = True
    
        # Use fitted target centroid as the updated position for next frame
        target_pos = (tx, ty)
    
        # 6.) Comparison star centroiding
        new_comps_pos = []
        
        for j, (px, py) in enumerate(comps_pred):
            Jcomp = max(0.6 * comps_r[j], 6.0)
        
            cx, cy, cfw, cpk, crms, cok = gauss_centroid_and_fwhm(
                img, px, py, comps_r[j],
                jump_px=Jcomp, min_snr=MIN_SNR_COMP
            )
        
            if cok and np.isfinite(cfw):
                new_comps_pos.append((cx, cy))
            else:
                # keep predicted position if fit failed (so drift model doesn’t collapse)
                new_comps_pos.append((px, py))
        
        comps_pos = new_comps_pos

        # 7.) Batched photometry: target + comps
        xys_all = [(tx, ty)] + list(comps_pos)
        
        flux_all, sky_pp_all, sky_std_all, n_sky_all, ok_all, r_ap, r_in, r_out = photometry_batch(
            img, xys_all, fwhm,
            k_ap=k_ap, k_in=k_in, k_out=k_out,
            method="exact",
            require_full_annulus=True
        )
        sky_std = sky_std_all[0]
        t_flux = flux_all[0]
        
        # Sky per pixel for target annulus
        sky_pp_list[k] = sky_pp_all[0]   
        comp_fluxes = flux_all[1:]

        # QC: require enough valid comparison stars
        n_ok = int(np.sum(np.isfinite(comp_fluxes) & (comp_fluxes > 0)))
        n_valid_comps[k] = n_ok
        if n_ok < MIN_VALID_COMPS:
            continue
        flag_mincomps[k] = True
        if (not np.isfinite(t_flux)) or (t_flux <= 0):
            continue
    
        # 8.) Write reg overlays occasionally
        if (i % WRITE_TRACKED_REG_EVERY) == 0:
            regs_now = [(tx, ty, target_r)] + [(px, py, r) for (px, py), r in zip(comps_pos, comps_r)]
            write_ds9_circles(outdir / f"tracked_f{i:03d}.reg", regs_now)
    
        # 9.) Target noise (counts -> electrons -> noise)
        t_flux_e  = t_flux * gain
        #sky_pp_e  = sky_pp * gain
        n_ap  = np.pi * (r_ap**2)
        n_sky = np.pi * (r_out**2 - r_in**2)
        
        # Variance per pixel in sky region (Poisson sky + read noise)
        var_sky_pp = (sky_std * gain)**2 + read_noise_e**2

        
        # Total variance:
        # - source Poisson
        # - sky+read noise inside aperture
        # - uncertainty from estimating the sky level and subtracting it
        var_e = max(t_flux_e, 0.0) + n_ap * var_sky_pp + (n_ap**2) * var_sky_pp / max(n_sky, 1.0)
        noise_e = np.sqrt(var_e)
        
        target_noise[k]  = noise_e
        target_flux_e[k] = t_flux_e
    
        # 10.) Store fluxes (COUNTS for target and comps)
        all_target[k] = t_flux
        all_comps[k, :len(comp_fluxes)] = comp_fluxes
        
    # Convert to arrays (in counts)
    target_array = np.array(all_target, dtype=float)
    comp_array   = np.array(all_comps,  dtype=float)
    
    # Frame-level "good" mask based on explicit flags
    good_frame = (
        flag_ref_ok &
        flag_targ_ok &
        flag_mincomps
    )

    log_print("\nQC summary:")
    log_print(f"  Total frames processed: {n_frames}")
    log_print(f"  Ref OK:       {np.sum(flag_ref_ok)}")
    log_print(f"  Target OK:    {np.sum(flag_targ_ok)}")
    log_print(f"  Min comps OK: {np.sum(flag_mincomps)}  (threshold={MIN_VALID_COMPS})")
    log_print(f"  Final good:   {np.sum(good_frame)}  ({100*np.mean(good_frame):.1f}%)")
    
    

    Tref = T0_BJD
    phase = ((t_bjd - Tref) / Period) % 1.0
    phase_centered = ((phase + 0.5) % 1.0) - 0.5  # -0.5..+0.5 with mid-transit at 0


    # -- OOT mask for ensemble weights --
    oot_mode = ens_cfg.get("oot_mode", "post_window")
    min_oot_points = int(ens_cfg.get("min_oot_points", 20))
    
    if oot_mode == "post_window":
        pmin = float(ens_cfg.get("oot_phase_min", 0.023))
        pmax = float(ens_cfg.get("oot_phase_max", 0.05))
        OOT_MASK = (
            good_frame &
            np.isfinite(phase_centered) &
            (phase_centered >= pmin) & (phase_centered <= pmax)
        )
    else:
        # Fallback options
        OOT_MASK = oot_mask_from_geometry(
            phase_centered, period=Period, a_rs=A_RS, inc_deg=INC, k0=0.0, safety=1.20
        ) & good_frame
    
    oot_count = int(np.sum(OOT_MASK))
    log_print(f"✅ OOT points for ensemble weights: {oot_count}")
    
    if oot_count < min_oot_points:
        log_print(f"⚠️ Only {oot_count} OOT points in post-window; falling back to |phase| > 0.05 (still QC-gated).")
        OOT_MASK = good_frame & np.isfinite(phase_centered) & (np.abs(phase_centered) > 0.05)
    

    # Build ensemble from comps using OOT-weighted statistics
    ens, Rel_flux, M_j, w_comp, kept_comps, sigma_comps = build_ensemble_weighted(
        comp_array, OOT_MASK, min_valid_comps=MIN_VALID_COMPS
    )
    # ens is dimensionless (~1). Differential light curve in counts-space:
    
    # Differential light curve in counts-space (counts / counts)
    lc_abs = target_array / ens
    
    # Error model
    t_flux_e = target_array * gain
    frac_targ = np.full_like(lc_abs, np.nan, dtype=float)
    ok_t = np.isfinite(target_noise) & np.isfinite(t_flux_e) & (t_flux_e > 0)
    frac_targ[ok_t] = target_noise[ok_t] / t_flux_e[ok_t]
    use_full_ens = bool(ens_cfg.get("use_full_ensemble_error", False))
    inflate = float(ens_cfg.get("ensemble_inflate", 1.20))
    
    # Ensemble fractional error:
    # Use OOT-weight + per-star OOT scatter
    frac_ens = np.zeros_like(lc_abs, dtype=float)
    
    if use_full_ens:
        if (w_comp is not None) and (sigma_comps is not None):
            
            # Sigma_comps are per-star scatter of Rel_flux (dimensionless)
            frac_ens_const = float(np.sqrt(np.sum((w_comp**2) * (sigma_comps**2))))
            frac_ens[:] = frac_ens_const
        else:
            
            # Fallback if weighting failed
            frac_ens[:] = 0.0
    else:
        
        # Ignore ensemble error explicitly
        frac_ens[:] = 0.0
    
    lc_err_abs = np.abs(lc_abs) * np.sqrt(frac_targ**2 + frac_ens**2)
    lc_err_abs *= inflate
    lc_err_abs = np.abs(lc_err_abs)
    lc_err_abs *= beta
    
    # Normalisation
    scale = np.nanmedian(lc_abs[good_frame])
    lc_abs /= scale
    lc_err_abs /= scale
    

    # - Phase plot prep -
    # (t_bjd and phase_centered already computed above for OOT-weighted ensemble)
    
    # If AIRMASS is missing/NaN for most frames, do not let it wipe out the dataset.
    airmass_ok = np.isfinite(airmass)
    n_good = int(np.sum(good_frame))
    n_air  = int(np.sum(airmass_ok & good_frame))
    if n_good > 0 and n_air < max(5, int(0.2 * n_good)):
        log_print(f"⚠️ AIRMASS is finite for only {n_air}/{n_good} good frames. Proceeding without AIRMASS guard.")
        airmass_ok = np.ones_like(good_frame, dtype=bool)
    
    base_mask = (
        good_frame &
        np.isfinite(phase_centered) &
        np.isfinite(lc_abs) &
        np.isfinite(lc_err_abs) &
        (lc_err_abs > 0)
    )
    
    log_print("DEBUG counts:",
          "good_frame", int(np.sum(good_frame)),
          "phase", int(np.sum(np.isfinite(phase_centered))),
          "lc", int(np.sum(np.isfinite(lc_abs))),
          "err", int(np.sum(np.isfinite(lc_err_abs))),
          "err>0", int(np.sum(lc_err_abs > 0)),
          "airmass_ok", int(np.sum(airmass_ok)),
          "base_mask", int(np.sum(base_mask)))

    base_mask &= (lc_abs > 0.9) & (lc_abs < 1.1)
    
    m = good_frame & np.isfinite(lc_abs)
    log_print("lc_abs min/max on good frames:", np.nanmin(lc_abs[m]), np.nanmax(lc_abs[m]))
    
    # Restrict to a phase window for plotting/fitting
    log_print("After flux sanity:", int(np.sum(base_mask)))
    PHASE_MIN, PHASE_MAX = -0.01, 0.05
    base_mask &= (phase_centered >= PHASE_MIN) & (phase_centered <= PHASE_MAX)
    log_print("After phase window:", int(np.sum(base_mask)))
    
    N_total = len(phase_centered)
    N_qc    = int(np.sum(good_frame))
    N_base  = int(np.sum(base_mask))
    
    log_print("\nFrame diagnostic (pre-clipping):")
    log_print(f"  Total frames:              {N_total}")
    log_print(f"  Passed QC flags:           {N_qc}  ({100*N_qc/N_total:.1f}%)")
    log_print(f"  Passed QC+finite+guards:   {N_base}  ({100*N_base/N_total:.1f}%)")
    
    # Sort by phase AFTER masking
    order = np.argsort(phase_centered[base_mask])
    
    x    = phase_centered[base_mask][order]
    y    = lc_abs[base_mask][order]
    yerr = np.abs(lc_err_abs[base_mask][order])  # force non-negative
    
    
    if x.size < 10 or y.size < 10:
        raise SystemExit(f"❌ Not enough valid points for curve_fit: x={x.size}, y={y.size}")

    # 3.) Clipping (MAD-b,ased) to remove remaining cloud/guiding outliers
    # MAD is much more stable than std when you have clouds.
    def _mad(a):
        med = np.nanmedian(a)
        return np.nanmedian(np.abs(a - med))
    
    y_med = np.nanmedian(y)
    mad   = _mad(y)
    
    # Convert MAD to sigma-equivalent (~1.4826*MAD)
    sigma_rob = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nan
    
    # If sigma_rob is zero (very clean) or NaN, skip clipping
    if np.isfinite(sigma_rob) and sigma_rob > 0:
        clip_k = 5.0  # ~5σ robust clip
        good_lc = (
            np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) &
            (yerr > 0) &
            (y > y_med - clip_k * sigma_rob) &
            (y < y_med + clip_k * sigma_rob)
        )
    else:
        good_lc = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    
    N_post = int(np.sum(good_lc))
    log_print(f"  After robust clip:         {N_post}  ({100*N_post/N_total:.1f}%)")
    log_print(f"  Clipped additional points: {len(y) - N_post}")
    
    # Apply clip mask
    x    = x[good_lc]
    y    = y[good_lc]
    yerr = yerr[good_lc]
    
    # Now build a NEW mask on the sliced arrays
    yerr_cap = np.nanpercentile(yerr, 99)
    good2 = np.isfinite(yerr) & (yerr > 0) & (yerr <= yerr_cap)
    
    x    = x[good2]
    y    = y[good2]
    yerr = yerr[good2]

    idx_base = np.where(base_mask)[0][order]
    idx_used = idx_base[good_lc]
    idx_used = idx_used[good2]

    # 4.) Plot
    plt.figure(figsize=(10, 4))
    plt.errorbar(x, y, yerr=yerr, fmt='.', elinewidth=1, capsize=2, alpha=0.6, label="Data")
    plt.xlabel("Orbital phase", fontsize=14)
    plt.ylabel("Relative flux", fontsize=14)
    plt.ylim(0.95, 1.04)
    plt.title("Phase-folded light curve", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"lightcurve_phase_{cube_path.stem}.png", dpi=150)
    plt.close()
    
    
 
    
    # --- Batman Plot ---
    
    # Curve_fit start (4 params; jitter handled by MCMC)
    def _f_curvefit_4(x_phase, rp, dt_days, c0, c1):
        return batman_flux_fixedLD(
            rp, dt_days, c0, c1, x_phase,
            period=Period, a_rs=A_RS, inc_deg=INC, u1=U1_0, u2=U2_0
        )
    
    p0_cf = [0.12, 0.0, 1.0, 0.0]
    bounds_cf = (
        [0.03, -0.02 * Period, 0.90, -2.0],
        [0.25,  0.02 * Period, 1.10,  2.0],
    )
    
    popt_cf, pcov_cf = curve_fit(
        _f_curvefit_4, x, y,
        p0=p0_cf, bounds=bounds_cf,
        sigma=yerr, absolute_sigma=True,
        maxfev=40000
    )
    
    rp_init, dt_init, c0_init, c1_init = popt_cf
    log_print(f"[curve_fit 5D-init] rp={rp_init:.5f}, depth≈{100*rp_init**2:.2f}%  "
          f"dt={dt_init*24*60:.2f} min  c0={c0_init:.5f} c1={c1_init:.5f}")
    
    # MCMC setup
    ndim, nwalkers = 5, 40
    start = np.array([rp_init, dt_init, c0_init, c1_init, -3.0])  # log10_sj init
    p0_walkers = start + 1e-4 * np.random.randn(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim,
        lambda th, xx, yy, ss: log_posterior_5d(
            th, xx, yy, ss, Period=Period, A_RS=A_RS, INC=INC, U1_0=U1_0, U2_0=U2_0
        ),
        args=(x, y, yerr)
    )
    
    state = sampler.run_mcmc(p0_walkers, 2000, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, 4000, progress=False)
    
    chain = sampler.get_chain(flat=True)
    #labels = ["rp", "dt [d]", "c0", "c1", "log10_sj"]
    
    med = np.median(chain, axis=0)
    #lo  = np.percentile(chain, 16, axis=0)
    #hi  = np.percentile(chain, 84, axis=0)
    
    rp_m, dt_m, c0_m, c1_m, log10_sj_m = med
    
    y_med = batman_flux_fixedLD(
        rp_m, dt_m, c0_m, c1_m, x,
        period=Period, a_rs=A_RS, inc_deg=INC, u1=U1_0, u2=U2_0
    )
    
    resid = y - y_med


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    
    ax1.errorbar(x, y, yerr=yerr, fmt='.', ecolor='gray', elinewidth=1, capsize=2, label="Data")
    ax1.plot(x, y_med, 'r-', lw=2, label="Median BATMAN model")
    ax1.set_ylabel("Relative flux", fontsize=14)
    ax1.set_title("Phase-folded light curve with MCMC transit fit", fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=12)
    
    ax2.errorbar(x, resid, yerr=yerr, fmt='.', ecolor='gray', elinewidth=1, capsize=2)
    ax2.axhline(0.0, color='k', lw=1, linestyle='--')
    ax2.set_xlabel("Orbital phase", fontsize=14)
    ax2.set_ylabel("Residuals", fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / f"lightcurve_phase_mcmc_{cube_path.stem}.png", dpi=200)
    plt.close()

    
    
    
    # --- Comparison Star Diagnostic: refit per subset, score by BIC ---
    
    # Work in the same point-set as the final fit:
    # idx_used aligns with x, y, yerr, seg, y_med

    N_limit = 150
    comps_used  = comp_array[idx_used, :]
    target_used = target_array[idx_used].astype(float)
    n_used, n_comps = comps_used.shape
    min_comps = 14
    max_comps = min(15, n_comps)   
    top_k = 10
    rp0 = float(rp_init)
    dt0 = float(dt_init)
    
    # Evaluation points: seg==1 + within phase window + finite yerr
    eval_mask = (
        (x >= PHASE_MIN) & (x <= PHASE_MAX) &
        np.isfinite(yerr) & (yerr > 0)
    )
    
    idx_eval = np.where(eval_mask)[0]
 
    use_idx = idx_eval[:min(N_limit, idx_eval.size)]
    if len(use_idx) < 10:
        log_print(f"⚠️ Subset ranking has only {len(use_idx)} points; results may be unstable.")

    x_eval    = x[use_idx]
    yerr_eval = yerr[use_idx]
    comps_eval  = comps_used[use_idx, :]
    target_eval = target_used[use_idx]
    
    log_print(f"\nSubset diagnostic on {n_comps} comparison stars "
          f"({len(use_idx)} points used: seg==1 and phase in [{PHASE_MIN},{PHASE_MAX}]).")
    
    run_summary["subset_fits"] = []

    def subset_oot_rms(js):
        js = list(js)
    
        comp_sub = comps_eval[:, js]
    
        use_mask = (
            np.isfinite(x_eval) &
            np.isfinite(target_eval) &
            np.isfinite(yerr_eval) & (yerr_eval > 0) &
            np.isfinite(np.nanmedian(comp_sub, axis=1))
        )
    
        ens_sub, Rel_sub, M_sub, w_sub, kept_sub, sig_sub = build_ensemble_weighted(
            comp_sub, use_mask, min_valid_comps=8
        )
    
        ok = (
            np.isfinite(ens_sub) & (ens_sub > 0) &
            np.isfinite(target_eval) &
            np.isfinite(x_eval) &
            np.isfinite(yerr_eval) & (yerr_eval > 0)
        )
        if np.sum(ok) < 10:
            return None
    
        lc = (target_eval[ok] / ens_sub[ok])
        med_lc = np.nanmedian(lc)
        if not np.isfinite(med_lc) or med_lc <= 0:
            return None
    
        ydat = lc / med_lc
        xdat = x_eval[ok]
        sig  = yerr_eval[ok]
    
        # Fit rp, dt, c0, c1 (same as before; LD fixed; no jitter)
        p0 = [rp0, dt0, 1.0, 0.0]
        bounds = (
            [0.03, -0.02 * Period, 0.90, -2.0],
            [0.25,  0.02 * Period, 1.10,  2.0],
        )
    
        def model_4(xx, rp, dt, c0, c1):
            return batman_flux_fixedLD(
                rp, dt, c0, c1, xx,
                period=Period, a_rs=A_RS, inc_deg=INC, u1=U1_0, u2=U2_0
            )
    
        try:
            popt, _ = curve_fit(
                model_4, xdat, ydat,
                p0=p0, bounds=bounds,
                sigma=sig, absolute_sigma=True,
                maxfev=20000
            )
        except Exception:
            return None
    
        # --- detrend and compute OOT RMS ---
    
        # Remove the fitted baseline by dividing it out
        # Baseline is (c0 + c1*x); transit part is the BATMAN dip.
        # model_4 already outputs baseline*transit, so:
        baseline = popt[2] + popt[3] * xdat
        if not np.all(np.isfinite(baseline)) or np.nanmin(baseline) <= 0:
            return None
    
        y_detr = ydat / baseline
    
        oot = (np.abs(xdat) > 0.015) & np.isfinite(y_detr)
        if np.sum(oot) < 10:
            return None
    
        score = float(np.nanstd(y_detr[oot], ddof=1))
        return score


    scores = []
    if n_comps < min_comps:
        log_print(f"⚠️ Only {n_comps} comparison stars available; subset ranking skipped (min_comps={min_comps}).")
        best_js = tuple(range(n_comps))
        top_list = [(np.nan, list(best_js), len(best_js))]
    
        # Still write summary and continue pipeline
        run_summary["subset_ranking"] = {
            "metric": "oot_rms",
            "n_eval_points": int(len(use_idx)),
            "best_oot_rms": None,
            "best_subset": [int(i) for i in best_js],
            "top_list": [{"oot_rms": float(b), "size": int(sz), "comps": [int(i) for i in js]}
                         for (b, js, sz) in top_list],
        }
    else:
        for k in range(min_comps, max_comps+1):
            n_k = math.comb(n_comps, k)
            log_print(f"  Theoretical subsets for k={k}: C({n_comps},{k}) = {n_k}")
            for js in combinations(range(n_comps), k):
                score = subset_oot_rms(js)
                if score is None:
                    continue
                scores.append((score, tuple(js)))
        
        if not scores:
            log_print("⚠️ No valid subsets survived ranking. Falling back to ALL comps.")
            best_js = tuple(range(n_comps))
            top_list = [(np.nan, list(best_js), len(best_js))]
        else:
            scores.sort(key=lambda t: t[0])   # lower OOT RMS = better
            best_score, best_js = scores[0]
            top_k = 3
            top_list = [(sc, list(js), len(js)) for (sc, js) in scores[:top_k]]
        
        log_print(f"\nBest OOT RMS: {best_score:.6f}  subset={list(best_js)}")
        for i,(sc,js,size) in enumerate(top_list,1):
            log_print(f"  #{i}: OOT_RMS={sc:.6f}, size={size}, comps={js}")
        
        run_summary["subset_ranking"] = {
            "metric": "OOT_RMS_DETRENDED",
            "n_eval_points": int(len(use_idx)),
            "oot_phase_cut": 0.015,
            "best_score": None if not np.isfinite(best_score) else float(best_score),
            "best_subset": [int(i) for i in best_js],
            "top_list": [{"oot_rms": float(sc), "size": int(sz), "comps": [int(i) for i in js]}
                         for (sc, js, sz) in top_list],
        }


    # Convert once
    target_noise_arr  = np.array(target_noise, dtype=float)
    target_flux_e_arr = np.array(target_flux_e, dtype=float)
    
    subset_mcmc_results = []
    
    for rank, (oot_rms, js, size) in enumerate(top_list, start=1):
        log_print(f"\n=== MCMC rerun for subset #{rank}: OOT_RMS={oot_rms:.6f}, size={size}, comps={js} ===")
    
        # 1.) Ensemble from this subset, evaluated on ALL frames (counts)
        comp_sub_full = comp_array[:, js]
        ens_subset_full, Rel_full, M_full, w_full, kept_full, sig_full = build_ensemble_weighted(
            comp_sub_full, OOT_MASK, min_valid_comps=MIN_VALID_COMPS
        )
    
        # 2.) Build point-set using the SAME base_mask logic as main run
        # Start from the explicit QC flags:
        base_sub = (
            good_frame &
            np.isfinite(phase_centered) &
            np.isfinite(target_array) &
            np.isfinite(ens_subset_full) & (ens_subset_full > 0)
        )
    
        # Apply same optional sanity guard if you kept it above:
        lc_abs_full = target_array / ens_subset_full
        base_sub &= np.isfinite(lc_abs_full)
    
        tn_all  = target_noise_arr
        tfe_all = target_flux_e_arr
        frac_t_all = np.full_like(tfe_all, np.nan, dtype=float)
        ok_all = np.isfinite(tn_all) & np.isfinite(tfe_all) & (tfe_all > 0)
        frac_t_all[ok_all] = tn_all[ok_all] / tfe_all[ok_all]
        
        base_sub &= np.isfinite(frac_t_all) & (frac_t_all < 0.02)   # tune 0.02

    
        if np.sum(base_sub) < 25:
            log_print("  Too few usable points after QC for this subset, skipping.")
            continue
    
        # Phase sort
        idx_sub_base = np.where(base_sub)[0]
        order = np.argsort(phase_centered[idx_sub_base])
        idx_sub_base = idx_sub_base[order]
        
        x_sub   = phase_centered[idx_sub_base]
        
        # 3.) Normalise LC (subset ensemble per frame)
        subset_flux_pts = comp_array[idx_sub_base][:, js]              # (Npts, k)
        ens_subset_pts  = np.nanmedian(subset_flux_pts, axis=1)        # (Npts,)
        
        ok_ens = np.isfinite(ens_subset_pts) & (ens_subset_pts > 0) & np.isfinite(target_array[idx_sub_base])
        if np.sum(ok_ens) < 25:
            log_print("  ⚠️ Too few valid ensemble points; skipping subset")
            continue
        
        lc_abs_sub = target_array[idx_sub_base] / ens_subset_pts
        scale_sub  = np.nanmedian(lc_abs_sub[ok_ens])
        if not np.isfinite(scale_sub) or scale_sub <= 0:
            log_print("  ⚠️ Bad normalisation scale; skipping subset")
            continue
        
        y_sub = lc_abs_sub / scale_sub

        # 4.) Errors: target fractional + ensemble fractional scatter
        cfg_err = cfg.get("error_model", {})
        ensemble_mode = str(cfg_err.get("ensemble_mode", "simple")).lower()  # "simple" or "full"
        simple_scale  = float(cfg_err.get("simple_ensemble_scale", 1.2))
        mask_mode     = str(cfg_err.get("ensemble_mask", "in_transit")).lower()  # "in_transit"|"oot"|"all"
        
        # Target fractional noise
        tn  = target_noise_arr[idx_sub_base]
        tfe = target_flux_e_arr[idx_sub_base]
        frac_t = np.full_like(y_sub, np.nan, dtype=float)
        
        ok_t = np.isfinite(tn) & np.isfinite(tfe) & (tfe > 0)
        frac_t[ok_t] = tn[ok_t] / tfe[ok_t]
        
        # choose which points to use to ESTIMATE ensemble scatter
        # Must be same length as y_sub / subset_flux_pts
        ph = x_sub  # <-- replace with your phase array for these points
        
        if mask_mode == "all":
            use = np.isfinite(ph)
        elif mask_mode == "oot":
            # your existing post-window mask (example)
            use = (ph >= pmin) & (ph <= pmax)
        elif mask_mode == "in_transit":
            Tdur_phase = float(cfg_err.get("tdur_phase", 0.015))  # <-- set this in config
            use = np.abs(ph) <= (0.5 * Tdur_phase)
        else:
            raise ValueError(f"Unknown ensemble_mask: {mask_mode}")
        
        # Always require finite comps on used points
        use &= np.isfinite(np.nanmedian(subset_flux_pts, axis=1))
        
        # Ensemble fractional noise
        if ensemble_mode == "simple":
            
            # No per-frame ensemble term; just inflate final errors
            frac_e = np.zeros_like(y_sub, dtype=float)
        
        else:
            # FULL mode: compute per-frame scatter across stars, but ONLY on `use` points
            # Compute frac_e for ALL points; for points not in `use`, fill by median of use (stable)
            frac_e = np.full_like(y_sub, np.nan, dtype=float)
        
            subset_flux_use = subset_flux_pts[use, :]  # (Nuse, k)
        
            Mj = np.nanmedian(subset_flux_use, axis=0)          # (k,)
            good_star = np.isfinite(Mj) & (Mj > 0)
        
            if np.sum(good_star) < 3:
                log_print("  ⚠️ Too few finite comp stars for noise model; using simple mode instead")
                frac_e[:] = 0.0
                ensemble_mode = "simple"
            else:
                rel = subset_flux_use[:, good_star] / Mj[None, good_star]  # (Nuse, k_eff)
        
                med_rel = np.nanmedian(rel, axis=1)
                mad = np.nanmedian(np.abs(rel - med_rel[:, None]), axis=1)
                sigma_star = 1.4826 * mad
        
                n_eff = np.sum(np.isfinite(rel), axis=1)
                n_eff = np.maximum(n_eff, 1)
        
                frac_e_use = sigma_star / np.sqrt(n_eff)
        
                # Put it back into full-length frac_e:
                frac_e[use] = frac_e_use
        
                # For points outside `use`, fill with median of the used estimate (so errors exist everywhere)
                fill = np.nanmedian(frac_e_use) if np.isfinite(np.nanmedian(frac_e_use)) else 0.0
                frac_e[~use] = fill
        
        # - Combine -
        yerr_sub = np.abs(y_sub) * np.sqrt(frac_t**2 + frac_e**2)
        yerr_sub *= beta

        # Apply the simple inflate factor (your 1.2) when in simple mode
        if ensemble_mode == "simple":
            yerr_sub *= simple_scale
        
        yerr_sub = np.abs(yerr_sub)
        
        # - Diagnostics -
        if rank == 1:
            def stats(name, a):
                a = np.asarray(a)
                good = np.isfinite(a)
                if good.sum() == 0:
                    log_print(f"[{name}] all non-finite")
                    return
                q = np.nanpercentile(a[good], [0, 1, 5, 50, 95, 99, 100])
                log_print(f"[{name}] finite={good.sum()}/{a.size}  min={q[0]:.3e} p1={q[1]:.3e} p5={q[2]:.3e} "
                      f"med={q[3]:.3e} p95={q[4]:.3e} p99={q[5]:.3e} max={q[6]:.3e}")
        
            stats("tfe (target_flux_e)", tfe)
            stats("tn (target_noise_e)", tn)
            stats("frac_t", frac_t)
            stats("frac_e", frac_e)
            stats("yerr_sub", yerr_sub)
        
        # - Clipping of outlier flux values (same as main) -
        def _mad(a):
            med = np.nanmedian(a)
            return np.nanmedian(np.abs(a - med))
        
        y_med0 = np.nanmedian(y_sub)
        mad0 = _mad(y_sub)
        
        # This converts MAD into an estimate of σ (standard deviation) if the underlying noise were Gaussian.
        sig0 = 1.4826 * mad0 if np.isfinite(mad0) and mad0 > 0 else np.nan
        
        if np.isfinite(sig0) and sig0 > 0:
            clip_k = 5.0
            keep = (
                np.isfinite(x_sub) & np.isfinite(y_sub) & np.isfinite(yerr_sub) &
                (yerr_sub > 0) &
                (y_sub > y_med0 - clip_k*sig0) &
                (y_sub < y_med0 + clip_k*sig0)
            )
        else:
            keep = np.isfinite(x_sub) & np.isfinite(y_sub) & np.isfinite(yerr_sub) & (yerr_sub > 0)
        
        x_sub = x_sub[keep]
        y_sub = y_sub[keep]
        yerr_sub = yerr_sub[keep]
        idx_used_sub = idx_sub_base[keep]   # global frame indices aligned to x_sub/y_sub/resid_sub
        
        if len(x_sub) < 25:
            log_print("  Too few points after clipping, skipping.")
            continue
            
        # 5.) curve_fit for starts (4 params; LD fixed; no segment term)
        def _f_curvefit_sub(x_phase, rp, dt_days, c0, c1):
            return batman_flux_fixedLD(
                rp, dt_days, c0, c1, x_phase,
                period=Period, a_rs=A_RS, inc_deg=INC, u1=U1_0, u2=U2_0
            )
        
        p0_cf = [0.12, 0.0, 1.0, 0.0]
        bounds_cf = (
            [0.03, -0.02 * Period, 0.90, -2.0],
            [0.25,  0.02 * Period, 1.10,  2.0],
        )
        
        popt_sub, pcov_sub = curve_fit(
            _f_curvefit_sub, x_sub, y_sub,
            p0=p0_cf, bounds=bounds_cf,
            sigma=yerr_sub, absolute_sigma=True,
            maxfev=40000
        )
        
        rp_init, dt_init, c0_init, c1_init = popt_sub
        log_print(f"  [curve_fit subset #{rank}] rp={rp_init:.5f}, depth≈{100*rp_init**2:.2f}%  "
              f"dt={dt_init*24*60:.2f} min  c0={c0_init:.5f} c1={c1_init:.5f}")
        
        # 6.) MCMC (5D): rp, dt, c0, c1, log10_sj
        ndim, nwalkers = 5, 40
        start = np.array([rp_init, dt_init, c0_init, c1_init, -3.0])
        p0_walkers = start + 1e-4 * np.random.randn(nwalkers, ndim)
        
        def _logpost_wrap(theta, xx, yy, ss):


            return log_posterior_5d(
                theta, xx, yy, ss,
                Period=Period, A_RS=A_RS, INC=INC, U1_0=U1_0, U2_0=U2_0
            )
        
        sampler_sub = emcee.EnsembleSampler(
            nwalkers, ndim, _logpost_wrap, args=(x_sub, y_sub, yerr_sub)
        )
        
        state_sub = sampler_sub.run_mcmc(p0_walkers, 2000, progress=False)
        sampler_sub.reset()
        sampler_sub.run_mcmc(state_sub, 4000, progress=False)
        
        chain_sub = sampler_sub.get_chain(flat=True)
        labels_sub = ["rp", "dt [d]", "c0", "c1", "log10_sj"]
        
        med_sub = np.median(chain_sub, axis=0)
        lo_sub  = np.percentile(chain_sub, 16, axis=0)
        hi_sub  = np.percentile(chain_sub, 84, axis=0)
        
        rp_m, dt_m, c0_m, c1_m, log10_sj_m = med_sub
        
        y_med_sub = batman_flux_fixedLD(
            rp_m, dt_m, c0_m, c1_m, x_sub,
            period=Period, a_rs=A_RS, inc_deg=INC, u1=U1_0, u2=U2_0
        )
        resid_sub = y_sub - y_med_sub
        
        comps_int = [int(j) for j in js]
        subset_mcmc_results.append({
            "rank": rank,
            "comps": comps_int,
            "x": x_sub,
            "y": y_sub,
            "yerr": yerr_sub,
            "y_med": y_med_sub,
            "resid": resid_sub,
            "chain": chain_sub,
            "med": med_sub,
            "labels": labels_sub,
            "idx_used_sub": idx_used_sub,
        })
        
        run_summary["subset_fits"].append({
            "rank": rank,
            "OOT_RMS": float(oot_rms),
            "comps": comps_int,
            "n_used": int(len(x_sub)),
            "params": {lab: {"median": float(m), "p16": float(l), "p84": float(h)}
                       for lab, m, l, h in zip(labels_sub, med_sub, lo_sub, hi_sub)},
            "derived": {
                "depth_frac": float(med_sub[0]**2),
                "depth_percent": float(100*med_sub[0]**2),
                "dt_minutes": float(med_sub[1]*24*60),
                "dseg_ppt": float(med_sub[4]*1e3),
            }
        })
        
        # Plot
        fig = corner.corner(
        chain_sub,
        labels=labels_sub,
        plot_contours=False, 
        plot_density=False,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".5f",
        )
        fig.savefig(outdir / f"mcmc_corner_subset{rank}_{cube_path.stem}.png", dpi=200)
        plt.close(fig)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                                       gridspec_kw={"height_ratios": [5, 1]})
    
        ax1.errorbar(x_sub, y_sub, yerr=yerr_sub, fmt='.', ecolor='gray',
                     elinewidth=1, capsize=2, label="Data")
        ax1.plot(x_sub, y_med_sub, 'r-', lw=2, label="Median seg model")
        #ax1.set_ylim(0.95, 1.05)
        ax1.set_ylabel("Relative flux")
        ax1.set_title(f"MCMC transit fit (subset #{rank}) OOT RMS={oot_rms:.3f}")
        ax1.grid(alpha=0.3)
        ax1.legend()
    
        ax2.axhline(0.0, color='k', lw=1)
        ax2.errorbar(x_sub, resid_sub, yerr=yerr_sub, fmt='.', ecolor='gray',
                     elinewidth=1, capsize=2)
        ax2.set_xlabel("Orbital phase")
        ax2.set_ylabel("Residuals")
        ax2.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.savefig(outdir / f"lightcurve_phase_mcmc_subset{rank}_{cube_path.stem}.png", dpi=150)
        plt.close()
        
        
    # -- EXTRA PLOTS --
    if subset_mcmc_results:
        best = subset_mcmc_results[0]
        resid_best = best["resid"]
        idx_used_best = best.get("idx_used_sub", None)
        if idx_used_best is None:
            log_print("⚠️ No idx_used_sub stored; skipping extra plots.")
        else:
            if airmass is None:
                log_print("⚠️ Skipping residuals vs airmass: airmass is None")
            else:
                am = airmass[idx_used_best]
                mask = np.isfinite(am) & np.isfinite(resid_best) & (am > 0)
    
                log_print("Residuals vs airmass points (best subset):", int(np.sum(mask)))
    
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.scatter(am[mask], resid_best[mask], s=10, alpha=0.7)
                ax1.axhline(0, ls="--")
                ax1.set_xlabel("Airmass")
                ax1.set_ylabel("Residual")
    
                ax2.hist(resid_best[mask], bins=40)
                ax2.set_xlabel("Residual")
                ax2.set_ylabel("Count")
    
                fig.suptitle("Residuals vs Airmass (best subset)")
                fig.tight_layout()
                plt.savefig(outdir / f"residuals_vs_airmass_bestsubset_{cube_path.stem}.png", dpi=200)
                plt.close()
    
            nbins_list = np.array([1, 2, 4, 8, 16, 32])
            mask_valid = np.isfinite(resid_best)
            rms_vals = binned_rms(resid_best[mask_valid], nbins_list)
    
            plt.figure(figsize=(6,4))
            plt.loglog(nbins_list, rms_vals, 'o-', label="Data")
            plt.loglog(nbins_list, rms_vals[0]/np.sqrt(nbins_list), 'k--', label=r"White noise $\propto 1/\sqrt{N}$")
            plt.xlabel("Bin size (number of points)")
            plt.ylabel("RMS of binned residuals")
            plt.title("RMS vs bin size (best subset)")
            plt.grid(alpha=0.3, which='both')
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / f"analysis_rms_bins_bestsubset_{cube_path.stem}.png", dpi=200)
            plt.close()
    else:
        log_print("⚠️ No subset MCMC results available; skipping extra plots.")
        
    # -- BETA --
    def compute_beta_from_residuals(resid, bin_sizes):
        rms1 = np.nanstd(resid, ddof=1)
        rms_vals = []
        beta_vals = []
    
        for N in bin_sizes:
            nb = len(resid) // N
            if nb < 2:
                rms_vals.append(np.nan)
                beta_vals.append(np.nan)
                continue
    
            binned = np.nanmean(resid[:nb*N].reshape(nb, N), axis=1)
            rmsN = np.nanstd(binned, ddof=1)
            rms_vals.append(rmsN)
            beta_vals.append(rmsN / (rms1 / np.sqrt(N)))
    
        return rms1, np.array(rms_vals), np.array(beta_vals)

    rms1, rms_bins, beta_vals = compute_beta_from_residuals(
        resid_final,
        bin_sizes=[1, 2, 3, 4, 6, 8]
    )
    
    run_summary["noise"] = {
        "beta_measured_median": float(np.nanmedian(beta_vals[1:])),
        "beta_measured_max": float(np.nanmax(beta_vals[1:])),
        "bin_sizes": bin_sizes,
        "beta_vals": beta_vals.tolist(),
}
    # -- NEW EXTRA PLOTS --
    def _time_hours_from_mid(t):
        t = np.asarray(t, dtype=float)
        t0 = np.nanmedian(t[np.isfinite(t)])
        return (t - t0) * 24.0
    
    def plot_airmass_vs_time(t_bjd, airmass, outdir, stem="cube"):
        t_hr = _time_hours_from_mid(t_bjd)
        m = np.isfinite(t_hr) & np.isfinite(airmass)
    
        if np.sum(m) < 5:
            log_print("⚠️ Not enough finite points for airmass plot; skipping.")
            return
        
        tt = t_hr[m]
        aa = airmass[m]
        order = np.argsort(tt)
       
        plt.figure(figsize=(7, 3.5))
        plt.plot(tt[order], aa[order], "o-", ms=3)
        plt.xlabel("Time from mid-run [hours]")
        plt.ylabel("Airmass")
        plt.title("Airmass vs time")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        outpath = Path(outdir) / f"diagnostic_airmass_vs_time_{stem}.png"
        plt.savefig(outpath, dpi=200)
        plt.close()
        log_print(f"✅ Saved {outpath.name}")
    
    def plot_fwhm_vs_time(t_bjd, fwhm_pix, outdir, stem="cube"):
        t_hr = _time_hours_from_mid(t_bjd)
        fwhm_pix = np.asarray(fwhm_pix, dtype=float)
        m = np.isfinite(t_hr) & np.isfinite(fwhm_pix)

        fwhm = np.asarray(fwhm_pix, float)
        
        # Basic physical sanity limits
        FWHM_MIN = 1.0
        FWHM_MAX = 30.0
        
        flag_fwhm_ok = np.isfinite(fwhm) & (fwhm > FWHM_MIN) & (fwhm < FWHM_MAX)
    
        m = np.isfinite(t_hr) & flag_fwhm_ok
        tt = t_hr[m]
        ff = fwhm[m]
        order = np.argsort(tt)
        if np.sum(m) < 5:
            log_print("⚠️ Not enough finite points for FWHM plot; skipping.")
            return
    
        plt.figure(figsize=(7, 3.5))
        plt.plot(tt[order], ff[order], "o-")
        plt.xlabel("Time from mid-run [hours]")
        plt.ylabel("FWHM [pixels]")
        plt.title("Seeing (FWHM) vs time")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        outpath = Path(outdir) / f"diagnostic_fwhm_vs_time_{stem}.png"
        plt.savefig(outpath, dpi=200)
        plt.close()
        log_print(f"✅ Saved {outpath.name}")
    
    def plot_sky_vs_time(t_bjd, sky_pp_adu_per_pix, outdir, stem="cube"):
        t_hr = _time_hours_from_mid(t_bjd)
        sky_pp_adu_per_pix = np.asarray(sky_pp_adu_per_pix, dtype=float)
        m = np.isfinite(t_hr) & np.isfinite(sky_pp_adu_per_pix)
    
        if np.sum(m) < 5:
            log_print("⚠️ Not enough finite points for sky plot; skipping.")
            return
    
        plt.figure(figsize=(7, 3.5))
        plt.plot(t_hr[m], sky_pp_adu_per_pix[m], "o-", ms=3)
        plt.xlabel("Time from mid-run [hours]")
        plt.ylabel("Sky background [ADU / pixel]")
        plt.title("Sky background vs time")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        outpath = Path(outdir) / f"diagnostic_sky_vs_time_{stem}.png"
        plt.savefig(outpath, dpi=200)
        plt.close()
        log_print(f"✅ Saved {outpath.name}")
    
        outpath = Path(outdir) / f"mcmc_corner_{stem}.png"
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        log_print(f"✅ Saved {outpath.name}")
        
    stem = Path(cube_path).stem if cube_path else "cube"
    plot_airmass_vs_time(t_bjd, airmass, outdir, stem=stem)
    plot_fwhm_vs_time(t_bjd, fwhm_list, outdir, stem=stem)
    plot_sky_vs_time(t_bjd, sky_pp_list, outdir, stem=stem)
    
    # - TEMP DEBUGGING -
    log_print(np.median(frac_targ))
    log_print(np.median(frac_ens))
    log_print(np.median(lc_err_abs))
    
    # - Write summary - 
    summary_path = outdir / f"fit_summary_{cube_path.stem}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    
    log_print(f"🧾 Wrote summary: {summary_path}")

    # --- Summary ---
    log_print(f"📂 Outputs in: {outdir}")


# --- ENTRYPOINT WRAPPER ---
def main(argv: list[str] | None = None, *, config_path: str | None = None, verbose: bool = False) -> int:
    """
    Unified entrypoint used by:
      - installed CLI (qub-pipeline photometry) calling main(config_path=..., verbose=...)
      - scripts/qub_photometry.py wrapper calling main() with no args
      - direct execution: python scripts/qub_photometry.py --cube ... --regions ... --outdir ...
    """
    import argparse
    import json as _json
    from pathlib import Path as _Path

    default_cfg_path = str(_Path(__file__).with_name("config.json"))

    ap = argparse.ArgumentParser(description="QUB Photometry (INT-style).")
    ap.add_argument("--config", type=str, default=None, help="JSON config file (optional).")
    ap.add_argument("--cube", type=str, default=None, help="Override cube path.")
    ap.add_argument("--regions", type=str, default=None, help="Override DS9 region path.")
    ap.add_argument("--outdir", type=str, default=None, help="Override output directory.")
    ap.add_argument("--verbose", action="store_true", help="Enable debug-level logging.")
    ap.add_argument("--flip-frame", type=int, default=None, help="Optional flip frame override (legacy).")
    ap.add_argument("--postflip-regions", type=str, default=None, help="Optional post-flip DS9 region path (legacy).")

    args = ap.parse_args(argv)

    # Resolve config path priority:
    # 1) explicit --config
    # 2) config_path kw (from installed CLI)
    # 3) config.json beside this module if it exists
    cfg_path = args.config or config_path
    if cfg_path is None and _Path(default_cfg_path).exists():
        cfg_path = default_cfg_path

    cfg_user = load_config(cfg_path) if cfg_path else {}
    cfg = deep_update(_json.loads(_json.dumps(DEFAULTS)), cfg_user or {})

    # ---- Resolve paths ----
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    phot  = cfg.get("photometry", {}) if isinstance(cfg.get("photometry", {}), dict) else {}
    seg   = cfg.get("segmentation", {}) if isinstance(cfg.get("segmentation", {}), dict) else {}

    cube_path_s = args.cube or paths.get("cube_path")
    reg_path_s  = args.regions or phot.get("regions_path") or paths.get("reg_path") or paths.get("regions_path")
    outdir_s    = args.outdir or phot.get("outdir") or paths.get("outdir")

    # If cube not specified, try to read it from reduction/products.json under out_root
    if not cube_path_s:
        out_root = paths.get("out_root")
        run_id = paths.get("run_id")
        target_name = (cfg.get("target", {}) or {}).get("name", "target")
        if out_root:
            base = _Path(str(out_root)).expanduser().resolve()
            cand_reduction_dirs = []
            if run_id:
                cand_reduction_dirs.append(base / str(target_name).strip().replace(" ", "_") / str(run_id) / "reduction")
                cand_reduction_dirs.append(base / str(run_id) / "reduction")
            cand_reduction_dirs.append(base / "reduction")
            for d in cand_reduction_dirs:
                p = d / "products.json"
                if p.exists():
                    try:
                        prod = _json.loads(p.read_text(encoding="utf-8"))
                        cube_path_s = prod.get("cube_fits") or prod.get("cube")
                        if cube_path_s:
                            break
                    except Exception:
                        pass

    if not cube_path_s or not reg_path_s or not outdir_s:
        raise SystemExit(
            "❌ cube_path, reg_path/regions_path, and outdir must be set via config or CLI overrides. "
            "Set photometry.regions_path and photometry.outdir (and optionally paths.cube_path), "
            "or pass --cube/--regions/--outdir."
        )

    cube_path = _Path(cube_path_s).expanduser().resolve()
    reg_path  = _Path(reg_path_s).expanduser().resolve()
    outdir    = _Path(outdir_s).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Logging (prefer passed verbose kw, then CLI flag)
    use_verbose = bool(verbose) or bool(args.verbose)
    setup_logging(outdir / "photometry.log", verbose=use_verbose)
    logging.getLogger("qub_photometry").info("Output directory: %s", outdir)

    # Apply legacy segmentation overrides
    if args.flip_frame is not None:
        seg["flip_frame"] = int(args.flip_frame)
    if args.postflip_regions:
        paths["postflip_reg_path"] = str(_Path(args.postflip_regions).expanduser().resolve())

    cfg["paths"] = paths
    cfg["segmentation"] = seg
    cfg["photometry"] = phot

    run_photometry(cube_path, reg_path, outdir, cfg=cfg)
    return 0


# --- RUN ---
DEFAULT_CONFIG_PATH = str(Path(__file__).with_name("config.json"))  # kept for backward-compat

if __name__ == "__main__":
    raise SystemExit(main())
