[![DOI](https://zenodo.org/badge/1136032536.svg)](https://doi.org/10.5281/zenodo.18279844)

# QUB Transit Photometry Pipeline

This repository contains a small, config-driven pipeline for reducing QUB observatory data into a calibrated FITS cube and then extracting a differential light curve via aperture photometry.

## Quickstart

1. Create/modify a config file (start from `config/example_config.json`).
2. Put your data in the paths referenced by the config (e.g. `science/`, `dark/`, and a DS9 region file).
3. Run:

```bash
python scripts/run_pipeline.py --config config/example_config.json
```

## What it produces

Each run writes a top-level `run_metadata.json` and `pipeline.log` into the run directory for full reproducibility.

- **Reduction** (`scripts/qub_reduction.py`)
  - calibrated science cube FITS
  - manifest/time tables (CSV)
  - `products.json` describing file products
  - `reduction.log`

- **Photometry** (`scripts/qub_photometry.py`)
  - differential light curve + diagnostics
  - `photometry.log` (set `--verbose` for debug)

Logs are kept separate per stage so you can quickly see where a run failed.

## Configuration

The pipeline expects a single JSON config as the source of truth. CLI flags are optional overrides.

Key fields:
- `paths.science_dir`, `paths.dark_dir`
- `paths.reg_path` (DS9 regions: target first, then comps)
- `paths.reduction_outdir`, `paths.photometry_outdir`
- `target.ra`, `target.dec` and `observatory.*` (optional; used for airmass/BJD calculations if enabled by your script)

## Reference

`reference/INT_pipeline.py` is included as your benchmark reference pipeline.


## Install (recommended)

```bash
python -m pip install -U pip
python -m pip install -e .
```

After installing, you can run:

```bash
qub-pipeline run --config config/example_config.json
# or
