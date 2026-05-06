#!/usr/bin/env python3
"""Merge multi-view, multi-band AirMSPI results into one NetCDF.

Output variables and dimensions are aligned with retrieval-style products:
- Full resolution (dim_x, dim_y): datalon, datalat, dataElevation, dataLand_water_mask
- Downsampled (dim_x_downsampling, dim_y_downsampling): lon, lat, elevation, Land_water_mask
- Multi-view/multi-band (dim_x_downsampling, dim_y_downsampling, dim_view, dim_band):
  I, Q, U, DoLP, ErrI, ErrQ, ErrU, ErrDoLP, theta0, thetav, faipfai0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
import yaml

# Spyder/IDE-friendly defaults (used when no CLI args are provided)
DEFAULT_CONFIG = "config_v6a.yaml"
DEFAULT_INPUTS: list[str] = []  # Optional explicit input file list for Spyder run
DEFAULT_INPUT_GLOB = "*.nc"
DEFAULT_OUTPUT = "AirMSPI_multiview_multiband.nc"
DEFAULT_FACTOR = 25


def _crop_to_factor(a: np.ndarray, factor: int) -> np.ndarray:
    ny, nx = a.shape
    ny2 = (ny // factor) * factor
    nx2 = (nx // factor) * factor
    if ny2 == 0 or nx2 == 0:
        raise ValueError(f"shape {a.shape} is too small for factor={factor}")
    return a[:ny2, :nx2]


def _downsample_mean2d(a: np.ndarray, factor: int) -> np.ndarray:
    c = _crop_to_factor(a, factor)
    ny, nx = c.shape
    return np.nanmean(c.reshape(ny // factor, factor, nx // factor, factor), axis=(1, 3))


def _choose_first(ds: xr.Dataset, names: Iterable[str]) -> str:
    for n in names:
        if n in ds:
            return n
    raise KeyError(f"None of the candidate variables exist: {list(names)}")


def _to_numpy2d(ds: xr.Dataset, varname: str) -> np.ndarray:
    arr = ds[varname].values
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"{varname} must be 2D after squeeze, got {arr.shape}")
    return np.asarray(arr, dtype=np.float64)






def _load_merge_config(cfg_path: str) -> tuple[Path, list[int], list[int], int]:
    cfg_file = Path(cfg_path)
    if not cfg_file.is_absolute():
        cfg_file = Path(__file__).resolve().parent / cfg_file
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root_dir = Path(cfg["output"]["root_dir"])
    if not root_dir.is_absolute():
        root_dir = (cfg_file.parent / root_dir).resolve()

    view_indices = cfg.get("sensor", {}).get("trajectory", {}).get("cross_track_selected_view_indices", []) or []
    bands = [int(w) for w in cfg.get("bands", {}).get("wavelength_nm", [])]
    factor = int(cfg.get("downsample", {}).get("factor", DEFAULT_FACTOR))
    return root_dir, [int(v) for v in view_indices], bands, factor


def _inputs_from_config(root_dir: Path, bands: list[int]) -> list[Path]:
    if bands:
        files = [root_dir / f"AirMSPI_{int(w)}nm.nc" for w in bands]
        return [f for f in files if f.exists()]
    return []

def _looks_like_airmspi_file(path: Path) -> bool:
    """Accept either merged-style or single-band generated AirMSPI files."""
    try:
        with xr.open_dataset(path) as ds:
            vars_set = set(ds.variables)
            merged_style = {"I", "Q", "U", "DoLP", "theta0", "thetav", "faipfai0"}
            single_band_style = {"I_downsampled_registered", "Q_downsampled_registered", "U_downsampled_registered", "DoLP_downsampled_registered", "VZA_downsampled_registered", "RAA_downsampled_registered"}
            lat_candidates = {"datalat", "latitude", "lat"}
            lon_candidates = {"datalon", "longitude", "lon"}
            has_geo = bool(vars_set & lat_candidates) and bool(vars_set & lon_candidates)
            return has_geo and (merged_style.issubset(vars_set) or single_band_style.issubset(vars_set))
    except Exception:
        return False

def _resolve_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge AirMSPI NetCDF files using config_v6a settings")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to config file")
    parser.add_argument("--inputs", nargs="+", help="Optional explicit input NetCDF file list")
    parser.add_argument("--output", type=str, help="Optional explicit output NetCDF file path")
    parser.add_argument("--factor", type=int, help="Optional downsampling factor override")
    return parser.parse_args(argv)


def _resolve_inputs_from_config(config: str) -> tuple[list[Path], Path, list[int], int]:
    root_dir, view_indices, bands, factor = _load_merge_config(config)
    input_files = _inputs_from_config(root_dir, bands)
    if not input_files:
        input_files = sorted(root_dir.glob(DEFAULT_INPUT_GLOB))
    input_files = [p for p in input_files if _looks_like_airmspi_file(p)]
    output_path = root_dir / DEFAULT_OUTPUT
    return input_files, output_path, view_indices, factor


def main(config: str = DEFAULT_CONFIG, inputs: list[str] | None = None, output: str | None = None, factor: int | None = None) -> None:
    root_dir, selected_view_indices, cfg_bands, cfg_factor = _load_merge_config(config)

    if inputs:
        input_files = [Path(i) for i in inputs]
    else:
        auto_inputs, auto_output, _, _ = _resolve_inputs_from_config(config)
        if not auto_inputs:
            raise FileNotFoundError(f"No AirMSPI-like input NetCDF files found from config: {config}")
        input_files = auto_inputs

    # Enforce cross_track_selected_view_indices if files are view-ordered
    if selected_view_indices and len(input_files) >= max(selected_view_indices):
        input_files = [input_files[i - 1] for i in selected_view_indices]

    output_path = Path(output) if output else (root_dir / DEFAULT_OUTPUT)
    ds_factor = factor if factor is not None else cfg_factor

    print(f"[INFO] config={config}")
    print(f"[INFO] root_dir={root_dir}")
    print(f"[INFO] cross_track_selected_view_indices={selected_view_indices}")
    print(f"[INFO] bands={cfg_bands}")
    print(f"[INFO] downsample.factor={ds_factor}")

    opened = [xr.open_dataset(f) for f in input_files]

    lat_name = _choose_first(opened[0], ["datalat", "latitude", "lat"])
    lon_name = _choose_first(opened[0], ["datalon", "longitude", "lon"])
    elev_name = _choose_first(opened[0], ["dataElevation", "elevation"])
    lwm_name = _choose_first(opened[0], ["dataLand_water_mask", "Land_water_mask", "land_water_mask"])

    datalat = _to_numpy2d(opened[0], lat_name)
    datalon = _to_numpy2d(opened[0], lon_name)
    data_elev = _to_numpy2d(opened[0], elev_name)
    data_lwm = _to_numpy2d(opened[0], lwm_name)

    lat_ds = _downsample_mean2d(datalat, ds_factor)
    lon_ds = _downsample_mean2d(datalon, ds_factor)
    elev_ds = _downsample_mean2d(data_elev, ds_factor)
    lwm_ds = _downsample_mean2d(data_lwm, ds_factor)

    stack = []
    bands_ref = None
    for ds in opened:
        i_name = _choose_first(ds, ["I", "I_downsampled_registered"])
        q_name = _choose_first(ds, ["Q", "Q_downsampled_registered"])
        u_name = _choose_first(ds, ["U", "U_downsampled_registered"])
        dolp_name = _choose_first(ds, ["DoLP", "DoLP_downsampled_registered"])
        erri_name = _choose_first(ds, ["ErrI"]) if "ErrI" in ds else None
        errq_name = _choose_first(ds, ["ErrQ"]) if "ErrQ" in ds else None
        erru_name = _choose_first(ds, ["ErrU"]) if "ErrU" in ds else None
        errdolp_name = _choose_first(ds, ["ErrDoLP"]) if "ErrDoLP" in ds else None
        theta0_name = _choose_first(ds, ["theta0", "theta_0"])
        thetav_name = _choose_first(ds, ["thetav", "VZA_downsampled_registered"])
        faipfai0_name = _choose_first(ds, ["faipfai0", "RAA_downsampled_registered"])

        band_name = "band" if "band" in ds.dims else ("dim_band" if "dim_band" in ds.dims else None)
        if band_name is None:
            # fallback: infer single band
            bands = np.array([np.nan], dtype=np.float64)
        else:
            bands = np.asarray(ds[band_name].values, dtype=np.float64)

        if bands_ref is None:
            bands_ref = bands
        elif len(bands_ref) != len(bands):
            raise ValueError("All inputs must have the same band size")

        stack.append(
            dict(
                I=np.asarray(ds[i_name].values, dtype=np.float64),
                Q=np.asarray(ds[q_name].values, dtype=np.float64),
                U=np.asarray(ds[u_name].values, dtype=np.float64),
                DoLP=np.asarray(ds[dolp_name].values, dtype=np.float64),
                ErrI=np.asarray(ds[erri_name].values, dtype=np.float64) if erri_name else np.full_like(np.asarray(ds[i_name].values, dtype=np.float64), np.nan),
                ErrQ=np.asarray(ds[errq_name].values, dtype=np.float64) if errq_name else np.full_like(np.asarray(ds[i_name].values, dtype=np.float64), np.nan),
                ErrU=np.asarray(ds[erru_name].values, dtype=np.float64) if erru_name else np.full_like(np.asarray(ds[i_name].values, dtype=np.float64), np.nan),
                ErrDoLP=np.asarray(ds[errdolp_name].values, dtype=np.float64) if errdolp_name else np.full_like(np.asarray(ds[i_name].values, dtype=np.float64), np.nan),
                theta0=np.asarray(ds[theta0_name].values, dtype=np.float64),
                thetav=np.asarray(ds[thetav_name].values, dtype=np.float64),
                faipfai0=np.asarray(ds[faipfai0_name].values, dtype=np.float64),
            )
        )

    # Expected shape per file: (dim_x_downsampling, dim_y_downsampling, dim_band)
    keys = list(stack[0].keys())
    merged = {k: np.stack([s[k] for s in stack], axis=2) for k in keys}  # -> x, y, view, band

    nx_ds, ny_ds = lat_ds.shape
    nview = len(stack)
    nband = merged["I"].shape[-1]
    if cfg_bands:
        bands_ref = np.asarray(cfg_bands, dtype=np.float64)
    expected_views = len(selected_view_indices or [])
    if expected_views and nview != expected_views:
        print(f"[WARN] config selected views={expected_views}, but merged files={nview}.")

    out = xr.Dataset(
        data_vars={
            "DoLP": (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), merged["DoLP"]),
            "datalon": (("dim_x", "dim_y"), datalon),
            "lon": (("dim_x_downsampling", "dim_y_downsampling"), lon_ds),
            "Height_AirMSPI": (("height_dim", "height_dim2"), np.array([[20.0]], dtype=np.float64)),
            "thetav": (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), merged["thetav"]),
            "Land_water_mask": (("dim_x_downsampling", "dim_y_downsampling"), lwm_ds),
            "elevation": (("dim_x_downsampling", "dim_y_downsampling"), elev_ds),
            "datalat": (("dim_x", "dim_y"), datalat),
            "Band_AirMSPI": (("dim_band", "band_scalar"), np.asarray(bands_ref).reshape(-1, 1)),
            "Q": (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), merged["Q"]),
            "dataLand_water_mask": (("dim_x", "dim_y"), data_lwm),
            "lat": (("dim_x_downsampling", "dim_y_downsampling"), lat_ds),
            "ErrI": (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), merged["ErrI"]),
            "U": (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), merged["U"]),
            "ErrU": (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), merged["ErrU"]),
            "dataElevation": (("dim_x", "dim_y"), data_elev),
            "faipfai0": (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), merged["faipfai0"]),
            "theta0": (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), merged["theta0"]),
            "ErrQ": (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), merged["ErrQ"]),
            "I": (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), merged["I"]),
            "ErrDoLP": (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), merged["ErrDoLP"]),
        },
        coords={
            "dim_x": np.arange(datalat.shape[0]),
            "dim_y": np.arange(datalat.shape[1]),
            "dim_x_downsampling": np.arange(nx_ds),
            "dim_y_downsampling": np.arange(ny_ds),
            "dim_view": np.arange(nview),
            "dim_band": np.arange(nband),
            "height_dim": [0],
            "height_dim2": [0],
            "band_scalar": [0],
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(output_path)
    for ds in opened:
        ds.close()
    print(f"Saved merged file: {output_path}")
    print(f"dims: dim_x={datalat.shape[0]}, dim_y={datalat.shape[1]}, "
          f"dim_x_downsampling={nx_ds}, dim_y_downsampling={ny_ds}, dim_view={nview}, dim_band={nband}")


if __name__ == "__main__":
    args = _resolve_args()
    main(config=args.config, inputs=args.inputs, output=args.output, factor=args.factor)
