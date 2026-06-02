#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

import grid_data_builder


def _lognormal_reff_veff(rg_um: float, sigma_g: float) -> tuple[float, float]:
    lns = np.log(float(sigma_g))
    reff = float(rg_um) * np.exp(2.5 * lns * lns)
    veff = np.exp(lns * lns) - 1.0
    return float(reff), float(veff)


def _get_time_slice(da: xr.DataArray, t: int) -> np.ndarray:
    return np.asarray(da.isel(Time=t).values)




def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _latlon_neighbor_dist_km(lat_2d: np.ndarray, lon_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d_north = _haversine_km(lat_2d[:-1, :], lon_2d[:-1, :], lat_2d[1:, :], lon_2d[1:, :])
    d_east = _haversine_km(lat_2d[:, :-1], lon_2d[:, :-1], lat_2d[:, 1:], lon_2d[:, 1:])
    return d_north, d_east



def _latlon_from_grid_index(ny: int, nx: int, lat0_deg: float, lon0_deg: float, dx_m: float, dy_m: float) -> tuple[np.ndarray, np.ndarray]:
    dlat = dy_m / 111320.0
    dlon = dx_m / (111320.0 * np.cos(np.deg2rad(lat0_deg)))
    yy = np.arange(ny, dtype=float)[:, None]
    xx = np.arange(nx, dtype=float)[None, :]
    lat = lat0_deg + yy * dlat
    lon = lon0_deg + xx * dlon
    return lat + np.zeros((ny, nx)), lon + np.zeros((ny, nx))

def _compute_z_levels_m(ds: xr.Dataset, t: int, g: float, reduce: str) -> np.ndarray:
    ph = _get_time_slice(ds["PH"], t)
    phb = _get_time_slice(ds["PHB"], t)
    z_w = (ph + phb) / g
    z_m = 0.5 * (z_w[:-1, :, :] + z_w[1:, :, :])
    if reduce == "horizontal_median":
        return np.nanmedian(z_m, axis=(1, 2))
    return np.nanmean(z_m, axis=(1, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config_wrf_psd.yaml")))
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.exists():
        candidate = Path(__file__).with_name(args.config)
        if candidate.exists():
            config_path = candidate
        else:
            raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = yaml.safe_load(config_path.read_text())
    wrf_cfg = cfg["wrf_input"]
    z_cfg = cfg["z_source"]
    grid_cfg = cfg["grid"]
    plume_cfg = cfg["plume_partition"]
    micro = cfg["microphysics"]
    ri_cfg = cfg["refractive_index"]
    out_cfg = cfg["output"]

    ds = xr.open_dataset(wrf_cfg["path"], engine="netcdf4")
    t = int(wrf_cfg["time_index"])

    q3d_raw = _get_time_slice(ds[wrf_cfg["q_field"]], t)
    q_scale = float(wrf_cfg.get("q_scale", 1.0e-3))
    q3d = np.maximum(q3d_raw, 0.0) * q_scale
    q_max = wrf_cfg.get("q_max")
    if q_max is not None:
        q3d = np.minimum(q3d, float(q_max))

    lat_raw = _get_time_slice(ds[wrf_cfg["lat_var"]], t)
    lon_raw = _get_time_slice(ds[wrf_cfg["lon_var"]], t)

    latlon_mode = wrf_cfg.get("latlon_mode", "from_file")
    if latlon_mode == "from_index":
        ny, nx = lat_raw.shape
        lat2d, lon2d = _latlon_from_grid_index(
            ny=ny,
            nx=nx,
            lat0_deg=float(wrf_cfg["sw_corner_lat_deg"]),
            lon0_deg=float(wrf_cfg["sw_corner_lon_deg"]),
            dx_m=float(wrf_cfg.get("grid_dx_m", 40.0)),
            dy_m=float(wrf_cfg.get("grid_dy_m", 40.0)),
        )
        dx_m_est = float(wrf_cfg.get("grid_dx_m", 40.0))
        dy_m_est = float(wrf_cfg.get("grid_dy_m", 40.0))
    else:
        lat2d, lon2d = lat_raw, lon_raw
        dx_north_km, dy_east_km = _latlon_neighbor_dist_km(lat2d, lon2d)
        dx_m_est = float(np.nanmedian(dx_north_km) * 1000.0)
        dy_m_est = float(np.nanmedian(dy_east_km) * 1000.0)

    z_levels_m = _compute_z_levels_m(ds, t, g=float(z_cfg.get("g", 9.81)), reduce=z_cfg.get("reduce", "horizontal_mean"))

    df, geom, options = grid_data_builder.build_from_les_arrays(
        qvapor_zyx=q3d,
        z_levels_m=z_levels_m,
        lat_2d=lat2d,
        lon_2d=lon2d,
        dx_m=dx_m_est,
        dy_m=dy_m_est,
        coarse=tuple(int(v) for v in grid_cfg["coarse"]),
        rho_air_kgm3=float(grid_cfg["rho_air_kgm3"]),
        default_reff=12.0,
        default_veff=0.10,
    )

    # plume/background partition on coarse q field from resulting cv proxy
    thr = float(plume_cfg["threshold"])
    plume_mask = df["cv"].values > thr

    reff_f, veff_f = _lognormal_reff_veff(micro["fine"]["rg_um"], micro["fine"]["sigma_g"])
    reff_c, veff_c = _lognormal_reff_veff(micro["coarse"]["rg_um"], micro["coarse"]["sigma_g"])

    basis = micro.get("mode_fraction_basis", "number")
    key = "fine_fraction_number" if basis == "number" else "fine_fraction_volume"
    ff_plume = float(micro["plume"][key])
    ff_bg = float(micro["background"][key])

    mode1_fraction = np.where(plume_mask, ff_plume, ff_bg)
    mode2_fraction = 1.0 - mode1_fraction

    wl = ri_cfg["wavelength_nm"]
    bi = int(ri_cfg["band_index"])
    if bi < 0 or bi >= len(wl):
        raise ValueError("band_index out of range")

    plume_mr = float(ri_cfg["plume_n"][bi])
    plume_mi = float(ri_cfg["plume_k"][bi])
    bg_mr = float(ri_cfg["background_n"][bi])
    bg_mi = float(ri_cfg["background_k"][bi])

    # expand to 2-mode columns
    df["mode1_fraction"] = mode1_fraction
    df["mode1_reff"] = reff_f
    df["mode1_veff"] = veff_f
    df["mode1_mr"] = np.where(plume_mask, plume_mr, bg_mr)
    df["mode1_mi"] = np.where(plume_mask, plume_mi, bg_mi)

    df["mode2_fraction"] = mode2_fraction
    df["mode2_reff"] = reff_c
    df["mode2_veff"] = veff_c
    df["mode2_mr"] = np.where(plume_mask, plume_mr, bg_mr)
    df["mode2_mi"] = np.where(plume_mask, plume_mi, bg_mi)

    options.mode_count = 2
    options.include_per_grid_refractive_index = True

    out = Path(out_cfg["csv_path"])
    if out.exists() and not bool(out_cfg.get("overwrite", False)):
        raise FileExistsError(f"Output exists: {out}")

    out_path = grid_data_builder.write_extended_grid_csv(str(out), df, geom, options)
    print(f"config: {config_path}")
    print(f"q_field={wrf_cfg['q_field']}, q_scale={q_scale}, q_max={q_max}")
    print(f"cv range [g/m^3]: min={float(np.nanmin(df['cv'])):.6g}, max={float(np.nanmax(df['cv'])):.6g}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
