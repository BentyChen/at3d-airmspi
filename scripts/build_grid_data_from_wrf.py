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
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    wrf_cfg = cfg["wrf_input"]
    z_cfg = cfg["z_source"]
    grid_cfg = cfg["grid"]
    plume_cfg = cfg["plume_partition"]
    micro = cfg["microphysics"]
    ri_cfg = cfg["refractive_index"]
    out_cfg = cfg["output"]

    ds = xr.open_dataset(wrf_cfg["path"], engine="netcdf4")
    t = int(wrf_cfg["time_index"])

    q3d = _get_time_slice(ds[wrf_cfg["q_field"]], t)
    lat2d = _get_time_slice(ds[wrf_cfg["lat_var"]], t)
    lon2d = _get_time_slice(ds[wrf_cfg["lon_var"]], t)

    z_levels_m = _compute_z_levels_m(ds, t, g=float(z_cfg.get("g", 9.81)), reduce=z_cfg.get("reduce", "horizontal_mean"))

    df, geom, options = grid_data_builder.build_from_les_arrays(
        qvapor_zyx=q3d,
        z_levels_m=z_levels_m,
        lat_2d=lat2d,
        lon_2d=lon2d,
        dx_m=float(np.nanmedian(grid_data_builder._latlon_neighbor_dist_km(lat2d, lon2d)[0]) * 1000.0),
        dy_m=float(np.nanmedian(grid_data_builder._latlon_neighbor_dist_km(lat2d, lon2d)[1]) * 1000.0),
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
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
