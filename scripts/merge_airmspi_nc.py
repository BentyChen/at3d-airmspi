#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import yaml


def _load_cfg(cfg_path: str):
    p = Path(cfg_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    with open(p, 'r', encoding='utf-8') as f:
        c = yaml.safe_load(f)
    root = Path(c['output']['root_dir'])
    if not root.is_absolute():
        root = (p.parent / root).resolve()
    bands = [int(v) for v in c['bands']['wavelength_nm']]
    views = [int(v) for v in c.get('sensor', {}).get('trajectory', {}).get('cross_track_selected_view_indices', [])]
    return root, bands, views


def _downsample_mean2d(a: np.ndarray, factor: int) -> np.ndarray:
    ny, nx = a.shape
    ny2 = (ny // factor) * factor
    nx2 = (nx // factor) * factor
    c = a[:ny2, :nx2]
    return np.nanmean(c.reshape(ny2 // factor, factor, nx2 // factor, factor), axis=(1, 3))




def _resolve_view_indices(ds: xr.Dataset, selected_views: list[int]) -> list[int]:
    nview_all = int(ds.sizes['view'])
    if not selected_views:
        return list(range(nview_all))

    # If file already contains only selected views (common case), keep all
    if max(selected_views) >= nview_all:
        # try matching by view labels like "view_1", "view_3"
        if 'view' in ds.coords:
            labels = [str(v) for v in ds['view'].values]
            mapped = []
            for v in selected_views:
                key = f'view_{v}'
                if key in labels:
                    mapped.append(labels.index(key))
            if mapped:
                return mapped
        return list(range(nview_all))

    return [v - 1 for v in selected_views]

def main(config: str = 'config_v6a.yaml', factor: int = 25, output: str | None = None):
    root_dir, bands, selected_views = _load_cfg(config)
    files = [root_dir / f'AirMSPI_{b}nm.nc' for b in bands]
    miss = [str(f) for f in files if not f.exists()]
    if miss:
        raise FileNotFoundError(f'Missing band files: {miss}')

    opened = [xr.open_dataset(f) for f in files]
    d0 = opened[0]

    # full-res geometry (single-band文件没有lat/lon，先用像素网格填充)
    ny_full = int(d0.sizes['y'])
    nx_full = int(d0.sizes['x'])
    datalat = np.full((ny_full, nx_full), np.nan, dtype=np.float64)
    datalon = np.full((ny_full, nx_full), np.nan, dtype=np.float64)
    dataElevation = np.zeros((ny_full, nx_full), dtype=np.float64)
    dataLandWater = np.ones((ny_full, nx_full), dtype=np.float64)

    # downsampled registered grid
    ny_ds = int(d0.sizes['y_gds'])
    nx_ds = int(d0.sizes['x_gds'])
    view_idx = _resolve_view_indices(d0, selected_views)
    nview = len(view_idx)
    nband = len(bands)

    I = np.full((ny_ds, nx_ds, nview, nband), np.nan, dtype=np.float64)
    Q = np.full_like(I, np.nan)
    U = np.full_like(I, np.nan)
    DoLP = np.full_like(I, np.nan)
    thetav = np.full_like(I, np.nan)
    faipfai0 = np.full_like(I, np.nan)
    theta0 = np.full_like(I, np.nan)
    dataI = np.full((nview, nband, ny_full, nx_full), np.nan, dtype=np.float64)

    for ib, ds in enumerate(opened):
        I[..., ib] = np.transpose(ds['I_downsampled_registered'].values[view_idx, :, :], (1, 2, 0))
        Q[..., ib] = np.transpose(ds['Q_downsampled_registered'].values[view_idx, :, :], (1, 2, 0))
        U[..., ib] = np.transpose(ds['U_downsampled_registered'].values[view_idx, :, :], (1, 2, 0))
        DoLP[..., ib] = np.transpose(ds['DoLP_downsampled_registered'].values[view_idx, :, :], (1, 2, 0))
        thetav[..., ib] = np.transpose(ds['VZA_downsampled_registered'].values[view_idx, :, :], (1, 2, 0))
        faipfai0[..., ib] = np.transpose(ds['RAA_downsampled_registered'].values[view_idx, :, :], (1, 2, 0))
        theta0_val = float(np.nanmean(ds['theta0_original'].values)) if 'theta0_original' in ds else 0.0
        theta0[..., ib] = theta0_val
        dataI[:, ib, :, :] = ds['I_original'].values[view_idx, :, :]

    # 25x25 from full-res dims
    lat = _downsample_mean2d(datalat, factor)
    lon = _downsample_mean2d(datalon, factor)
    elevation = _downsample_mean2d(dataElevation, factor)
    land = _downsample_mean2d(dataLandWater, factor)

    ErrI = np.zeros_like(I)
    ErrQ = np.zeros_like(I)
    ErrU = np.zeros_like(I)
    ErrDoLP = np.zeros_like(I)

    out = xr.Dataset(
        data_vars={
            'DoLP': (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), DoLP),
            'datalon': (("dim_x", "dim_y"), datalon),
            'lon': (("dim_x_downsampling", "dim_y_downsampling"), lon),
            'Height_AirMSPI': (("height_dim", "height_dim2"), np.array([[20.0]], dtype=np.float64)),
            'thetav': (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), thetav),
            'Land_water_mask': (("dim_x_downsampling", "dim_y_downsampling"), land),
            'dataI': (("dim_view", "dim_band", "dim_x", "dim_y"), dataI),
            'elevation': (("dim_x_downsampling", "dim_y_downsampling"), elevation),
            'datalat': (("dim_x", "dim_y"), datalat),
            'Band_AirMSPI': (("dim_band", "band_scalar"), np.asarray(bands, dtype=np.float64).reshape(-1, 1)),
            'Q': (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), Q),
            'dataLand_water_mask': (("dim_x", "dim_y"), dataLandWater),
            'lat': (("dim_x_downsampling", "dim_y_downsampling"), lat),
            'ErrI': (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), ErrI),
            'U': (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), U),
            'ErrU': (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), ErrU),
            'dataElevation': (("dim_x", "dim_y"), dataElevation),
            'faipfai0': (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), faipfai0),
            'theta0': (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), theta0),
            'ErrQ': (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), ErrQ),
            'I': (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), I),
            'ErrDoLP': (("dim_x_downsampling", "dim_y_downsampling", "dim_view", "dim_band"), ErrDoLP),
        }
    )

    out_path = Path(output) if output else (root_dir / 'AirMSPI_multiview_multiband.nc')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_path)
    for ds in opened:
        ds.close()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge single-band AirMSPI_*nm.nc files into one multiband product')
    parser.add_argument('--config', type=str, default='config_v6a.yaml', help='Path to config file')
    parser.add_argument('--factor', type=int, default=25, help='Downsampling factor')
    parser.add_argument('--output', type=str, default=None, help='Output merged NetCDF path')
    args = parser.parse_args()
    main(config=args.config, factor=args.factor, output=args.output)
