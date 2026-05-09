#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr
import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract one Time index from WRF netCDF and save as a small file")
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
    ext_cfg = cfg.get("wrf_extract", {})

    src_path = Path(ext_cfg.get("source_path", wrf_cfg["path"]))
    t = int(ext_cfg.get("time_index", wrf_cfg["time_index"]))
    dst_path = Path(ext_cfg.get("output_path", f"{src_path}_t{t}.nc"))
    overwrite = bool(ext_cfg.get("overwrite", True))

    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {dst_path}")

    ds = xr.open_dataset(src_path, engine="netcdf4")
    ds_t = ds.isel(Time=slice(t, t + 1))
    ds_t.to_netcdf(dst_path)

    # Optionally update main wrf_input in-place for downstream builder convenience.
    if bool(ext_cfg.get("update_builder_config", True)):
        wrf_cfg["path"] = str(dst_path)
        wrf_cfg["time_index"] = 0
        cfg["wrf_input"] = wrf_cfg
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        print(f"updated config: wrf_input.path={dst_path}, wrf_input.time_index=0")

    print(f"extracted time index {t} -> {dst_path}")


if __name__ == "__main__":
    main()
