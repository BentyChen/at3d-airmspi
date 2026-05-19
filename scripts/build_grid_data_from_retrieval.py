#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:15:51 2026

@author: benting
"""

import grid_data_builder

# Spyder-friendly editable settings.
input_nc = "../data/retrieval_1d/2019_0806_1839_N_Pxl25_3_3.nc"
output_csv = "../data/synthetic_cloud_fields/jpl_les/retrieval_2019_0806_1839_extended.csv"
dx_km = "0.16"
dy_km = "0.16"
z_levels_km = "0.01:0.5:20"
mode_count = "2"

# Uniform-atmosphere mode: keep the grid shape and lat/lon from the nc file,
# but copy all atmospheric retrieval properties from one input pixel to every
# horizontal pixel. Pixel indices are 0-based in the original nc array.
uniform_atmosphere = False
uniform_pixel_row = 12
uniform_pixel_col = 12

args = [
    "--input-nc", input_nc,
    "--output-csv", output_csv,
    "--dx-km", dx_km,
    "--dy-km", dy_km,
    "--z-levels-km", z_levels_km,
    "--mode-count", mode_count,
]

if uniform_atmosphere:
    args.extend([
        "--uniform-atmosphere",
        "--uniform-pixel-row", str(uniform_pixel_row),
        "--uniform-pixel-col", str(uniform_pixel_col),
    ])

grid_data_builder.main(args)
