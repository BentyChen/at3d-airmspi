#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:08:54 2026

@author: benting
"""

from airmspi_image_simulation import plot_simulation_results

# 方案1：IQU一张 + 角度四图一张
plot_simulation_results(
    "output/airmspi_sim_8_004/registered/470nm_view_9.npz",
    option="option1"
)

plot_simulation_results(
    "output/airmspi_sim_8_004/original/470nm_view_1.npz",
    option="option1"
)