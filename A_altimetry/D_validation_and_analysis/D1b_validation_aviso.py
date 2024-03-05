"""
AVISO/CMEMS: 1993.01 - 2018.12
- Absolute Dynamic Topography (SLA added to MDT)

ENV + CS2 (my data): 2002.07 - 2018.10
- Dynamic Ocean Topography (egm2008, GOCO05c)

> Compute anomalies relative to the overlap time series mean
> Compute correlation maps

Last modified: 20 Apr 2021
"""
import numpy as np
from numpy import ma

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import pandas as pd
import xarray as xr

import scipy.io as sio
import scipy.stats as ss
import scipy as sp
from scipy.stats.stats import pearsonr

import sys

plt.ion()
#-------------------------------------------------------------------
# Directories
#-------------------------------------------------------------------
voldir = '/Volumes/SamT5/PhD/data/'
griddir = voldir + 'altimetry_cpom/3_grid_dot/'
avisodir = voldir + 'aviso/'

localdir = '/Users/ocd1n16/PhD_local/'
figdir = localdir + 'data_figures/Figures_apr21/'

auxscriptdir = localdir + 'scripts/aux_func/'
sys.path.append(auxscriptdir)
import aux_corr_maps as rmap

# function that extracts altimetry data
sys.path.append(localdir + 'scripts/D_validation_and_analysis/')
import d2_extract_altim_anom as altim

# -----------------------------------------------------------------------------
print("Loading altimetry file .. \n")
# -----------------------------------------------------------------------------
altfile = 'dot_all_30bmedian_goco05c.nc'
#altfile = 'dot_all_30bmedian_egm08.nc'

with xr.open_dataset(griddir + altfile) as alt:
    print(alt.keys())

# - - - - - - - - - - - - - - - - -
# GRID coordinates
# - - - - - - - - - - - - - - - - -
# at bin edges
alt_eglat, alt_eglon = np.meshgrid(alt.edge_lat, alt.edge_lon)
# at bin centres
alt_glat, alt_glon = np.meshgrid(alt.latitude, alt.longitude)

# -----------------------------------------------------------------------------
#avisofile = 'monthly_50s_adt_aviso_93_18.nc'
avisofile = 'cmems_adt_1993_2018.nc'
# using SLA or adt data is the same!!

with xr.open_dataset(avisodir + avisofile) as cmems:
    print(cmems.keys())

# - - - - - - - - - - - - - - - - -
# overlap period
# - - - - - - - - - - - - - - - - -
date_start = '2011-11-01'
date_end = '2017-11-01'

# de-trend data
#geoidtype = 'egm08' # 'goco05c'
#alt_crop = altim.extract_dot_maps(geoidtype, date_start, date_end)

cmems_crop = cmems.gadt.sel(time=slice(date_start, date_end))
alt_crop = alt.dot.sel(time=slice(date_start, date_end))

cmems_anom = cmems_crop - cmems_crop.mean('time')
alt_anom = alt_crop - alt_crop.mean('time')

# - - - - - - - - - - - - - - - - -
# mask regions where coverage is less than 70%
# - - - - - - - - - - - - - - - - -
# number of valid instances at every grid point
# convert into percentage
cmems_count = cmems_anom.count('time')
cmems_percent = cmems_count/len(cmems_anom.time)

thresh = 0.8  # consider only values where >80% of the time there are measurm.
cmems_anom = cmems_anom.transpose('glon', 'glat', 'time')
cmems_anom.values[cmems_percent<thresh] = np.nan

alt_anom = alt_anom.transpose('longitude', 'latitude', 'time')
#alt_anom = alt_anom.transpose('lon', 'lat', 'time')
alt_anom.values[cmems_percent<thresh] = np.nan

# -----------------------------------------------------------------------------
# correlation map
# -----------------------------------------------------------------------------
print("Computing correlation map and preparing plot \n")

londim, latdim, timdim = alt_anom.shape

corr_map, p_map = [ma.zeros((londim, latdim)) for _ in range(2)]

for i in range(londim):
    for j in range(latdim):
        altim_ts = ma.masked_invalid(alt_anom[i, j, :].values)
        cmems_ts = ma.masked_invalid(cmems_anom[i, j, :].values)

        # mask an entry if it's missing from the other time series
        cmems_mts = ma.masked_array(cmems_ts, altim_ts.mask)
        altim_mts = ma.masked_array(altim_ts, cmems_ts.mask)
        
        if cmems_mts.count() > 12 and altim_mts.count() > 12:
            corr_map[i, j], p_map[i, j] = pearsonr(altim_mts.compressed(),
                                                cmems_mts.compressed())
        else:
            corr_map[i, j] = p_map[i, j]= ma.masked

# -----------------------------------------------------------------------------
cbar_range = [-1, 1]
cmap = cm.get_cmap('RdBu_r', 17)
corr_map[p_map>0.1] = np.nan

fig, ax, m = rmap.plot_signif_corr(corr_map, p_map, alt_glon,
                            alt_glat, cbar_range, cmap)

"""
params = {
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    'figure.figsize': [9, 8]
}

from matplotlib import rcParams
rcParams.update(params)
"""