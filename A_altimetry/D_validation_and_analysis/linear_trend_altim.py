"""
Test linear trend function over the whole SO or using a regional average

Last modified: 14 Oct 2022
"""


import numpy as np
from numpy import ma

import xarray as xr
import pandas as pd
import scipy as sp
import scipy.stats as ss

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.dates as mdates

import sys
#-------------------------------------------------------------------
# Directories
#-------------------------------------------------------------------
voldir = '/Volumes/SamT5/PhD/data/'
griddir = voldir + 'altimetry_cpom/3_grid_dot/'

localdir = '/Users/ocd1n16/PhD_local/'

auxscriptdir = localdir + 'scripts/aux_func/'
sys.path.append(auxscriptdir)
import aux_func_trend as fc


# ALTIMETRY data
geoidtype = 'goco05c' # 'goco05c', 'egm08'
satellite = 'all'
sig = 3

date_start = '2002-07-01'
date_end = '2018-10-01'


altfile = 'dot_' + satellite + '_30bmedian_' + geoidtype + '_sig' + str(sig) + '.nc'

with xr.open_dataset(griddir+altfile) as alt:
    print(alt.keys())

print("Crop altimetry to \n\n > > %s - %s\n\n" % (date_start, date_end))

alt_crop = alt.sel(time=slice(date_start, date_end))
dot = alt_crop.dot
lat = alt_crop.latitude.values
lon = alt_crop.longitude.values

# GRID coordinates
eglat, eglon = np.meshgrid(alt_crop.edge_lat, alt_crop.edge_lon)


sla = dot - dot.mean('time')
#-----------------------------------
# pick a region to compute SLA
#-----------------------------------

weddell = sla.sel(longitude=slice(-60, 0)).mean(('longitude', 'latitude'))
sla_weddell = weddell - weddell.mean()

eant = sla.sel(longitude=slice(0, 150)).mean(('longitude', 'latitude'))
sla_eant = eant - eant.mean()

fig, ax = plt.subplots()
ax.plot(sla_weddell.time, sla_weddell.values, c='m', label='Weddell')
ax.plot(sla_eant.time, sla_eant.values+0.1, c='navy', label='E Ant')
ax.legend()
#-----------------------------------
# DE-TREND 
#-----------------------------------
arr = sla_weddell
confidence = 0.95

ndays = mdates.date2num(list(arr.time.values)) 
#dt = ndays - ndays[0] # for computing linear trend

# mask nans if present
arr = ma.masked_invalid(arr.data)

# extract only valid data and their time indices from masked arrays
arr_vals = arr[~arr.mask].squeeze()
tim = ndays[~arr.mask].squeeze()

slope, interc, r, pp, stderr = ss.linregress(tim, arr_vals)
# confidence interval
ci = stderr * sp.stats.t.ppf((1+confidence)/2., obs-1)

trend = 
arr_det = arr  - (ndays * slope + interc)

fig, ax = plt.subplots()
ax.plot(sla_weddell.time, arr, c='k')
ax.plot(sla_weddell.time, arr_det, c='m', label='de-trended')
ax.legend()


print("\n>>> Computing linear trend at every grid point ..")

date = dot.time.dt.strftime("%m.%Y").values
ndays = mdates.date2num(list(dot.time.values))
dt = ndays - ndays[0]

# DIMENSIONS
londim, latdim, timdim = dot.shape

# compute linear trend at every grid point
interc, slope, ci, pval = [ma.zeros((londim, latdim)) for _ in range(4)]
for r in range(londim):
    for c in range(latdim):
        arr_trend, _ = fc.trend_ci(dot[r,c,:], 0.95)
        interc[r, c] = arr_trend.intercept.values
        slope[r, c] = arr_trend.slope.values
        pval[r, c] = arr_trend.p_val.values

# trend in mm/yr with the GMSLR 
slope_mm_yr = slope * 1e3 * 365.25

# apply a mask to plot only trend values where p_val < 0.1
slope_mm_yr[pval>0.1] = np.nan

# [..] function that plots the trend

# detrend the time series at every point
trend = slope[:,:,np.newaxis] * dt[np.newaxis, np.newaxis, :] + interc[:,:,np.newaxis]
dot_det = dot - trend #- gmslr
  