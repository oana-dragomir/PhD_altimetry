"""
~~ Drake Passage South

Import Ocean Bottom Pressure (OBP) data.
Compute monthly averages for the OBP.

Import SSHA data.
Compute correlation maps from the overlapping period 

Last modified: 16 Apr 2021
"""
import numpy as np
from numpy import ma

import pandas as pd
import xarray as xr
import sys

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.dates as mdates

#-------------------------------------------------------------------
# Directories
#-------------------------------------------------------------------
voldir = '/Volumes/SamT5/PhD/data/'
griddir = voldir + 'altimetry_cpom/3_grid/'
bprdir = voldir + 'BPRs/DPsouth/'
topodir = voldir + 'topog/'
lmdir = voldir + 'land_masks/'

localdir = '/Users/ocd1n16/PhD_local/'
figdir = localdir + 'data_figures/Figures_apr21/'

auxscriptdir = localdir + 'scripts/aux_func/'
sys.path.append(auxscriptdir)
import aux_func_trend as fc
import aux_stereoplot as st
import aux_corr_maps as rmap

# function that extracts altimetry data
sys.path.append(localdir + 'scripts/D_validation_and_analysis/')
import d2_extract_altim_anom as altim

# -----------------------------------------------------------------------------
# Bottom Pressure data

# Drake Passage South
# [2000-11-28] 1992.11.08 to 2011-12-06
filename = 'dps_all.txt'

# BPR positions
lat_myrtle, lon_myrtle = -60.6200, -53.8488
lat_dps, lon_dps = -60.8570, -54.7147
lat_dpsd, lon_dpsd = -60.8249, -54.7264

# BPR depths (min/max) in m
# DPS : 984-1170
# DPS deep: 1920-1980
# Myrtle C: 2793

# -----------------------------------------------------------------------------
# read lines and store only necessary columns in lists
flag, year, yrday, hour, press, pred_tide, drift, residual =[[] for _ in range(8)]
with open(bprdir+filename,'r') as source:
    for line in source:
        a = line.split("\t")
        aa = np.fromstring(a[0], dtype=float, sep='\t')
        flag.append(aa[1].astype(int))
        year.append(aa[2].astype(int))
        yrday.append(aa[3].astype(int))
        hour.append(aa[4])
        #press.append(aa[5])
        #pred_tide.append(aa[6])
        #drift.append(aa[7])
        residual.append(aa[8])

# remove data where flag != 0
numi = len(year)
print("Initial number of observations: %s" %len(year))
print("Removing flagged data ...")

flag = np.asarray(flag)
def mask_flagged(arr):
    marr = ma.masked_array(arr)
    marr[flag!=0] = ma.masked
    return marr.compressed()

year = mask_flagged(year)
yrday = mask_flagged(yrday)
hour = mask_flagged(hour)
residual = mask_flagged(residual)

numf = len(year)
numflag = numi-numf
print("Number of flagged observations: %s \n" % numflag)

# --------------------------------------------------------
# Monthly averages of OBP
# --------------------------------------------------------
print("Convert day of the year to date format ..\n")
month, day = [ma.zeros((len(year),)) for _ in range(2)]
obp_date = []
for i in range(len(month)):
    yy, dd = str(year[i]), str(yrday[i])
    date = pd.to_datetime(yy+dd, format="%Y%j")
    obp_date.append(date)
    month[i] = date.month
    day[i] = date.day

print("Convert time and obp values into xarray dataset ..\n")
bpr = xr.DataArray(residual, coords={'time': obp_date}, dims='time')

#sort array by time coord due to some overlap in sampling
bpr = bpr.sortby('time')

# average the predicted  bottom pressure monthly
monthly_bpr = bpr.resample(time='1MS').mean()

# -----------------------------------------------------------------------------
print("Loading altimetry file .. \n")
# -----------------------------------------------------------------------------
#altfile = 'dot_all_30bmedian_goco05c.nc'
altfile = 'dot_all_30bmedian_egm08.nc'

with xr.open_dataset(griddir + altfile) as alt:
    print(alt.keys())

# - - - - - - - - - - - - - - - - -
# GRID coordinates
# - - - - - - - - - - - - - - - - -
# at bin edges
alt_eglat, alt_eglon = np.meshgrid(alt.edge_lat, alt.edge_lon)
# at bin centres
alt_glat, alt_glon = np.meshgrid(alt.latitude, alt.longitude)
# - - - - - - - - - - - - - - - - -

londim, latdim, timdim = alt.dot.shape

# -----------------------------------------------------------------------------
# overlap period
# 2000-11-28 to 2011-12-06 [2002.07 - 2011.11]
time_start = '2002-07-01'
time_end = '2011-11-01'

obp_overlap = monthly_bpr.sel(time=slice(time_start, time_end))
dot_overlap = alt.dot.sel(time=slice(time_start, time_end))

# - - - - - - - - - - - - - - - - -
date_overlap = dot_overlap.time.dt.strftime("%Y/%m").values
ndays_overlap =  mdates.date2num(list(dot_overlap.time.values)) 
# - - - - - - - - - - - - - - - - -

#------------------------------------------------------------------
# multi-year anomaly 
# - - - - - - - - - - - - - - - - -
sla_overlap = dot_overlap - dot_overlap.mean('time')
obp_anom_overlap = obp_overlap - obp_overlap.mean('time')

# -----------------------------------------------------------------
#            ~ ~ ~     PLOTS    ~ ~ ~
# -----------------------------------------------------------------
cbar_range = [-.7, .7]
cmap = cm.get_cmap('RdBu_r', 17)

# with linear trend, with seasonal cycle
corr, pval = rmap.r_map_ts(sla_overlap.values,
                          obp_anom_overlap.values)
corr[pval>0.1] = np.nan
fig, ax, m = rmap.plot_signif_corr(corr, pval, alt_glon,
                            alt_glat, cbar_range, cmap)
m.scatter(lon_dps, lat_dps, marker='x', c='k',
    s=40, latlon=True, zorder=6)
#------------------------------------------------------------------
# detrend data
# - - - - - - - - - - - - - - - - -
print("Computing linear trends ..")

# - - - - - - - DOT - - - - - - - - - -
interc, slope, ci, pval = [ma.zeros((londim, latdim)) for _ in range(4)]
for r in range(londim):
  for c in range(latdim):
      arr_trend = fc.trend_ci(ndays_overlap, dot_overlap.values[r,c,:], 0.95)
      interc[r, c] = arr_trend.intercept.values
      slope[r, c] = arr_trend.slope.values
      pval[r, c] = arr_trend.p_val.values

dot_trend = (slope[:,:,np.newaxis] * ndays_overlap[np.newaxis, np.newaxis, :] 
            + interc[:,:,np.newaxis])
dot_det_overlap = dot_overlap - dot_trend

# - - - - - - - DOT det anom - - - - - - - - -
sla_det_overlap = dot_det_overlap - dot_det_overlap.mean('time')


# - - - - - - - - - OBP  - - - - - - - -
arr_trend = fc.trend_ci(ndays_overlap, obp_overlap, 0.95)
obp_trend = arr_trend.slope.values * ndays_overlap + arr_trend.intercept.values

# OPB det anomaly
obp_det_overlap = obp_overlap - obp_trend
obp_det_anom_overlap = obp_det_overlap - obp_det_overlap.mean('time')

# -----------------------------------------------------------------
#            ~ ~ ~     PLOTS    ~ ~ ~
# -----------------------------------------------------------------
cbar_range = [-.7, .7]
cmap = cm.get_cmap('RdBu_r', 17)

# with linear trend, with seasonal cycle
corr_det, pval_det = rmap.r_map_ts(sla_det_overlap.values,
                          obp_det_anom_overlap.values)
corr_det[pval_det>0.1] = np.nan
fig, ax, m = rmap.plot_signif_corr(corr_det, pval_det, alt_glon,
                            alt_glat, cbar_range, cmap)
m.scatter(lon_dps, lat_dps, marker='x', c='k',
    s=40, latlon=True, zorder=6)

