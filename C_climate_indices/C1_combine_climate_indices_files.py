"""
Combine climate indices to create the file climate_indices_79_19.nc
ASL not included because I decided to download it after I created this file.

Last modified: 10 Mar 2025
"""


# climate indices analysis
import numpy as np
from numpy import ma

from netCDF4 import num2date

import xarray as xr

import pandas as pd

import scipy.stats as ss
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import aux_func_trend as ft

def plot_idx(idx_var, idx_time, title):
    pos_val = idx_var[idx_var>0]
    neg_val = idx_var[idx_var<0]
    pos_time = idx_time[idx_var>0]
    neg_time = idx_time[idx_var<0]

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axhline(0, ls=':', c='grey', zorder=1)
    ax.plot(idx_time, idx_var, c='dimgrey', lw=1.5, zorder=1)
    
    ax.scatter(pos_time, pos_val, c='crimson', s=15, zorder=2)
    ax.scatter(neg_time, neg_val, c='b', s=15, zorder=2)
    
    xticks = pd.to_datetime(idx_time)
    ax.set_xticks(xticks, minor=True)
    ax.set_xlim(idx_time[0]-np.timedelta64(60, 'D'),
        idx_time[-1]+np.timedelta64(60, 'D'))
    ax.grid(True, which="minor", c='lightgrey')
    ax.grid(True, which="major")
    
    ax.set_title(title, loc='left')
    plt.tight_layout()

    return ax

# Directories
workdir = '/Volumes/SamT5/PhD/PhD_data/'
idxdir = workdir + 'climate_indices/'
figdir = workdir + '../PhD_figures/'
#----------------------------------------------
# files
#----------------------------------------------
sam_file = idxdir + 'sam_nerc_bas.txt'

soi_ncar_file = idxdir + 'soi_ncar_annual_stand.txt'
soi_ncep_file = idxdir + 'SOI_ncepNOAA_stand.txt'

nino3_file = idxdir + 'nino3_noaa.txt'
nino34_file = idxdir + 'nino34_noaa.txt'
nino4_file = idxdir + 'nino4_noaa.txt'

asl_file = idxdir + 'SOI_Index_NOAA.txt'
#----------------------------------------------
# SAM
#----------------------------------------------
file_data = np.loadtxt(sam_file)

file_yr = file_data[:,0].astype(int)
date_start = str(file_yr[0]) + '-01-01'
date_end = str(file_yr[-1]) + '-12-01'

file_index = file_data[:,1:].flatten()
sam_all = xr.Dataset({'sam':("time",file_index)},
                 coords={"time":pd.date_range(date_start, date_end,
                                              freq="1MS")})
index = sam_all.sel(time=slice('1979-01-01', '2019-12-01'))
#----------------------------------------------
# SOI - ncar
#----------------------------------------------
file_data = np.loadtxt(soi_ncar_file)

file_yr = file_data[:,0].astype(int)
date_start = str(file_yr[0]) + '-01-01'
date_end = str(file_yr[-1]) + '-12-01'

file_index = file_data[:,1:].flatten()
soi_ncar_all = xr.Dataset({'soi_ncar':("time",file_index)},
                 coords={"time":pd.date_range(date_start, date_end,
                                              freq="1MS")})
soi1_crop = soi_ncar_all.sel(time=slice('1979-01-01', '2019-12-01'))

index['soi_ncar'] = ('time', soi1_crop.soi_ncar.values)
#----------------------------------------------
# SOI - ncep
#----------------------------------------------
file_data = np.loadtxt(soi_ncep_file)

file_yr = file_data[:,0].astype(int)
date_start = str(file_yr[0]) + '-01-01'
date_end = str(file_yr[-1]) + '-12-01'

file_index = file_data[:,1:].flatten()
soi_ncep_all = xr.Dataset({'soi_ncep':("time",file_index)},
                 coords={"time":pd.date_range(date_start, date_end,
                                              freq="1MS")})
soi2_crop = soi_ncep_all.sel(time=slice('1979-01-01', '2019-12-01'))

index['soi_ncep'] = ('time', soi2_crop.soi_ncep.values)
#----------------------------------------------
# Nino 3 
#----------------------------------------------
file_data = np.loadtxt(nino3_file)

file_yr = file_data[:,0].astype(int)
date_start = str(file_yr[0]) + '-01-01'
date_end = str(file_yr[-1]) + '-12-01'

file_index = file_data[:,1:].flatten()
nino3_all = xr.Dataset({'nino3':("time",file_index)},
                 coords={"time":pd.date_range(date_start, date_end,
                                              freq="1MS")})
nino3_crop = nino3_all.sel(time=slice('1979-01-01', '2019-12-01'))

index['nino3'] = ('time', nino3_crop.nino3.values)
#----------------------------------------------
# Nino 3.4
#----------------------------------------------
file_data = np.loadtxt(nino34_file)

file_yr = file_data[:,0].astype(int)
date_start = str(file_yr[0]) + '-01-01'
date_end = str(file_yr[-1]) + '-12-01'

file_index = file_data[:,1:].flatten()
nino34_all = xr.Dataset({'nino34':("time",file_index)},
                 coords={"time":pd.date_range(date_start, date_end,
                                              freq="1MS")})
nino34_crop = nino34_all.sel(time=slice('1979-01-01', '2019-12-01'))

index['nino34'] = ('time', nino34_crop.nino34.values)
#----------------------------------------------
# Nino 4
#----------------------------------------------
file_data = np.loadtxt(nino4_file)

file_yr = file_data[:,0].astype(int)
date_start = str(file_yr[0]) + '-01-01'
date_end = str(file_yr[-1]) + '-12-01'

file_index = file_data[:,1:].flatten()
nino4_all = xr.Dataset({'nino4':("time",file_index)},
                 coords={"time":pd.date_range(date_start, date_end,
                                              freq="1MS")})
nino4_crop = nino4_all.sel(time=slice('1979-01-01', '2019-12-01'))

index['nino4'] = ('time', nino4_crop.nino4.values)
#----------------------------------------------
# save file
#----------------------------------------------

#index.to_netcdf(idxdir + 'climate_indices_79_19.nc')
#----------------------------------------------
#----------------------------------------------

sys.exit()

# - - - - - - - - crop data 
crop_start = '2010-10-01'
crop_end = '2015-12-01'

indx = index.sel(time=slice(crop_start, crop_end))

# - - - - - - - - 
# standardise data (remove mean, divide by std; mean=0, std=1)
indx_anom = indx - indx.mean()
indx_stand = indx_anom / indx.std(ddof=1)

xtim = indx.time.values

# - - - - - - - - 
# standardised PLOT
# - - - - - - - - 
ax = plot_idx(indx_stand.sam.values, xtim, 'SAM')
ax = plot_idx(indx_stand.soi_ncep.values, xtim, 'SOI-NCEP')
ax = plot_idx(indx_stand.soi_ncar.values, xtim, 'SOI_NCAR')
ax = plot_idx(indx_stand.nino3.values, xtim, 'Nino 3')
ax = plot_idx(indx_stand.nino34.values, xtim, 'Nino 3.4')
ax = plot_idx(indx_stand.nino4.values, xtim, 'Nino 4')

# - - - - - - - - 
# SEASONAL CYCLE removed 
# - - - - - - - - 
indx_mclim = indx.groupby("time.month").mean("time")
indx_mclim_anom = indx.groupby("time.month") - indx_mclim

ax = plot_idx(indx_mclim_anom.sam.values, xtim, 'SAM, seas')
ax = plot_idx(indx_mclim_anom.soi_ncep.values, xtim, 'SOI-NCEP, seas')
ax = plot_idx(indx_mclim_anom.soi_ncar.values, xtim, 'SOI_NCAR, seas')
ax = plot_idx(indx_mclim_anom.nino3.values, xtim, 'Nino 3, seas')
ax = plot_idx(indx_mclim_anom.nino34.values, xtim, 'Nino 3.4, seas')
ax = plot_idx(indx_mclim_anom.nino4.values, xtim, 'Nino 4, seas')

# compare the two SOI indices
rcoeff, pval = pearsonr(indx_stand.soi_ncep.values, 
                            indx_stand.soi_ncar.values)
fig, ax = plt.subplots(figsize=(10, 3))
ax.axhline(0, ls=':', c='grey')
ax.plot(xtim, indx_stand.soi_ncep.values, c='k', label='SOI-NCEP')
ax.plot(xtim, indx_stand.soi_ncar.values, c='teal', label='SOI-NCAR')
ax.set_xlim(xtim[0]-np.timedelta64(60, 'D'), 
    xtim[-1] + np.timedelta64(60, 'D'))
ax.legend()
plt.tight_layout()

# compare the three nino indices
rcoeff, pval = pearsonr(indx_stand.nino3.values, 
                            indx_stand.nino34.values)
fig, ax = plt.subplots(figsize=(10, 3))
ax.axhline(0, ls=':', c='grey')
ax.plot(xtim, indx_stand.nino3.values, c='indigo', label='Nino 3')
ax.plot(xtim, indx_stand.nino34.values, c='teal', label='Nino 3.4')
ax.plot(xtim, indx_stand.nino4.values, c='darkorange', label='Nino 4')
ax.set_xlim(xtim[0]-np.timedelta64(60, 'D'), 
    xtim[-1] + np.timedelta64(60, 'D'))
ax.legend()
plt.tight_layout()

"""
# positive values with red, negative with blue
pos_vals = index.idx_standard.where(index.idx_standard > 0)
neg_vals = index.idx_standard.where(index.idx_standard < 0)

# - - - - - - - - 
# check and remove trend if significant (check p-val)
idx_ndays = mdates.date2num(list(clim_all.time.values)) 
idx_dt = idx_ndays - idx_ndays[0] # for computing linear trend

# trend per day using the standardised index
trend = ft.trend_ci(idx_ndays, clim_all.idx.values, 0.95)
print("\n Linear trend computed from monthly averages")
print("trend [/yr]: %s" % (trend.slope.values*365.25))
print("CI [/yr]: %s" % (trend.ci.values*365.25))
print("p-value: %s" % (trend.p_val.values))

idx_det = clim_all.idx - trend.slope.values * idx_dt
clim_all["idx_det"] = (("time", idx_det.values))
"""