import numpy as np
from numpy import ma
from netCDF4 import num2date

import xarray as xr
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import detrend

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

from palettable.cmocean.diverging import Balance_19
from palettable.scientific.diverging import Vik_19
from palettable.colorbrewer.diverging import RdBu_11_r

from pycurrents.num import interp1

import aux_stereoplot as st 
import aux_func_trend as ft
import matlab_func as matf

import sys
from copy import deepcopy

import gsw

def r_map_ts(arr, ts):
    """
    ts must have size tim
    arr must have the shape (lon, lat, tim)
    """
    #  correlation between ug in the holland box and u10
    arr = ma.masked_invalid(arr)
    ts = ma.masked_invalid(ts)
    
    corr_map, pval_map = [ma.zeros(arr.shape[:-1]) for _ in range(2)]
    londim, latdim = arr.shape[:-1]
    for i in range(londim):
        for j in range(latdim):
            map_ij = arr[i, j, :]
            
            ts_m = ma.masked_array(ts, map_ij.mask)
            map_ij = ma.masked_array(map_ij, ts_m.mask)

            if ts.count() > 3 and map_ij.count() >3:
                corr_map[i, j], pval_map[i, j] = pearsonr(ts_m.compressed(),
                                                          map_ij.compressed())
            else:
                corr_map[i, j] = pval_map[i, j] = ma.masked
    return corr_map, pval_map

def plot_corr(var, pval_var, var_glon, var_glat):
  cbar_range = [-.7, .7]
  cmap = cm.get_cmap('RdBu_r', 11)
  cbar_units = 'r'
  cbar_extend='both'

  plt.ion()
  fig, ax, m = st.spstere_plot(var_glon, var_glat, var,
                         cbar_range, cmap, cbar_units, cbar_extend)
  cs1 = m.contourf(var_glon, var_glat, pval_var,
                   levels=[0., 0.1], colors='none', 
                   hatches=['////', None], zorder=3, latlon=True)
  cs2 = m.contour(var_glon, var_glat, pval_var,
                  levels=[0., 0.1], colors='k',
                  zorder=3, linewidths=.8, latlon=True)

  # location of S1 and Holland box
  #m.scatter(s1_lon, s1_lat,
  #          marker='*', s=60, 
  #          latlon=True,
  #          c='gold', edgecolor='k', 
  #          lw=.5, zorder=7)
  #bathymetry contours
  lp = m.contour(tglon, tglat, topo.elevation,
            levels=[-4000, -1000, -500],
            colors=['slategray', 'mediumvioletred', 'k'],
            latlon=True, zorder=2)
  lp_labels = ['4000 m', '1000 m']
  for i in range(len(lp_labels)):
      lp.collections[i].set_label(lp_labels[i])
  ax.legend(loc=2, fontsize=9)

  return fig, ax , m

  #-------------------------------------------------------------------

# Directories
voldir = '/Volumes/SamT5/PhD/data/'
bindir = voldir + 'altimetry_cpom/2_grid_offset/'
griddir = voldir + 'altimetry_cpom/3_grid/'
topodir = voldir + 'topog/'
icedir = voldir + 'NSIDC/sic/'
eradir = voldir + 'reanalyses/'
lmdir = voldir + 'land_masks/'
coastdir = lmdir + 'holland_vic/'
modeldir = voldir + 'moorings/model_pahol/'
idxdir = voldir + 'climate_indices/'

localdir = '/Users/ocd1n16/PhD_local/'
scriptdir = localdir + 'scripts/'

figdir = localdir + 'data_notes/Figures_v8/'

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
time_units='days since 0001-01-01 00:00:00.0'

# bathymetry file
with xr.open_dataset(topodir + 'coarse_gebco_p5x1_latlon.nc') as topo:
    print(topo.keys())
tglat, tglon = np.meshgrid(topo.lat, topo.lon)
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

altim_start = '2002-10-01'
altim_end = '2018-10-31'

# -----------------------------------------------------------------
#       ~ ~ ~     ALTIMETRY DATA      ~ ~ ~
# -----------------------------------------------------------------
altfile = 'dot_all_30bmedian.nc'

with xr.open_dataset(griddir+altfile) as alt:
    print(alt.keys())

# GRID coordinates
# at bin edges
alt_eglat, alt_eglon = np.meshgrid(alt.edge_lat, alt.edge_lon)
# at bin centres
alt_glat, alt_glon = np.meshgrid(alt.latitude, alt.longitude)
alt_gmlat, alt_gmlon = np.meshgrid(alt.mlat, alt.mlon)

#-----------------------------------
# DE-TREND
#-----------------------------------
print("\n>>> Computing linear trend at every grid point ..")
alt_crop = alt.sel(time=slice(altim_start, altim_end))
dot = alt.dot.sel(time=slice(altim_start, altim_end))

dot_date = dot.time.dt.strftime("%m/%Y").values
alt_ndays = mdates.date2num(list(dot.time.values))
alt_dt = alt_ndays - alt_ndays[0]
# DIMENSIONS
alondim, alatdim, atimdim = dot.shape


# compute linear trend at every grid point
dot_interc, dot_slope, dot_ci, dot_pval = [ma.zeros((alt_glat.shape)) for _ in range(4)]
for r in range(alondim):
    for c in range(alatdim):
        arr_trend = ft.trend_ci(alt_ndays, dot.values[r,c,:], 0.95)
        dot_interc[r, c] = arr_trend.intercept.values
        dot_slope[r, c] = arr_trend.slope.values
        dot_pval[r, c] = arr_trend.p_val.values

# trend in mm/yr without GMSLR 
dot_slope_mm_yr = dot_slope * 1e3 * 365.25 - 3.2

# apply a mask to plot only trend values where p_val < 0.1
dot_slope_mm_yr[dot_pval>0.1] = np.nan

# ** subtract the global mean sea level   
gmslr = alt_dt * 0.0032/365.25

# detrend the time series at every point
trend_dot = dot_slope[:,:,np.newaxis] * alt_ndays[np.newaxis, np.newaxis, :] + dot_interc[:,:,np.newaxis]
dot_det = dot - trend_dot - gmslr

# - - - - - - - - - - - - - - - - -
print("Computing gvel .. ")
dx = gsw.distance(alt_glon, alt_glat, axis=0)
dy = gsw.distance(alt_glon, alt_glat, axis=1)

dsla_x = dot_det.values[1:] - dot_det.values[:-1]
dsla_y = dot_det.values[:, 1:, :] - dot_det.values[:, :-1, :]

# take differences along one axis ..
dsla_dx = dsla_x/dx[:, :, np.newaxis]
dsla_dy = dsla_y/dy[:, :, np.newaxis]

# average along the other direction
dsla_dx_mid = 0.5*(dsla_dx[:, 1:, :]+dsla_dx[:, :-1, :])
dsla_dy_mid = 0.5*(dsla_dy[1:, :, :]+dsla_dy[:-1, :, :])

# scale by f and g
f = gsw.f(alt_gmlat)
g =  gsw.grav(alt_gmlat, 0)

vg = dsla_dx_mid * (g/f)[:, :, np.newaxis]
ug = (-1) * dsla_dy_mid * (g/f)[:, :, np.newaxis]

# - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - 
# crop to overlap period
altim = xr.Dataset({'dot' : (('lon', 'lat', 'time'), dot.values),
  'dot_det' : (('lon', 'lat', 'time'), dot_det),
  'ug' : (('mlon', 'mlat', 'time'), alt_crop.ug.values),
  'vg' : (('mlon', 'mlat', 'time'), alt_crop.vg.values),
  'ug_det' : (('mlon', 'mlat', 'time'), ug),
  'vg_det' : (('mlon', 'mlat', 'time'), vg)},
  coords={'time' : dot.time.values,
  'lon' : dot.longitude.values,
  'lat' : dot.latitude.values,
  'mlat' : alt.mlat.values,
  'mlon' : alt.mlon.values})

# - - - - -  crop data 
#----------------------------------------------
crop_start = '2010-12-01'
crop_end = '2015-12-01'
#----------------------------------------------

# multi-year anomalies
alt_anom = altim - altim.mean('time')
alt_anom_crop = alt_anom.sel(time=slice(crop_start, crop_end))

# remove seasonal cycle
alt_mclim = altim.groupby('time.month') - altim.groupby('time.month').mean('time')
alt_mclim_crop = alt_mclim.sel(time=slice(crop_start, crop_end))

# - - - - - - - - - - - - - - - - - 
altim_time = altim.time.dt.strftime('%m.%Y').values
altim_period = altim_time[0] + '-' + altim_time[-1]


from eofs.xarray import Eof

from mpl_toolkits.basemap import Basemap 
m = Basemap(projection='spstere',
  boundinglat=-50,
  resolution='i',
  lon_0=180,
  round=True)

# - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - 
# variables for EOF part
# - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - 
alt_var = dot_det

glat, glon = np.meshgrid(alt_var.latitude.values, alt_var.longitude.values)
gmlat = 0.5*(glat[1:, 1:] + glat[:-1, :-1])
gmlon = 0.5*(glon[1:, 1:] + glon[:-1, :-1])

# apply weighting to account for meridians converging at high lat
coslat = np.cos(np.deg2rad(alt_var.latitude.values)).clip(0., 1.)
wgts = np.sqrt(coslat)[..., np.newaxis]
solver = Eof(alt_var.T, weights=wgts)

#eof = solver.eofsAsCorrelation(neofs=15)
eof = solver.eofs(neofs=15)
pc = solver.pcs(npcs=15, pcscaling=1)
variance_frac = solver.varianceFraction()

# > > reconstruct DOT using the first 5 (or 10) modes
nm = 5
rec_dot = solver.reconstructedField(nm)

# remove seasonal cycle 
rec_seas_sla = rec_dot.groupby("time.month") - rec_dot.groupby("time.month").mean()
rec_seas_ssla = rec_seas_sla / rec_seas_sla.std("time", ddof=1)
rec_seas_ssla_s1 = rec_seas_ssla.sel(time=slice(crop_start, crop_end))

# time-mean
rec_sla = rec_dot - rec_dot.mean("time")
rec_ssla = rec_sla / rec_sla.std('time', ddof=1)
rec_ssla_s1 = rec_ssla.sel(time=slice(crop_start, crop_end))

rec_time = rec_ssla_s1.time.dt.strftime('%m.%Y').values
corr_period = rec_time[0] + '-' + rec_time[-1]

# - - - - - - - - - - - - - - - - - 
print("Computing gvel from rec DOT .. ")
# - - - - - - - - - - - - - - - - - 
dx = gsw.distance(glon, glat, axis=0)
dy = gsw.distance(glon, glat, axis=1)

dsla_x = rec_dot.T.values[1:] - rec_dot.T.values[:-1]
dsla_y = rec_dot.T.values[:, 1:, :] - rec_dot.T.values[:, :-1, :]

# take differences along one axis ..
dsla_dx = dsla_x/dx[:, :, np.newaxis]
dsla_dy = dsla_y/dy[:, :, np.newaxis]

# average along the other direction
dsla_dx_mid = 0.5*(dsla_dx[:, 1:, :]+dsla_dx[:, :-1, :])
dsla_dy_mid = 0.5*(dsla_dy[1:, :, :]+dsla_dy[:-1, :, :])

# scale by f and g
f = gsw.f(gmlat)
g =  gsw.grav(gmlat, 0)

vg_rec = dsla_dx_mid * (g/f)[:, :, np.newaxis]
ug_rec = (-1) * dsla_dy_mid * (g/f)[:, :, np.newaxis]

# - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - 
# for regional EOF
rec = xr.Dataset({'dot' : (('lon', 'lat', 'time'), rec_dot.T.values),
                  'ug' : (('mlon', 'mlat', 'time'), ug_rec),
                  'vg' : (('mlon', 'mlat', 'time'), vg_rec)},
                  coords={'lon' : rec_dot.longitude.values,
                          'lat' : rec_dot.latitude.values,
                          'mlon' : gmlon[:, 0],
                          'mlat' : gmlat[0,:],
                          'time' : rec_dot.time.values})
rec_anom = rec - rec.mean('time')
rec_anom_crop = rec_anom.sel(time=slice(crop_start, crop_end))

rec_mclim = rec.groupby('time.month') - rec.groupby('time.month').mean('time')
rec_mclim_crop = rec_mclim.sel(time=slice(crop_start, crop_end))

#----------------------------------------------
#----------------------------------------------
# cLIMATE INDICES
#----------------------------------------------
#----------------------------------------------
with xr.open_dataset(idxdir + 'climate_indices_79_19.nc') as clim_idx:
  print(clim_idx)

indx = clim_idx.sel(time=slice(crop_start, crop_end))

# - - - - - - - - 
# standardise data (remove mean, divide by std; mean=0, std=1)
indx_anom = indx - indx.mean()
indx_stand = indx_anom / indx.std(ddof=1)

indx_mclim = indx.groupby('time.month') - indx.groupby('time.month').mean('time')
indx_mclim_stand = indx_mclim / indx_mclim.std('time', ddof=1)

xtim = indx.time.values


# - - - - - - - - - - - - - - - - - 
#           PLOTS
# - - - - - - - - - - - - - - - - - 
# with seasonal cycle
corr, pval = r_map_ts(rec_anom_crop.dot.values,
                      indx_stand.nino4.values)
fig, ax, m = plot_corr(corr, pval, alt_glon, alt_glat)

fig.suptitle("Rmap of multi-year SLA (3 EOF) vs Nino 4 \n"
 "[period: %s]" % (corr_period))

#fig.savefig(figdir+'rmap_sla_mooring_3a_stand.png', dpi=fig.dpi)

# without seasonal cycle
corr, pval = r_map_ts(rec_mclim_crop.dot.values,
                      indx_mclim_stand.nino4.values)
fig, ax, m = plot_corr(corr, pval, alt_glon, alt_glat)

fig.suptitle("Rmap of de-seasonalised SLA (3 EOF) vs Nino 4 \n"
 "[period: %s]" % (corr_period))
