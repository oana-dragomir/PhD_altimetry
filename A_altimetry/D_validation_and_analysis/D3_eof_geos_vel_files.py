
import numpy as np
from numpy import ma

import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.dates as mdates

from palettable.cmocean.diverging import Balance_19

import aux_stereoplot as st 
import aux_func_trend as ft

import gsw

#-------------------------------------------------------------------
# Directories
voldir = '/Volumes/SamT5/PhD/data/'
griddir = voldir + 'altimetry_cpom/3_grid/'
topodir = voldir + 'topog/'

localdir = '/Users/ocd1n16/PhD_local/'
scriptdir = localdir + 'scripts/'

figdir = localdir + 'data_notes/Figures_jan21/'

# bathymetry file
with xr.open_dataset(topodir + 'coarse_gebco_p5x1_latlon.nc') as topo:
    print(topo.keys())
tglat, tglon = np.meshgrid(topo.lat, topo.lon)
#-----------------------------------
# altimetry file
altfile = 'dot_all_30bmedian.nc'
altfile = 'dot_env_30bmedian.nc'

with xr.open_dataset(griddir+altfile) as alt:
    print(alt.keys())
#-----------------------------------
# crop dataset in time
altim_start = '2010-11-01'
altim_end = '2018-10-31'

alt_crop = alt.sel(time=slice(altim_start, altim_end))

# make GRIDS 
#-----------------------------------
# at bin edges
alt_eglat, alt_eglon = np.meshgrid(alt_crop.edge_lat, alt_crop.edge_lon)
# at bin centres for SLA (glat) and velocity data (gmlat)
alt_glat, alt_glon = np.meshgrid(alt_crop.latitude, alt_crop.longitude)

alt_gmlat, alt_gmlon = np.meshgrid(alt_crop.mlat, alt_crop.mlon)

# for individual satellites
mlat = 0.5*(alt_crop.latitude.values[:-1] + alt_crop.latitude.values[1:])
mlon = 0.5*(alt_crop.longitude.values[:-1] + alt_crop.longitude.values[1:])
alt_gmlat, alt_gmlon = np.meshgrid(mlat, mlon)
#-------------------------------------------------------------------
# DE-TREND sea level
#-------------------------------------------------------------------
print("\n>>> Computing linear trend at every grid point ..")

dot = alt_crop.dot

dot_date = dot.time.dt.strftime("%m/%Y").values
dot_ndays = mdates.date2num(list(dot.time.values))
alt_dt = dot_ndays - dot_ndays[0]

# DIMENSIONS
alondim, alatdim, atimdim = dot.shape

# compute linear trend at every grid point
dot_interc, dot_slope, dot_ci, dot_pval = [ma.zeros((alt_glat.shape)) for _ in range(4)]
for r in range(alondim):
    for c in range(alatdim):
        arr_trend = ft.trend_ci(dot_ndays, dot.values[r,c,:], 0.95)
        dot_interc[r, c] = arr_trend.intercept.values
        dot_slope[r, c] = arr_trend.slope.values
        dot_pval[r, c] = arr_trend.p_val.values

# trend in mm/yr including the GMSLR 
dot_slope_mm_yr = dot_slope * 1e3 * 365.25

# apply a mask to plot only trend values where p_val < 0.1
dot_slope_mm_yr[dot_pval>0.1] = np.nan

cbar_range = [-10, 10]
cmap = cm.get_cmap('seismic', 41)
cbar_units = 'Trend anomaly (mm/yr)'
fig, ax, m = st.spstere_plot(alt_eglon, alt_eglat, dot_slope_mm_yr,
  cbar_range, cmap, cbar_units, 'both')

fig.suptitle("Linear trend in DOT [where p_val < 0.1] %s-%s" 
  % (dot_date[0], dot_date[-1]))

# ** subtract the global mean sea level   
#gmslr = alt_dt * 0.0032/365.25

# detrend the time series at every point
trend_dot = dot_slope[:,:,np.newaxis] * dot_ndays[np.newaxis, np.newaxis, :] + dot_interc[:,:,np.newaxis]
dot_det = alt_crop.dot - trend_dot #- gmslr

#-----------------------------------
# geostrophic velocities
#-----------------------------------
print("Computing gvel .. ")
# distance in m between grid points
dx = gsw.distance(alt_glon, alt_glat, axis=0)
dy = gsw.distance(alt_glon, alt_glat, axis=1)

# change in SLA between adjacent points along x and y dir
dsla_x = dot.values[1:] - dot.values[:-1]
dsla_y = dot.values[:, 1:, :] - dot.values[:, :-1, :]

# (spatial derivatives) take differences along each axis ..
dsla_dx = dsla_x/dx[:, :, np.newaxis]
dsla_dy = dsla_y/dy[:, :, np.newaxis]

# average along the other direction to match dimensions
dsla_dx_mid = 0.5*(dsla_dx[:, 1:, :]+dsla_dx[:, :-1, :])
dsla_dy_mid = 0.5*(dsla_dy[1:, :, :]+dsla_dy[:-1, :, :])

# scale by f (1/s) and g (m/s^2)
f = gsw.f(alt_gmlat)
g =  gsw.grav(alt_gmlat, 0)

vg = dsla_dx_mid * (g/f)[:, :, np.newaxis]
ug = (-1) * dsla_dy_mid * (g/f)[:, :, np.newaxis]

# - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - 
# save data
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

altim = xr.Dataset({'dot' : (('lon', 'lat', 'time'), dot.values),
  'ug' : (('mlon', 'mlat', 'time'), ug),
  'vg' : (('mlon', 'mlat', 'time'), vg)},
  coords={'time' : dot.time.values,
  'lon' : dot.longitude.values,
  'lat' : dot.latitude.values,
  'mlat' : mlat,
  'mlon' : mlon})
altim.to_netcdf(griddir + 'dot_vel_cs2_30bmedian.nc')

#-----------------------------------
# EOF 
#-----------------------------------
from eofs.xarray import Eof

# - - - - - - - - - - - - - - - - - 
# variables for EOF part 
# - - - - - - - - - - - - - - - - - 
alt_var = alt.dot

alt_var_time = alt_var.time.dt.strftime('%m.%Y').values
alt_var_period = alt_var_time[0] + '-' + alt_var_time[-1]

glat, glon = np.meshgrid(alt_var.latitude.values, alt_var.longitude.values)
gmlat = 0.5*(glat[1:, 1:] + glat[:-1, :-1])
gmlon = 0.5*(glon[1:, 1:] + glon[:-1, :-1])
# - - - - - - - - - - - - - - - - -
# apply weighting to account for meridians converging at high lat
coslat = np.cos(np.deg2rad(alt_var.latitude.values)).clip(0., 1.)
wgts = np.sqrt(coslat)[..., np.newaxis]
solver = Eof(alt_var.T, weights=wgts)

#eof = solver.eofsAsCorrelation(neofs=15)
eof = solver.eofsAsCovariance(neofs=10)
pc = solver.pcs(npcs=10, pcscaling=1)
variance_frac = solver.varianceFraction()

# - - - - - - - - - - - - - - - - -
# plot of cumulative variance for every mode 
# - - - - - - - - - - - - - - - - -
a = np.cumsum(variance_frac.values)
plt.ion()
fig, ax = plt.subplots()
ax.scatter(variance_frac.mode.values+1, a, s=7, c='k')
ax.axhline(a[9], ls=':', c='grey')
ax.axvline(10, ls=':', c='grey')
ax.annotate('{}{}'.format(str((a[9]*100).round(2)), '%'), 
  xy=(50, a[9]+0.01), xycoords='data')

ax.axhline(a[4], ls=':', c='grey')
ax.axvline(5, ls=':', c='grey')
ax.annotate('{}{}'.format(str((a[4]*100).round(2)), '%'), 
  xy=(50, a[4]+0.01), xycoords='data')
ax.set_title("Cumulative variance explained by each mode")
plt.tight_layout()

# - - - - - - - - - - - - - - - - - 
# >> circumpolar plots of EOFs
# - - - - - - - - - - - - - - - - -
cbar_range = [-4, 4]
cmap = Balance_19.mpl_colormap

for k in range(5, -1, -1):
  cbar_units = "EOF %s (m/std)" % str(k+1)
  std_k = eof[k].std(ddof=1)

  fig, ax, m = st.spstere_plot(glon, glat, eof[k].values.T/std_k.values,
  cbar_range, cmap, cbar_units, 'both')

  lp = m.contour(tglon, tglat, topo.elevation,
          levels=[-1000],
          colors=['darkslategray'], linestyles='-',
          latlon=True, zorder=2)
  lp_labels = ['1000 m']
  for i in range(len(lp_labels)):
    lp.collections[i].set_label(lp_labels[i])
  ax.legend(loc='upper left', fontsize=9, bbox_to_anchor=(-0.08, 0.9))
  ax.annotate("{:.1%}".format(variance_frac[k].values),
              xy=(.45, 0.5),
              xycoords='figure fraction',
              ha='left', va='bottom',
              weight='bold', fontsize=20)
  
  #fig.savefig(figdir+'eof%s_jan2011_dec2015_DOTdet.png' % str(k+1))

r, p = [np.zeros(10,) for _ in range(2)]
for i in range(10):
  r[i], p[i] = pearsonr(pc[:, i].sel(time=slice(date_start, date_end)).values,
   uc.det_anom_stand.values)

# - - - - - - - - - - - - - - - - - 
# time series of PC
# - - - - - - - - - - - - - - - - - 

k = 0
fig, axs = plt.subplots(nrows=5, figsize=(8,9))
[axs[i].plot(pc.time.values, pc[:, k+i], c='k', label='PC%s'%str(i+k+1)) for i in range(5)]
for ax in axs[:]:
  # insert here standardised time series to be compared with PCs
  ax.plot(uc.time.values, uc.det_anom_stand, c='c')
  ax.legend()
[axs[i].annotate('r: %.4f, p: %.4f' % (r[i+k], p[i+k]),
  xycoords='axes fraction', xy=(.7, .98), 
  bbox=dict(alpha=1, fc='whitesmoke', ec='silver')) for i in range(5)]
ax.set_title('UC anom stand')
plt.tight_layout() 

fig.savefig(figdir+'pc_nov02_oct18_DOTdet.png', dpi=fig.dpi)


# - - - - - - - - - - - - - - - - - 
# reconstruct DOT
# - - - - - - - - - - - - - - - - - 
nm = 5 # number of modes to use
rec_dot = solver.reconstructedField(nm)

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
rec.dot.attrs['long_name'] = 'dot_reconstructed_from_5eof'
rec.to_netcdf(griddir + 'dot_env_30bmedian_5eof.nc')
rec.to_netcdf(griddir + 'dot_all_30bmedian_5eof_nov2002_oct2018.nc')
