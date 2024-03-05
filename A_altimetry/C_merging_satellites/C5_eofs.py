"""
Detrend DOT 
Reconstruct DOT from the first 5/6 EOFs
Compute ug

Plot EOFs and PC

Last modified: 9 Nov 2021
"""

import numpy as np
from numpy import ma

import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from palettable.cmocean.diverging import Balance_19

import sys

import gsw

plt.ion()
#-------------------------------------------------------------------
# Directories
#-------------------------------------------------------------------
voldir = '/Volumes/SamT5/PhD/data/'
griddir = voldir + 'altimetry_cpom/3_grid_dot/'
topodir = voldir + 'topog/'

localdir = '/Users/ocd1n16/PhD_local/'
figdir = localdir + 'data_figures/Figures_nov21/'

auxscriptdir = localdir + 'scripts/aux_func/'
sys.path.append(auxscriptdir)
import aux_func_trend as fc
import aux_stereoplot as st
import aux_corr_maps as rmap

# function that extracts altimetry data
sys.path.append(localdir + 'scripts/D_validation_and_analysis/')
import d2_extract_altim_anom as altim

#-------------------------------------------------------------------
# bathymetry file
#-------------------------------------------------------------------
with xr.open_dataset(topodir + 'coarse_gebco_p5x1_latlon.nc') as topo:
    print(topo.keys())
tglat, tglon = np.meshgrid(topo.lat, topo.lon)

#------------------------------------
# limits for mooring/altimetry data
#------------------------------------
altim_start = '2002-07-01'
altim_end = '2018-10-15'

# -----------------------------------------------------------------
#            ~ ~ ~     ALTIMETRY DATA      ~ ~ ~
#
# using file < dot_satellite_30bmedian_geoidtype.nc > where:
# satellite = [all, cs2, env] 
# geoidtype = [egm08, goco05]
# -----------------------------------------------------------------
geoidtype = 'goco05c' # 'goco05c'
satellite = 'all'
sigma = 3

# extract dot and ug over the period selected
alt = altim.extract_dot_maps(geoidtype, satellite, sigma, altim_start, altim_end)

#-----------------------------------
# EOF (eofs)
#-----------------------------------
from eofs.xarray import Eof

# - - - - - - - - - - - - - - - - - 
# variables for EOF part
# - - - - - - - - - - - - - - - - - 
alt_var = alt.dot_det

# - - - - - - - - - - - - - - - - -
glat, glon = np.meshgrid(alt_var.latitude.values, alt_var.longitude.values)
gmlat = 0.5*(glat[1:, 1:] + glat[:-1, :-1])
gmlon = 0.5*(glon[1:, 1:] + glon[:-1, :-1])

# apply weighting to account for meridians converging at high lat
coslat = np.cos(np.deg2rad(alt_var.latitude.values)).clip(0., 1.)
wgts = np.sqrt(coslat)[..., np.newaxis]
solver = Eof(alt_var.T, weights=wgts)

#eof = solver.eofsAsCorrelation(neofs=15)
eof = solver.eofsAsCovariance(neofs=10)
pc = solver.pcs(npcs=10, pcscaling=1)
variance_frac = solver.varianceFraction()

# - - - - - - - - - - - - - - - - - 
a = np.cumsum(variance_frac.values)

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
# >> circumpolar plots of EOF patterns
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
	
  #fig.savefig(figdir+'eof%s_jan2011_det2015_DOT.png' % str(k+1))

k = 0
fig, axs = plt.subplots(nrows=5, figsize=(5,5))
[axs[i].plot(pc.time.values, pc[:, k+i], c='k', label='PC%s'%str(i+1)) for i in range(5)]
for ax in axs[:]:
  ax.legend()
plt.tight_layout() 
#fig.savefig(figdir+'pc_det_DOT_cs2_6.png', dpi=fig.dpi)


# > > reconstruct DOT using the first 5 (or 10) modes
nm = 3
rec_dot = solver.reconstructedField(nm)

# - - - - - - - - - - - - - - - - - 
def geos_vel(var):
    
    glat, glon = np.meshgrid(var.lat.values, var.lon.values)
    gmlat = 0.5*(glat[1:, 1:] + glat[:-1, :-1])
    gmlon = 0.5*(glon[1:, 1:] + glon[:-1, :-1])

    dx = gsw.distance(glon, glat, axis=0)
    dy = gsw.distance(glon, glat, axis=1)

    dsla_x = var.values[1:] - var.values[:-1]
    dsla_y = var.values[:, 1:, :] - var.values[:, :-1, :]

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

    return ug_rec, vg_rec, gmlon, gmlat
# - - - - - - - - - - - - - - - - - 
ug_rec, vg_rec, gmlon, gmlat = geos_vel(rec_dot.T)


# - - - - - - - - - - - - - - - - - 
rec = xr.Dataset({'dot' : (('lon', 'lat', 'time'), rec_dot.T.values),
                  'ug' : (('mlon', 'mlat', 'time'), ug_rec),
                  'vg' : (('mlon', 'mlat', 'time'), vg_rec)},
                  coords={'lon' : rec_dot.lon.values,
                          'lat' : rec_dot.lat.values,
                          'mlon' : gmlon[:, 0],
                          'mlat' : gmlat[0,:],
                          'time' : rec_dot.time.values})

rec_anom = rec - rec.mean('time')
rec_mclim = rec.groupby('time.month') - rec.groupby('time.month').mean('time')

# seasonal cycle
alt_mclim = rec.dot.groupby('time.month').mean('time')

cbar_range = [-4, 4]
cmap = Balance_19.mpl_colormap
cbar_units = "SLA (cm)"

for k in range(12):
  fig, ax, m = st.spstere_plot(glon, glat,
  	alt_mclim.sel(month=k+1).values*1e2,
  	cbar_range, cmap, cbar_units, 'both')

  lp = m.contour(tglon, tglat, topo.elevation,
          levels=[-1000],
          colors=['darkslategray'], linestyles='-',
          latlon=True, zorder=2)
  lp_labels = ['1000 m']
  for i in range(len(lp_labels)):
    lp.collections[i].set_label(lp_labels[i])
  ax.legend(loc='upper left', fontsize=9, bbox_to_anchor=(-0.08, 0.9))
  ax.annotate(str(alt_mclim.month[k].values),
              xy=(.5, 0.55),
              xycoords='figure fraction',
              ha='left', va='bottom',
              weight='bold', fontsize=20)

