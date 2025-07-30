"""
This is the function aux_altimetry.detrend_fc()

"""

import numpy as np
from numpy import ma

import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.dates as mdates
from palettable.scientific.diverging import Vik_20

import analysis_functions.aux_func as fc
import analysis_function.aux_stereoplot as st
import analysis_functions.aux_altimetry as altim

import sys

plt.ion()
#-------------------------------------------------------------------
# Directories
#-------------------------------------------------------------------
voldir = '/Volumes/SamT5/PhD/data/'
griddir = voldir + 'altimetry_cpom/3_grid_dot/'
figdir = voldir + '../PhD_figures/ch2_altimetry/'
#-------------------------------------------------------------------
# time window
date_start = '2010-02-01'
date_end = '2016-01-01'

#-------------------------------------------------------------------
# ALTIMETRY data
geoidtype = 'goco05c'#'eigen6s4v2' # 'goco05c', 'egm08'
satellite = 'all'
sigma = 3

altfile = 'dot_' + satellite + '_30bmedian_' + geoidtype + '_sig' + str(sigma) + '.nc'
with xr.open_dataset(griddir+altfile) as alt:
  print(alt.keys())

print("Crop altimetry to \n\n > > %s - %s\n\n" % (date_start, date_end))

alt_crop = alt.sel(time=slice(date_start, date_end))
dot = alt_crop.dot
lat = alt_crop.latitude.values
lon = alt_crop.longitude.values

#-------------------------------------------------------------------
# GRID coordinates
eglat, eglon = np.meshgrid(alt_crop.edge_lat, alt_crop.edge_lon)

#-------------------------------------------------------------------
#-------------------------------------------------------------------
# linear trend calculation
#-------------------------------------------------------------------
#-------------------------------------------------------------------
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


cbar_range = [-10, 10]
cmap = cm.get_cmap(Vik_20.mpl_colormap, 21)
cbar_units = ('Linear trend (mm/yr) where p-val<0.1 (%s-%s)' % (date[0], date[-1]))
fig, ax, m = st.spstere_plot(eglon, eglat, slope_mm_yr,
cbar_range, cmap, cbar_units, 'm')


fig.savefig(figdir + "linear_trend.pdf", dpi=1200)
