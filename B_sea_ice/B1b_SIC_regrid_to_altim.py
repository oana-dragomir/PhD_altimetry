"""
Bin SIC data from 25x25km to 0.75x0.75 deg (ERA)
and then interpolate to SSH grid (0.5 lat x 1 lon)

save in a file

Last modified: 10 Mar 2025
"""

import numpy as np
import numpy.ma as ma

import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from scipy.interpolate import griddata

import sys
import os

# Directories
workdir = '/Volumes/SamT5/PhD_data/'
icedir = workdir + 'NSIDC/sic/'
topodir = workdir + 'topog/'
altdir = workdir + 'altimetry_cpom/3_grid_dot/'
winddir = workdir + 'reanalyses/'

# import my functions
scriptdir = '/Volumes/SamT5/PhD_scripts/'
auxscriptdir = scriptdir + 'aux_func/'
sys.path.append(auxscriptdir)
import aux_stereoplot as st

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
#				SIC data
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
with xr.open_dataset(icedir + 'sic0_raw.nc') as ice:
    print(ice.keys())
ice_lat = ice.lat.values
ice_lon = ice.lon.values
ice_sic = ma.masked_invalid(ice.sic.values)

cbar_range = [0., 1.]
cmap = cm.get_cmap('cividis_r', 10)
cbar_units = 'SIC'
cbar_extend = 'neither'

# - - -  PLOT 	- - - - 
plt.ion()
k = 169
fig, ax, m = st.spstere_plot(ice_lon, ice_lat, ice.sic[:,:,k],
                       cbar_range, cmap, 
                       cbar_units, cbar_extend)
lp = m.contour(ice_lon, ice_lat, ice.sic[:,:,k],
			levels=[.75], colors='crimson',
			linewidths=1., latlon=True)
lp_label = '75%'
lp.collections[0].set_label(lp_label)
ax.legend(loc='lower right', 
	bbox_to_anchor=(.15, .85), 
	fontsize=12)
tim = ice.time.dt.strftime("%Y/%m").values[k]
ax.annotate("%s" % tim, 
  		xycoords='axes fraction', 
  		xy=(.45, .5), 
  		bbox=dict(facecolor='lavender',
        		  edgecolor='lavender',
            	  boxstyle='round'))
# - - - - - - - - - - - - 

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# 				ERA Interim data
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
with xr.open_dataset(winddir + 'eraint_1979_2018.nc') as wind:
    print(wind.keys())

# grid
wind_glat, wind_glon = np.meshgrid(wind.latitude, 
                                   wind.longitude)

dtim = ice.time.size
dwlon, dwlat = wind_glat.shape

# interpolate ice grid onto wind grid
wgrid_sic = ma.zeros((dwlon, dwlat, dtim))
for k in range(dtim):
	wgrid_sic[:, :, k] = griddata((ice_lon.flatten(), ice_lat.flatten()), 
	                          ice_sic[:,:,k].flatten(),
	                          (wind_glon, wind_glat), method='nearest')

#------------------------------------------------------------------
# - - -  PLOT 	- - - - 
plt.ion()
k = 169
fig, ax, m = st.spstere_plot(wind_glon, wind_glat,
					   wgrid_sic[:,:,k],
                       cbar_range, cmap, 
                       cbar_units, cbar_extend)
lp = m.contour(wind_glon, wind_glat, wgrid_sic[:,:,k], levels=[.15], 
          colors='crimson', linewidths=1., 
          latlon=True)
lp_label = '15%'
lp.collections[0].set_label(lp_label)
ax.legend(loc='lower right',
		bbox_to_anchor=(.15, .85),
		fontsize=12)
tim = ice.time.dt.strftime("%Y/%m").values[k]
ax.annotate("%s" % tim, 
    		xycoords='axes fraction', xy=(.45, .5), 
    		bbox=dict(facecolor='lavender',
            		  edgecolor='lavender',
            		  boxstyle='round'))

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
#				Altimetry data
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
#-------------------------------------------------------------------
# altimetry file 
#-------------------------------------------------------------------
geoidtype = 'goco05c'#'eigen6s4v2' # 'goco05c', 'egm08'
satellite = 'all'
sigma = 3

altfile = 'dot_' + satellite + '_30bmedian_' + geoidtype + '_sig' + str(sigma) + '.nc'

#-------------------------------------------------------------------
# load altimetry file
with xr.open_dataset(altdir+altfile) as alt:
    print(alt.keys())

# GRID coordinates
# at bin edges
alt_eglat, alt_eglon = np.meshgrid(alt.edge_lat.values,
								 alt.edge_lon.values)
# at bin centres
alt_glat, alt_glon = np.meshgrid(alt.latitude.values,
								 alt.longitude.values)


# interpolate ice grid onto altimetry grid
agrid_sic = ma.zeros((alt.dot.shape))
for k in range(dtim):
	agrid_sic[:, :, k] = griddata((wind_glon.flatten(), wind_glat.flatten()), 
	                          wgrid_sic[:,:,k].flatten(),
	                          (alt_glon, alt_glat), method='linear')

# - - -  PLOT 	- - - - 
plt.ion()
k = 169
fig, ax, m = st.spstere_plot(alt_eglon, alt_eglat,
					   agrid_sic[:,:,k],
                       cbar_range, cmap, 
                       cbar_units, cbar_extend)
lp = m.contour(alt_glon, alt_glat, agrid_sic[:,:,k], levels=[.15], 
          colors='crimson', linewidths=1., 
          latlon=True)
lp_label = '15%'
lp.collections[0].set_label(lp_label)
ax.legend(loc='lower right',
		bbox_to_anchor=(.15, .85),
		fontsize=12)
tim = ice.time.dt.strftime("%Y/%m").values[k]
ax.annotate("%s" % tim, 
    		xycoords='axes fraction', xy=(.45, .5), 
    		bbox=dict(facecolor='lavender',
            		  edgecolor='lavender',
            		  boxstyle='round'))
# - - - - - - - - - - - - 

ds = xr.Dataset({'sic' : (('lon', 'lat', 'time'), agrid_sic)},
	coords={'lon': alt.longitude.values,
	'lat': alt.latitude.values,
	'time': alt.time.values})

newfilename = 'sic_wgrid_p5latx1lon.nc'
ds.to_netcdf(icedir + newfilename)

