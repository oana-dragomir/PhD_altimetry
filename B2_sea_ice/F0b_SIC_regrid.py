"""
Bin SIC data from 25x25km to 0.75x0.75 deg (ERA)
and then interpolate to SSH grid (0.5 lat x 1 lon)

save in a file

Last modified: 3 Apr 2020
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
maindir = '/Volumes/SamT5/PhD/'
workdir = maindir + 'data/'
icedir = workdir + 'NSIDC/sic/'
topodir = workdir + 'topog/'
altdir = workdir + 'altimetry_cpom/3_grid_dot/'
winddir = workdir + 'reanalysis/'
sicfigdir = workdir + '../data_notes/Figures_SIC/'

# import my functions
fcdir = maindir + 'scripts/'
sys.path.append(fcdir)
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
#				Altimetry data
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
altfile = 'dot_all_30bmedian.nc'
with xr.open_dataset(altdir + altfile) as alt:
    print(alt.keys())

# GRID coordinates
# at bin edges
alt_eglat, alt_eglon = np.meshgrid(alt.edge_lat.values,
								 alt.edge_lon.values)
# at bin centres
alt_glat, alt_glon = np.meshgrid(alt.latitude.values,
								 alt.longitude.values)


# interpolate ice grid onto altimetry grid
itp_sic = ma.zeros((alt.dot.shape))
for k in range(len(ice.time)):
	itp_sic[:, :, k] = griddata((ice_lon.flatten(), ice_lat.flatten()), 
	                          ice_sic[:,:,k].flatten(),
	                          (alt_glon, alt_glat), method='linear')
itp_sic = ma.masked_invalid(itp_sic)

# - - -  PLOT 	- - - - 
plt.ion()
k = 6
fig, ax, m = st.spstere_plot(alt_eglon, alt_eglat,
					   itp_sic[:,:,k],
                       cbar_range, cmap, 
                       cbar_units, cbar_extend)
lp = m.contour(alt_glon, alt_glat, itp_sic[:,:,k], levels=[.15], 
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

ds = xr.Dataset({'sic' : (('lon', 'lat', 'time'), itp_sic)},
	coords={'lon': alt.longitude.values,
	'lat': alt.latitude.values,
	'time': alt.time.values})

newfilename = 'sic_grid_p5latx1lon.nc'
ds.to_netcdf(icedir + newfilename)

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Weddell [-60, 40] SIC on the altimetry grid
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# crop to weddell lon [-40, 70]
with xr.open_dataset(icedir + newfilename) as ice0:
  print(ice0.keys)

ice = ice0.sel(time=slice('2002-07-01', '2018-10-01'))

# extract lat, lon, and sic
ice_lat = ice.lat.values
ice_lon = ice.lon.values
ice_sic = ice.sic.values

# dimensions
londim, latdim, timdim = ice_sic.shape

# crop to Weddell region
sic = ice_sic[(ice_lon>-60)&(ice_lon<40)]
lon = ice_lon[(ice_lon>-60)&(ice_lon<40)]

# save gridded data in a new dataset
sic_wed = xr.Dataset({'sic' : (('lon', 'lat', 'time'), sic)},
  coords={'lon' : lon,
  'lat' : ice_lat,
  'time' : ice.time.values})

# save in a new file
sic_wed.to_netcdf(icedir + 'sic_grid_p5latx1lon_weddell.nc')

sys.exit()
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# 				ERA Interim data
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
with xr.open_dataset(winddir + 'eraint.nc') as wind:
    print(wind.keys())

# crop to the same time lat range
wind_crop = wind.sel(latitude=slice(-50, -90))

# grid
wind_glat, wind_glon = np.meshgrid(wind_crop.latitude, 
                                   wind_crop.longitude)