"""
Use the SIC raw dataset to crop to the Weddell region [-60/40]
and to the polynia times

Last modified: 17 Apr 2020
"""

import numpy as np 
from numpy import ma

import xarray as xr
import pandas as pd

from scipy.interpolate import griddata

# Directories
maindir = '/Users/ocd1n16/OneDrive - University of Southampton/PhD/'
workdir = maindir + 'data/'
icedir = workdir + 'NSIDC/sic/'

with xr.open_dataset(icedir + 'sic_raw.nc') as ice0:
    print(ice0.keys())

# crop to the polynia times
#ice = ice0.sel(time=slice('2016-04-01', '2016-11-01'))
ice = ice0.sel(time=slice('2002-07-01', '2018-10-01'))

# extract lat, lon, and sic
ice_lat = ice.lat.values
ice_lon = ice.lon.values
ice_sic = ice.sic.values

# dimensions
r, c, timdim = ice_sic.shape

# crop to Weddell region
sic = ice_sic[(ice_lon>-60)&(ice_lon<40)]
lon = ice_lon[(ice_lon>-60)&(ice_lon<40)]
lat = ice_lat[(ice_lon>-60)&(ice_lon<40)]

# new GRID
lon_g = np.linspace(-60, 40, 401)
lat_g = np.linspace(-50, -82, 513)
glat, glon = np.meshgrid(lat_g, lon_g)

sic_grid = np.asarray([griddata((lon ,lat), sic[:,i], 
	(glon, glat)) for i in range(timdim)])

# save gridded data in a new dataset
sic_wed = xr.Dataset({'sic' : (('time', 'lon', 'lat'), sic_grid)},
	coords={'lon' : lon_g,
	'lat' : lat_g,
	'time' : ice.time.values})

# save in a new file
sic_wed.to_netcdf(icedir + 'weddell_sic_grid_all.nc')