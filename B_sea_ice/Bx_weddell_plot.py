
# import modules
import numpy as np
from numpy import ma
import sys

import xarray as xr

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# directories 
workdir = '/Users/ocd1n16/Documents/PhD/data/'
altimdir = workdir + 'SSH_SLA/'

# filename and path to it
altim_path = altimdir + 'DOT_weddell.nc'

# read the dataset and print the contents
with xr.open_dataset(altim_path) as altim:
    print(altim.keys())

# extract the DOT
glat, glon = np.meshgrid(altim.latitude, altim.longitude)
import math
lon_w = -61
lon_e=41
lat_s=-73
lat_n=-50

lon_0 = lon_w + (lon_e - lon_w)/2.
ref = lat_s if abs(lat_s) > abs(lat_n) else lat_n
lat_0 = math.copysign(90., ref)
proj = 'spstere'
prj = Basemap(projection=proj, lon_0=lon_0, lat_0=lat_0,
              boundinglat=-49, resolution='l')
lons = [lon_w, lon_e, lon_w, lon_e, lon_0, lon_0]
lats = [lat_s, lat_s, lat_n, lat_n, lat_s, lat_n]
x, y  = prj(lons, lats)
ll_lon, ll_lat = prj(min(x), min(y), inverse=True)
ur_lon, ur_lat = prj(max(x), max(y), inverse=True)

m = Basemap(projection='stere', lat_0=lat_0, lon_0=lon_0, 
            llcrnrlon=ll_lon, llcrnrlat=ll_lat, 
            urcrnrlon=ur_lon, urcrnrlat=ur_lat, resolution='l')


plt.ion()
fig, ax = plt.subplots()

m.pcolormesh(glon, glat, altim.dot[:, :, 0], latlon=True)
