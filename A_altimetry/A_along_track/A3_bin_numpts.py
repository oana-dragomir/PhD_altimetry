"""
Bin data to look at a map of the NUMBER of points in every grid

Last modified: 24 Mar 2023

# CS2 TIME: 2010-11/2018.10
# ENV TIME: 2002-07/2012.03
"""

## libraries go here
import numpy as np

from scipy.stats import binned_statistic_2d as bin2d

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import xarray as xr

import sys, os, fnmatch

# Define directories
#-------------------------------------------------------------------------------
voldir = '/Volumes/SamT5/PhD_data/'
ncdir = voldir + 'altimetry_cpom/1_raw_nc/'
lmdir = voldir + 'land_masks/'
#------------------------------------------------------------------
# FILES AND TIME AXES
#------------------------------------------------------------------
# read_files = []
# for filename in os.listdir(ncdir):
#     if fnmatch.fnmatch(filename, 'month*.nc'):
#         print(filename)
#         read_files.append(os.path.join(ncdir + filename))
        
# all_files = sorted(read_files)


#pick file to plot

plot_annotation = '2008.09'
filename = 'month0209.nc'

with xr.open_dataset(ncdir + filename) as ds:
    print(ds.keys())

#------------------------------------------------------------------
# LAND MASK
#------------------------------------------------------------------
# lon grid is -180/180, 0.5 lat x 1 lon
# lm shape=(mid_lon, mid_lat)
# land=1, ocean=0
land_file = 'land_mask_gridded_50s.nc'

with xr.open_dataset(lmdir + land_file) as topo:
    print(topo.keys())
#------------------------------------------------------------------
# GRID
#------------------------------------------------------------------
# bin edges
edges_lon = np.linspace(-180, 180, num=361, endpoint=True)
edges_lat = np.linspace(-82, -50, num=65, endpoint=True)
eglat, eglon = np.meshgrid(edges_lat, edges_lon)

#------------------------------------------------------------------
#------------------------------------------------------------------
# keep only data further than 10km from nearest coastline
#------------------------------------------------------------------  
ds1 = ds.where(ds.distance_m>1e4)

#------------------------------------------------------------------
# BIN DATA
#------------------------------------------------------------------    
print("binning data ..")
# number of points in bins
npts = np.histogram2d(ds1.Longitude.values,
                    ds1.Latitude.values,
                    bins=(edges_lon, edges_lat))[0]  

#------------------------------------------------------------------
# apply gridded land mask
#------------------------------------------------------------------   
npts[topo.landmask==1] = 0

#------------------------------------------------------------------ 
plt.ion()
m = Basemap(projection='spstere',
    boundinglat=-50,
    resolution='i',
    lon_0=180,
    round=True)
cmap =cm.get_cmap("bone_r", 30)
# - - - - - - - - - - - -

fig, ax = plt.subplots(figsize=(5, 5))
cs = m.pcolormesh(eglon, eglat, npts,
            vmin=0, vmax=400, latlon=True, cmap=cmap)
fig.colorbar(cs, ax=ax, extend='max', shrink=0.5, pad=0.1)
m.drawcoastlines(color='purple')
ax.annotate(plot_annotation, xy=(.4, .5),
            xycoords='figure fraction',
            weight='bold')
plt.tight_layout()

