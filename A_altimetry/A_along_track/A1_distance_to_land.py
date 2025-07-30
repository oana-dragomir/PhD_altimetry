"""
Compute the distance of every spot measurement to the 
nearest coastline and add it as a new variable (distance_m)
to the existing nc file.

Last modified: 12 Mar 2021

Update: the geo-py package has not been maintained and there 
are errors when trying to install it (24 Mar 2023); 
a different package should be found to calculate the 
distance on the ellipsoid to the nearest coastline (but no time for me..)


"""

import numpy as np
from numpy import ma

from netCDF4 import Dataset, num2date

import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon as mpolyg
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rcParams
import matplotlib

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import pickle

from geo import ellipsoid  

import sys
import os
import glob


# Directories
workdir = '/Volumes/SamT5/PhD_data/'
ncdir = workdir + 'altimetry_cpom/1_raw_nc/'
lmdir = workdir + 'land_masks/'

# coastlines (nested lists containing the closed contours)
with open(lmdir + 'coastline_nested_lists.pkl', 'rb') as f:
    coast = pickle.load(f) 

# 1. compute distance to the closest coastline point;
# extract all coastline points from the contours
clon = np.asarray([float(item) for sublist in coast[:, 0] for item in sublist])
clat = np.asarray([float(item) for sublist in coast[:, 1] for item in sublist])

# 1.2 remove duplicate points where contours overlap
coast_pts = np.c_[clon, clat]
df_coast_pts = pd.DataFrame(coast_pts)
df_coast_pts_unique = df_coast_pts.drop_duplicates(keep='first')

ucoast_pts = df_coast_pts_unique.values

# - - - - -  
plon, plat = ucoast_pts[:,0], ucoast_pts[:, 1]
# - - - - - - - - - - - - - - - - - - - - - - - - 
# 2. make polygons to check which points are inside


# - - - - - - - - - - - - - - - - - - - - - - - - 
# read file but don't close the dataset yet

for filepath in glob.iglob(ncdir+"201*.nc"):

    filename = os.path.basename(filepath)
    print(filename)

    ds = Dataset(filepath, 'r+')
    lat = ds['Latitude'][:]
    lon = ds['Longitude'][:]
    # - - - - - - - - - - - - - - - - - - - - - - - - 

    dist = np.zeros(len(lat))
    idx_dist = np.arange(len(lat))

    # box dimensions/width
    for l0 in np.linspace(-179.75, 179.75, 720, endpoint=True):
        
        # # # crop data/coastline into longitude slices # # #

        dL = 0.5    # half width of the coastline box
        dl = 0.25   # half width of the data box

        # indices and lat/lon in the box
        alt_crop = np.logical_and(lon>l0-dl, lon<l0+dl)
        idx_dist_i = idx_dist[alt_crop] 
        lat_box = lat[alt_crop]
        lon_box = lon[alt_crop]
        
        # coastline points in the box; coast box is 0.5 lon degrees wider
        # add a condition at the discontinuity -180/180
        if l0 ==-179.75:
            coast_crop = np.logical_or(plon<l0+dL, plon>179.75)
            plon_box = plon[coast_crop]
            plat_box = plat[coast_crop]

        else:
            coast_crop = np.logical_and(plon>l0-dL, plon<l0+dL)
            plon_box = plon[coast_crop]
            plat_box = plat[coast_crop]

        # loop ..
        for j, (lon_j, lat_j) in enumerate(zip(lon_box, lat_box)):
            distance = [ellipsoid.distance((lon_j, lat_j), (ulon, ulat)) 
                        for (ulon, ulat) in zip(plon_box, plat_box)]
            # pick minimum distance
            dist[idx_dist_i[j]] = min(distance)

    ds.createVariable("distance_m", "f4", ("nrows"))
    ds["distance_m"][:] = dist
    ds["distance_m"].units = 'metres'
    ds["distance_m"].long_name = 'distance on the WGS84 ellipsoid from the nearest coastline point'

    ds.close()


sys.exit()