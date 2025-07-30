"""
Create files with gridded LRM/SAR/SARin data from CS2 (after 2010.11)
after correcting the along-track SLA leads.

- correct for O/L
- grid retrackers separately
- save arrays in a new file

OFFSET:
- discard data less than 10km away from land
- bin ocean/leads separately 
     (compute mean/median of lat, lon, ssh, mss in every cell;
      interp to cell centre after the land mask)
- apply gridded land mask (for the points that might be inside contours)

- test difference between mean/median average in bins; how do I treat the monthly std?
- compute monthly climatology + error estimate for every point


Last modified: 10 Mar 2025
"""

## libraries go here
import numpy as np
from numpy import ma

from datetime import datetime
today = datetime.today()

import time as runtime
t_start = runtime.process_time()

from scipy.stats import binned_statistic_2d as bin2d

import xarray as xr
import pandas as pd

import os
import sys

# Define directories
#-------------------------------------------------------------------------------
voldir = '/Volumes/SamT5/PhD/PhD_data/'
ncdir = voldir + 'altimetry_cpom/1_raw_nc/'
bindir = voldir + 'altimetry_cpom/2_grid_offset/'

lmdir = voldir + 'land_masks/'

scriptdir = '/Volumes/SamT5/PhD/PhD_scripts/ch2_altimetry/'
auxscriptdir = scriptdir + 'aux_func/'

sys.path.append(scriptdir)
from aux_1_filenames import cs2_id_list
filenames = cs2_id_list

#-------------------------------------------------------------------------------
time = pd.date_range('2010-11-01', '2018-10-01', freq='1MS')

itt = len(time)
yr = time.year.values
months = time.month.values

#-------------------------------------------------------------------------------
# LAND MASK
#-------------------------------------------------------------------------------
# lon grid is -180/180, 0.5 lat x 1 lon
# lm shape=(mid_lon, mid_lat)
# land=1, ocean=0
#-------------------------------------------------------------------------------
lm = xr.open_dataset(lmdir+'land_mask_gridded_50s.nc')
lmask = lm.landmask.values

#------------------------------------------------------------------
# GRID
#------------------------------------------------------------------
# bin edges
edges_lon = np.linspace(-180, 180, num=361, endpoint=True)
edges_lat = np.linspace(-82, -50, num=65, endpoint=True)
eglat, eglon = np.meshgrid(edges_lat, edges_lon)

# bin centres
mid_lon = 0.5*(edges_lon[1:] + edges_lon[:-1])
mid_lat = 0.5*(edges_lat[1:] + edges_lat[:-1])
glat, glon = np.meshgrid(mid_lat, mid_lon)

londim, latdim = glat.shape

# # # # # # # # # # # # 
statistic = 'median'
# # # # # # # # # # # # 

print("- - - - - - - - - - - - - - ")
print("> > bin statistic: %s" % statistic)
print("- - - - - - - - - - - - - - \n")
#------------------------------------------------------------------
# OL offset
#------------------------------------------------------------------
with xr.open_dataset(bindir + 'b02_OL_offset_cs2_30' + statistic +'.nc') as offset:
    print(offset.keys())

ol_offset = offset.ol_dif.values

ssha_sar_all, ssha_sarin_all = [np.zeros((itt, londim, latdim)) for _ in range(2)]
npts_sar_all, npts_sarin_all = [np.zeros((itt, londim, latdim)) for _ in range(2)]

for j, filename in enumerate(filenames):
    print(filename)

    filepath = ncdir + filename + '.nc'

    ds = xr.open_dataset(filepath)
    lat = ds.Latitude.values
    lon = ds.Longitude.values
    ssh = ds.Elevation.values
    surf = ds.SurfaceType.values
    dist = ds.distance_m.values
    mss = ds.MeanSSH.values
    retrack = ds.Retracker.values
    
    # time start-end date
    print("date start: %s" % ds.Time[0].dt.strftime('%m.%Y').values)
    print("date end: %s" % ds.Time[-1].dt.strftime('%m.%Y').values)

    if (ds.Time[0].dt.year.values != yr[j] 
        or ds.Time[0].dt.month.values != months[j]):
        print("Year or month do not agree with time coordinate!")
        sys.exit()
    #------------------------------------------------------------------
    # 1 keep only data further than 10km from nearest coastline
    #------------------------------------------------------------------  
    ssh = ssh[dist>1e4]
    lon = lon[dist>1e4]
    lat = lat[dist>1e4]
    mss = mss[dist>1e4]
    surf = surf[dist>1e4]
    retrack = retrack[dist>1e4]

    ssha = ssh-mss

    #------------------------------------------------------------------
    # 2 add OL offset to along-track lead SLA
    #------------------------------------------------------------------  
    # add the monthly climatology correction to the along_track leads
    month = ds.Time.dt.month.values[j]

    ssha[surf==2] += ol_offset[month-1]

    #------------------------------------------------------------------
    # 3 split into retrackers; level SAR/SARIn to start with
    #------------------------------------------------------------------ 
    ssha_sar = ssha[retrack==2]
    lon_sar = lon[retrack==2]
    lat_sar = lat[retrack==2]

    ssha_sarin = ssha[retrack==3]
    lon_sarin = lon[retrack==3]
    lat_sarin = lat[retrack==3]

    #------------------------------------------------------------------
    # 4. BIN DATA
    #------------------------------------------------------------------    
    print("binning SAR data ..")
    x, y, var = lon_sar, lat_sar, ssha_sar
    ssha_sar_bin = bin2d(x, y, var, statistic=statistic,
                 bins=[edges_lon, edges_lat]).statistic
    # number of points in bins
    npts_sar = np.histogram2d(x, y, bins=(edges_lon, edges_lat))[0]    

    print("binning SARIn data ..")
    x, y, var = lon_sarin, lat_sarin, ssha_sarin
    ssha_sarin_bin = bin2d(x, y, var, statistic=statistic,
                 bins=[edges_lon, edges_lat]).statistic
    # number of points in bins
    npts_sarin = np.histogram2d(x, y, bins=(edges_lon, edges_lat))[0]     

    #------------------------------------------------------------------
    # 5. gridded land mask
    #------------------------------------------------------------------   
    ssha_sar_bin[lmask==1] = np.nan
    ssha_sarin_bin[lmask==1] = np.nan
    npts_sar[lmask==1] = 0
    npts_sarin[lmask==1] = 0

    ssha_sar_all[j, :, :] = ssha_sar_bin
    ssha_sarin_all[j, :, :] = ssha_sarin_bin
    npts_sar_all[j, :, :] = npts_sar
    npts_sarin_all[j, :, :] = npts_sarin

# save SLA in a pkl file
ds_all = xr.Dataset({'ssha_sar' : (('time', 'longitude', 'latitude'), ssha_sar_all),
                     'ssha_sarin' : (('time', 'longitude', 'latitude'), ssha_sarin_all),
                     'npts_sar' : (('time', 'longitude', 'latitude'), npts_sar_all),
                     'npts_sarin' : (('time', 'longitude', 'latitude'), npts_sarin_all),
                     'land_mask' : (('longitude', 'latitude'), lmask)},
                    coords={'longitude' : mid_lon,
                            'latitude' : mid_lat,
                            'time' : time})

newfilename = 'b03_bin_ssha_SAR_cs2_' + statistic+ '.nc'
ds_all.to_netcdf(bindir + newfilename)

t_stop = runtime.process_time()
print("execution time: %.1f min " %((t_stop-t_start)/60))

print('The end.')
