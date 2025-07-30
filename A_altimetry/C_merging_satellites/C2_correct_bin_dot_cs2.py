"""
CORRECT and GRID data

 - correct along-tracks SSH with OL & retrackers monthly climatology offset
 - subtract geoid (egm08, goco05c), apply  3m range
 - grid corrected data; bins must have > 30 points 
 - land mask is applied

file: dot_cs2_bmean.nc, dot_cs2_bmedian.nc

Last modified: 4 Jan 2022
"""
print('... reading libraries ...')

## libraries go here
import numpy as np
from numpy import ma

from datetime import datetime
today = datetime.today()

from scipy.stats import binned_statistic_2d as bin2d

import os
import sys
import pickle

import pandas as pd
import xarray as xr

#-------------------------------------------------------------------------------
# Define directories
voldir = '/Volumes/SamT5/PhD_data/'
ncdir = voldir + 'altimetry_cpom/1_raw_nc/'
bindir = voldir + 'altimetry_cpom/2_grid_offset/'
griddir = voldir + 'altimetry_cpom/3_grid_dot/'
geoiddir = voldir + 'geoid/geoid_eigen6s4v2/'
lmdir = voldir + 'land_masks/'

scriptdir = '/Volumes/SamT5/PhD_scripts/'
auxscriptdir = scriptdir + 'scripts/aux_func/'

sys.path.append(auxdir)
from aux_filenames import cs2_id_list as filenames
import aux_func_trend as ft

# # # # # # # # # # # # 
# - - - - - - - - - - - - - - - - - - - - - - - -
n_thresh = 30
statistic = 'mean'
geoidtype = '_eigen6s4v2' #'_goco05c' #'_egm08'
sig = 3
# - - - - - - - - - - - - - - - - - - - - - - - -
# # # # # # # # # # # # 
print("- - - - - - - - - - - - - - ")
print("> > bin statistic: %s" % statistic)
print("> > bin threshold: %s" % str(n_thresh))
print("> > geoid: %s" % geoidtype)
print("> > filter sigma: %s" % sig)
print("- - - - - - - - - - - - - - \n")
#------------------------------------------------------------------
# start from 2010-11-01
time = pd.date_range('2010-11-01', '2018-10-01', freq='1MS')
itt = len(filenames)

# Check all files have been created
notfound = 0
for i in range(itt):
    filename = ncdir + filenames[i] + '.nc'
    if (not os.path.isfile(filename)) or (not os.path.exists(filename)):
        notfound += 1

print("%s files in total, %s not found \n" % (itt, notfound))
print("first file %s" %filenames[0])
print('start time', time[0])
#------------------------------------------------------------------
# SEASONAL OFFSET 
#------------------------------------------------------------------
offsetfile = 'b02_OL_offset_cs2_'+ str(n_thresh) + statistic + '.nc'
with xr.open_dataset(bindir + offsetfile) as ol:
    print(ol.keys())
ol_dif = ol.ol_dif.values

offsetfile2 = 'b03_SAR_offset_cs2_' + str(n_thresh) + statistic +'.nc'
with xr.open_dataset(bindir + offsetfile2) as retracker1:
    print(retracker1.keys())
sarin_dif = retracker1.sar_dif.values

offsetfile3 = 'b04_LRM_offset_cs2_' + str(n_thresh) + statistic + '.nc'
with xr.open_dataset(bindir + offsetfile3) as retracker2:
    print(retracker2.keys())
sar_dif = retracker2.lrm_dif.values
#------------------------------------------------------------------

# LAND MASK
#-------------------------------------------------------------------------------
# lon grid is -180/180, 0.5 lat x 1 lon
# lm shape=(mid_lon, mid_lat)
# land=1, ocean=0
#-------------------------------------------------------------------------------
lm = xr.open_dataset(lmdir+'land_mask_gridded_50S.nc')
lmask = lm.landmask.values

#------------------------------------------------------------------
# BIN DATA
#------------------------------------------------------------------
# bin edges
edges_lon = np.linspace(-180, 180, num=361, endpoint=True)
edges_lat = np.linspace(-82, -50, num=65, endpoint=True)
eglat, eglon = np.meshgrid(edges_lat, edges_lon)

# bin centres
mid_lon = 0.5*(edges_lon[1:] + edges_lon[:-1])
mid_lat = 0.5*(edges_lat[1:] + edges_lat[:-1])
glat, glon = np.meshgrid(mid_lat, mid_lon)

#------------------------------------------------------------------
#------------------------------------------------------------------
## initialize array for the time mean SSH
lo, la = glon.shape
all_DOT = ma.zeros((lo, la, itt))
pts_in_bins = ma.zeros((lo, la, itt))
#-------------------------------------------------------------------------------
for i in range(itt):
    fname = filenames[i]
    print('Analysing M/Y: %s' % fname)
    
    filepath = ncdir + fname + '.nc'
    data = xr.open_dataset(filepath)

    ssh = data.Elevation.values
    surf = data.SurfaceType.values
    lat = data.Latitude.values
    lon = data.Longitude.values
    dist = data.distance_m.values
    retracker = data.Retracker.values

    #------------------------------------------------------------------
    # 2 bring leads to the same level as the open ocean data
    #------------------------------------------------------------------  
    month = time.month.values[i]
    ssh[surf==2] += ol_dif[month-1]

    # LRM, SAR, SARIN correction - ref to LRM
    #------------------------------------------------------------------    
    ssh[retracker==2] += sar_dif[month-1]
    ssh[retracker==3] += (sarin_dif[month-1] + sar_dif[month-1])

    # DOT
    print("geoid corrections ..")
    #------------------------------------------------------------------
    # subtract geoid; load only column with geoid height data
    gd_file = geoiddir + fname + geoidtype +'.txt'
    geoid_height = np.loadtxt(gd_file, usecols=2)
    #geoid_height = ma.masked_invalid(geoid_height)

    dot = ssh-geoid_height
    
    # corr 1 : |DOT| < 3 m
    dot_range = np.logical_and(dot<3, dot>-3)

    dot = dot[dot_range]
    lon = lon[dot_range]
    lat = lat[dot_range]
    dist = dist[dot_range]
    #------------------------------------------------------------------
    # 1 keep only data further than 10km from nearest coastline
    #------------------------------------------------------------------  
    dot = dot[dist>1e4]
    lon = lon[dist>1e4]
    lat = lat[dist>1e4]

    #------------------------------------------------------------------
    # 3. BIN DATA
    #------------------------------------------------------------------    
    print("binning DOT data ..")
    dot_bin = bin2d(lon, lat, dot, statistic=statistic,
            bins=[edges_lon, edges_lat]).statistic
    # number of points in bins
    npts = np.histogram2d(lon, lat, bins=(edges_lon, edges_lat))[0]    
    
    # mask bins that have less than a threshold number of points
    dot_bin[npts<n_thresh] = np.nan

    #------------------------------------------------------------------
    # 4. gridded land mask
    #------------------------------------------------------------------   
    dot_bin[lmask==1] = np.nan
    npts[lmask==1] = 0
  
    pts_in_bins[:, :, i] = npts
    #--------------------------------------------------------------------------
    # INTERPOLATION
    #--------------------------------------------------------------------------
    # remove missing data by nearest-neighbour interpolation
    # to prepare data for filtering
    interp_dot = ft.interp_nan(dot_bin, glon, glat)

    # Gaussian filtering
    filt_idot = ft.gaussian_filt(interp_dot, sigma=sig, mode='reflect')

    # land mask
    filt_idot[lmask==1] = np.nan

    all_DOT[:, :, i] = ma.masked_invalid(filt_idot)

mean_DOT = ma.mean(all_DOT, axis=-1)
all_SLA = all_DOT - mean_DOT[:, :, np.newaxis]

#--------------------------------------------------------------------------
# .nc file
newfile = 'dot_cs2_30b'+statistic + geoidtype +'_sig3.nc'
print("saving file %s" % newfile)
#--------------------------------------------------------------------------
ds = xr.Dataset({'dot' : (('longitude', 'latitude', 'time'), all_DOT.filled(np.nan)), 
                 'sla' : (('longitude', 'latitude', 'time'), all_SLA.filled(np.nan)),
                 'mdt' : (('longitude', 'latitude'), mean_DOT.filled(np.nan)),
                 'num_pts' : (('longitude', 'latitude', 'time'), pts_in_bins),
                 'land_mask' : (('longitude', 'latitude'), lmask)},
                coords={'longitude' : mid_lon,
                        'latitude' : mid_lat,
                        'time' : time,
                        'edge_lat' : edges_lat,
                        'edge_lon' : edges_lon})

ds.attrs['history'] = "Created " + today.strftime("%d/%m/%Y, %H:%M%S" )

ds.dot.attrs['units']='meters'
ds.sla.attrs['units']='metres'
ds.mdt.attrs['units']='metres'
ds.dot.attrs['long_name']='dynamic_ocean_topography'
ds.sla.attrs['long_name']='sea_level_anomaly'
ds.mdt.attrs['long_name']='mean_dynamic_ocean_topography'
ds.land_mask.attrs['long_name']='ocean_0_land_1' 

ds.attrs['description'] = ("CryoSat-2: \
|*Lat, Lon at bin centre and edges \
|*DOT (bin mean, only values in (-3, 3)m)\
|*SLA (ref to mean over all period)\
|*MDT \
|*time (2010-11-01, 2018-10-01)\
|*number of points in bins\
|only bins with > 30 points are used\
|*land mask (land=1, ocean=0).\
|Gaussian filter (sig=3, 150km) applied to binned data\
|Data have been filtered by sea-ice type\
and have the land mask applied.")

ds.to_netcdf(griddir + newfile)

print("Done.")
