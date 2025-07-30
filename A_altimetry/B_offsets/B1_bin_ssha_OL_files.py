"""
Create monthly grids of Ocean/Lead data
save all in a file: b0_bin_ssha_OL_satellite.nc where satellite=env or cs2.

Use these files afterwards to compute the offset.

OFFSET:
- discard data less than 10km away from land
- bin ocean/leads SSHA separately 
     (compute mean ssh, mss in every cell)
- apply gridded land mask (for the points that might be inside contours)
- keep track of number of points in each bin

Last modified: 10 Mar 2025
"""

## libraries go here
import time as runtime
t_start = runtime.process_time()

import numpy as np
from numpy import ma

from scipy.stats import binned_statistic_2d as bin2d

import xarray as xr
import pandas as pd

import sys

print(".... libraries read successfully")

# Define directories
#-------------------------------------------------------------------------------
voldir = '/Volumes/SamT5/PhD_data/'
ncdir = voldir + 'altimetry_cpom/1_raw_nc/'
bindir = voldir + 'altimetry_cpom/2_grid_offset/'
lmdir = voldir + 'land_masks/'

scriptdir = '/Volumes/SamT5/PhD_scripts/'
auxscriptdir = scriptdir + 'scripts/aux_func/'
sys.path.append(auxscriptdir)
from aux_1_filenames import cs2_id_list, env_id_list

#------------------------------------------------------------------
# FILES AND TIME AXES
#------------------------------------------------------------------
# CS2 TIME: 2010-11/2018.10
# ENV TIME: 2002-07/2012.03
fnames_env = env_id_list
fnames_cs2 = cs2_id_list

time_env = pd.date_range('2002-07-01', '2012-03-01', freq='1MS')
time_cs2 = pd.date_range('2010-11-01', '2018-10-01', freq='1MS')

#------------------------------------------------------------------
# LAND MASK
#------------------------------------------------------------------
# lon grid is -180/180, 0.5 lat x 1 lon
# lm shape=(mid_lon, mid_lat)
# land=1, ocean=0
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
#------------------------------------------------------------------
#------------------------------------------------------------------
# # # # # # # # # # # # 
statistic = 'median'
satellite = 'env'
# # # # # # # # # # # # 
print("- - - - - - - - - - - - - - ")
print("> > bin statistic: %s" % statistic)
print("> > satellite: %s" % satellite)
print("- - - - - - - - - - - - - - \n")
#------------------------------------------------------------------
#------------------------------------------------------------------
if satellite == 'env':
    time = time_env
    files = fnames_env
elif satellite =='cs2':
    time = time_cs2
    files = fnames_cs2

# - - - - - - - - - - - - - - - - - - - - - - - - - 
itt = len(time)
yr = time.year.values
months = time.month.values

# store data in some arrays
ssha_o_all, ssha_l_all = [np.zeros((itt, londim, latdim)) for _ in range(2)]
npts_o_all, npts_l_all = [np.zeros((itt, londim, latdim)) for _ in range(2)]

for j in range(itt):
    print(files[j])

    filepath = ncdir + files[j] + '.nc'
    
    ds = xr.open_dataset(filepath)
    lat = ds.Latitude.values
    lon = ds.Longitude.values
    ssh = ds.Elevation.values
    surf = ds.SurfaceType.values
    dist = ds.distance_m.values
    mss = ds.MeanSSH.values
    tim = ds.Time.values

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

    #------------------------------------------------------------------
    # 2 split into ocean/leads
    #------------------------------------------------------------------  
    ssh_o = ssh[surf==1]
    lon_o = lon[surf==1]
    lat_o = lat[surf==1]
    mss_o = mss[surf==1]
    ssha_o = ssh_o - mss_o

    ssh_l = ssh[surf==2]
    mss_l = mss[surf==2]
    lon_l = lon[surf==2]
    lat_l = lat[surf==2]
    ssha_l = ssh_l - mss_l

    #------------------------------------------------------------------
    # 3. BIN DATA
    #------------------------------------------------------------------    
    print("binning ocean data ..")
    ssha_o_bin = bin2d(lon_o, lat_o, ssha_o, 
                    statistic=statistic,
                    bins=[edges_lon, edges_lat]).statistic
    # number of points in bins
    npts_o = np.histogram2d(lon_o, lat_o, bins=(edges_lon, edges_lat))[0]  

    print("binning lead data ..")
    ssha_l_bin = bin2d(lon_l, lat_l, ssha_l,
                    statistic=statistic,
                    bins=[edges_lon, edges_lat]).statistic
    npts_l = np.histogram2d(lon_l, lat_l, bins=(edges_lon, edges_lat))[0] 

    #------------------------------------------------------------------
    # 4. apply gridded land mask
    #------------------------------------------------------------------   
    ssha_o_bin[lmask==1] = np.nan
    ssha_l_bin[lmask==1] = np.nan
    npts_o[lmask==1] = 0
    npts_l[lmask==1] = 0

    ssha_o_all[j, :, :] = ssha_o_bin
    ssha_l_all[j, :, :] = ssha_l_bin
    npts_o_all[j, :, :] = npts_o
    npts_l_all[j, :, :] = npts_l

# save SLA in a  file
ds_all = xr.Dataset({'ssha_o' : (('time', 'longitude', 'latitude'), ssha_o_all),
                     'ssha_l' : (('time', 'longitude', 'latitude'), ssha_l_all),
                     'npts_o' : (('time', 'longitude', 'latitude'), npts_o_all),
                     'npts_l' : (('time', 'longitude', 'latitude'), npts_l_all),
                     'land_mask' : (('longitude', 'latitude'), lmask)},
                    coords={'longitude' : mid_lon,
                            'latitude' : mid_lat,
                            'time' : time})
    
newfilename = 'b01_bin_ssha_OL_' + satellite + '_' + str(statistic) + '.nc'
print("\n filename %s \n" % newfilename)

ds_all.to_netcdf(bindir + newfilename)
print("File saved in %s" % bindir)

t_stop = runtime.process_time()
print("execution time: %.1f min " %((t_stop-t_start)/60))

print('The end.')
