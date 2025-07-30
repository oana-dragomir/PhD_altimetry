"""
Compute monthly means from daily sampled AVISO/CMEMS ADT
Subsample onto the altimetry/Rye2014 grid (1/3 deg from 1/4 deg)

Save arrays in a file

Time span:
- 1993 to 2018 (reprocessed, L4)

Last modified: 19 Apr 2021
"""

import numpy as np
from numpy import ma

from datetime import datetime

import pandas as pd

import xarray as xr
import xesmf as xe

import matplotlib.pyplot as plt

import glob
import sys
import os
#-------------------------------------------------------------------
# Directories
#-------------------------------------------------------------------
voldir = '/Volumes/SamT5/PhD/data/'
griddir = voldir + 'altimetry_cpom/3_grid_dot/'
adtdir = voldir + 'aviso/monthly_aviso/adt/'
# -----------------------------------------------------------------------------
#           my Altimetry product (for re-gridding)
# -----------------------------------------------------------------------------
altfile = 'dot_all_30bmedian_egm08.nc'

with xr.open_dataset(griddir + altfile) as alt:
    print(alt.keys())

# GRID coordinates
# at bin centres
alt_glat, alt_glon = np.meshgrid(alt.latitude, alt.longitude)

# -----------------------------------------------------------------------------
"""
# or on the Rye grid (1/3 deg)
mat  = sio.loadmat(matpath+'MADTtrend.mat')

# madt is in cm; convert to m
#rye_madt = ma.masked_invalid(mat['madt'][:])/1e2
#rye_mss = ma.masked_invalid(mat['meanmadt'][:])/1e2
gmlon = mat['lon_mat'][:]
gmlat = mat['lat_mat'][:]
"""
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Monthly averages
# --------------------------------------------------------------
# regrid using xarray regridding object
ds_out = xr.Dataset({'lat' : (['lat'], alt.latitude),
                    'lon':(['lon'], alt.longitude)})

time_ls = []
ii = 0
for filepath in glob.iglob(adtdir+"adt_*.nc"):
    fname = os.path.basename(filepath)
    print(fname)

    # extract the year/month of the file
    year, month = int(fname[4:8]), int(fname[9:11])
    time_ls.append(datetime(year, month, 1))

    adt = xr.open_dataset(filepath)

    # crop south of 50s
    adt_crop = adt.adt.sel(latitude = slice(-90, -50))

    # monthly mean
    adt_mean = adt_crop.mean('time')

    # > > re-grid
    lon = adt_crop.longitude.values
    lon[lon>180] = lon[lon>180] - 360
    adt_mean["longitude"] = lon
    adt_mean = adt_mean.sortby("longitude")

    regridder = xe.Regridder(adt_mean, ds_out, 'bilinear')
    gadt = regridder(adt_mean)

    if ii == 0:
        cmems_adt = adt_mean.values
        cmems_gadt = gadt.values
    else:
        cmems_adt = np.dstack((cmems_adt, adt_mean.values))
        cmems_gadt = np.dstack((cmems_gadt, gadt.values))

    ii += 1


cmems_ds = xr.Dataset({'adt' : (('time', 'lon', 'lat'), cmems_adt.T),
    'gadt' : (('time', 'glon', 'glat'), cmems_gadt.T)},
    coords = {'time' : time_ls,
            'lat' : adt_mean.latitude.values,
            'lon' : adt_mean.longitude.values,
            'glon' : alt.longitude.values,
            'glat' : alt.latitude.values})


newfname = 'cmems_adt_1993_2018.nc'
cmems_ds.to_netcdf(adtdir + newfname)

print("File %s /n saved to %s" % (newfname, adtdir))

sys.exit()

"""

ii = 0
for year in years:
    if year==2018:
        months_idx = months2018
    else:
        months_idx = months
    for month in months_idx:
        if month < 10:
            month_str = "0"+str(month)
        else:
            months_str = str(month)

        filename = var_str+str(year)+"_"+month_str+".nc"
        filepath = altimdir+filename
        print("Computing monthly mean for "+filename)

        data = Dataset(filepath, 'r+')
        var = data['adt'][:].T
        var_monthly_mean[:, :, ii] = np.nanmean(var, axis=-1)
        data.close()
        
        #regrid to meanmadt1
        gvar[:, :, ii] = griddata((ilon.flatten(), ilat.flatten()), 
                                  var_monthly_mean[:, :, ii].flatten(),
                                  (gmlon, gmlat), method='linear')
        
        date = datetime(year, month, 1)
        var_time[ii] = date2num(date, units=time_units, calendar='gregorian')
        ii += 1
"""
