"""
CORRECT and GRID ENVISAT data

 - correct along-tracks SSH with OL monthly climatology offset
 - subtract geoid (egm08, eigen), apply  3m range
 - grid corrected data; bins must have > 30 points 
 - land mask is applied

file: dot_env_bmean.nc, dot_env_bmedian.nc

Last modified: 6 Oct 2023
"""

import numpy as np
from numpy import ma

import matplotlib.pyplot as plt

from datetime import datetime
today = datetime.today()

import os
import sys
import pickle

import pandas as pd
import xarray as xr

#-------------------------------------------------------------------------------
# Define directories
voldir = '/Volumes/SamT5/PhD/data/'
ncdir = voldir + 'altimetry_cpom/1_raw_nc/'
bindir = voldir + 'altimetry_cpom/2_grid_offset/'
griddir = voldir + 'altimetry_cpom/3_grid_dot/'
lmdir = voldir + 'land_masks/'

localdir = '/Users/ocd1n16/PhD_git/'
scriptdir = localdir + 'scripts/'
auxdir = localdir + 'scripts/aux_func/'

# import my functions
sys.path.append(auxdir)
from aux_1_filenames import env_id_list as filenames
import aux_func_trend as ft

# # # # # # # # # # # # 
n_thresh = 30
statistic = 'median'
geoidtype = '_eigen6s4v2' #'_goco05c' #'_egm08'

if geoidtype == '_goco05c':
    geoiddir = voldir + 'geoid/geoid_goco05/'
elif geoidtype == '_egm08':
    geoiddir = voldir + 'geoid/geoid_emg08/'
elif geoidtype == '_eigen6s4v2':
    geoiddir = voldir + 'geoid/geoid_eigen6s4v2/'
# # # # # # # # # # # # 

print("- - - - - - - - - - - - - - ")
print("> > bin statistic: %s" % statistic)
print("> > bin threshold: %s points" % str(n_thresh))
print("> > geoid: %s" % geoidtype)
print("- - - - - - - - - - - - - - \n")
#------------------------------------------------------------------
time = pd.date_range('2002-07-01', '2012-03-01', freq='1MS')

itt = len(filenames)

# Check all files have been created
notfound = 0
for i in range(itt):
    filename = ncdir + filenames[i] + '.nc'
    if (not os.path.isfile(filename)) or (not os.path.exists(filename)):
        notfound += 1

print("%s files in total, %s not found \n" % (itt, notfound))

#------------------------------------------------------------------
# SEASONAL OFFSET 
#------------------------------------------------------------------
offsetfile = 'b02_OL_offset_env_' + str(n_thresh) + statistic +'.nc'
with xr.open_dataset(bindir + offsetfile) as offset:
    print(offset.keys())
ol_dif = offset.ol_dif.values
#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------
#-------------------------------------------------------------------------------
for i in range(1):
    fname = filenames[i]
    print('Analysing M/Y: %s' % fname)
    
    filepath = ncdir + fname + '.nc'
    data = xr.open_dataset(filepath)

    ssh = data.Elevation.values
    surf = data.SurfaceType.values
    lat = data.Latitude.values
    lon = data.Longitude.values
    dist = data.distance_m.values
    track_num = data.track_num.values
    tim = data.Time.values

    #------------------------------------------------------------------
    # 2 bring leads to the same level as the open ocean data
    #------------------------------------------------------------------  
    month = time.month.values[i]
    #print("month %s" % month)

    ssh[surf==2] += ol_dif[month-1]
    
    #data.Elevation[data.SurfaceType==2] += ol_dif[month-1]

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
    track_num = track_num[dot_range]
    tim = tim[dot_range]
    #------------------------------------------------------------------
    # 1 keep only data further than 10km from nearest coastline
    #------------------------------------------------------------------  
    dot = dot[dist>1e4]
    lon = lon[dist>1e4]
    lat = lat[dist>1e4]
    track_num = track_num[dist>1e4]
    tim = tim[dist>1e4]

lon15 = lon[tim>tim[track_num==3][0] + np.timedelta64(15, 'D')]
lat15 = lat[tim>tim[track_num==3][0] + np.timedelta64(15, 'D')]

fig, ax = plt.subplots()
ax.scatter(lon[track_num==3], lat[track_num==3])
ax.scatter(lon15, lat15)


fig, ax = plt.subplots()
ax.scatter(lon[track_num==3], lat[track_num==3])
ax.scatter(lon[track_num==10], lat[track_num==10])

print((tim[track_num==0][-1] - tim[track_num==0][0])/(3600*24))
