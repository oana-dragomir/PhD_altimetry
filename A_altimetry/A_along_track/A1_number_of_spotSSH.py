"""
Look at the number of samples in every month and how they are distributed
across the retrackers after removing data less than 10 km away from coastline.


Last modified: 24 Mar 2023
"""

import numpy as np
from numpy import ma

from netCDF4 import Dataset, num2date

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import datetime

import sys
# --------------------------------------------------------
# Directories
workdir = '/Volumes/SamT5/PhD/data/'
ncdir = workdir + 'altimetry_cpom/1_raw_nc/'
coastdir = workdir + 'land_masks/holland_vic/'
figdir = ncdir + 'nc_figures/'

# filenames
filenamespath = '/Users/ocd1n16/PhD_git/'
sys.path.append(filenamespath)
from aux_1_filenames import cs2_id_list, env_id_list

time_units = 'days since 1950-01-01 00:00:00.0'

# ENVISAT
# -----------------------------------------------------------------------------
itt = len(env_id_list)
print("Number of ENVISAT files: %s" % itt)

ntot_env, num_env, num_o_env, num_l_env, time_env = [ma.zeros((itt)) for _ in range(5)]

for i in range(itt):
    filename = ncdir + env_id_list[i] + '.nc'

    print("processing "+env_id_list[i])

    ds = Dataset(filename, 'r+')
    surf = ds['SurfaceType'][:]
    dist = ds['distance_m'][:]
    time_env[i] = ds['Time'][:][0]
    ds.close()

    # TOTAL number of points to start with
    ntot_env[i] = len(surf)

    #pick only points further than 10 km from the coastline
    surf = surf[dist>1e4]

    # split into O-L
    ocean = surf[surf==1]
    leads = surf[surf==2]

    num_env[i] = len(surf)
    num_o_env[i] = len(ocean)
    num_l_env[i] = len(leads)

ntot_env = ma.masked_equal(ntot_env, 0)
num_env = ma.masked_equal(num_env, 0)
num_o_env = ma.masked_equal(num_o_env, 0)
num_l_env = ma.masked_equal(num_l_env, 0)


# CS2
# -----------------------------------------------------------------------------
itt = len(cs2_id_list)
print("Number of CryoSat-2 files: %s" % itt)

ntot, num_cs2, no_lrm, nl_lrm, no_sar, nl_sar, no_sarin, nl_sarin = [np.zeros((itt)) for
                                                           _ in range(8)]
time_cs2 = ma.zeros((itt))

for i in range(itt):
    filename = ncdir + cs2_id_list[i] + '.nc'
    
    print("processing "+cs2_id_list[i])

    ds = Dataset(filename, 'r+')
    surf = ds['SurfaceType'][:]
    dist = ds['distance_m'][:]
    time_cs2[i] = ds['Time'][:][0]
    lat = ds['Latitude'][:]
    retracker = ds['Retracker'][:]
    ds.close()

    # total number of points
    ntot[i] = len(surf)

    #discard points less than 10km away from coastline
    surf = surf[dist>1e4]
    lat = lat[dist>1e4]
    retracker = retracker[dist>1e4]
    
    num_cs2[i] = len(surf)
    # split into O-L
    # ------------------------------------
    o_lat, l_lat = lat[:], lat[:]

    o_lat[surf==2] = ma.masked
    l_lat[surf==1] = ma.masked

    # -------------------------------------
    LRM_o_lat = o_lat[retracker==1]
    LRM_l_lat = l_lat[retracker==1]

    SAR_o_lat = o_lat[retracker==2]
    SAR_l_lat = l_lat[retracker==2]

    SARin_o_lat = o_lat[retracker==3]
    SARin_l_lat = l_lat[retracker==3]
    
    no_lrm[i] = ma.count(LRM_o_lat)
    nl_lrm[i] = ma.count(LRM_l_lat)
    no_sar[i] = ma.count(SAR_o_lat)
    nl_sar[i] = ma.count(SAR_l_lat)
    no_sarin[i] = ma.count(SARin_o_lat)
    nl_sarin[i] = ma.count(SARin_l_lat)

ntot = ma.masked_equal(ntot, 0)
num_cs2 = ma.masked_equal(num_cs2, 0)
no_lrm = ma.masked_equal(no_lrm, 0)
nl_lrm = ma.masked_equal(nl_lrm, 0)
no_sar = ma.masked_equal(no_sar, 0)
nl_sar = ma.masked_equal(nl_sar, 0)
no_sarin = ma.masked_equal(no_sarin, 0)
nl_sarin = ma.masked_equal(nl_sarin, 0)

#------------------------------------------------------
# time axis
date_cs2 = num2date(time_cs2, units=time_units,
                    calendar='gregorian',
                    only_use_cftime_datetimes=False)
date_env = num2date(time_env, units=time_units,
                    calendar='gregorian',
                    only_use_cftime_datetimes=False)


#------------------------------------------------------
# fraction of data points that are less than 10 km away from coastline
# CS2
cs2_on_land = ntot - num_cs2
cs2_fr_on_land = cs2_on_land / ntot

fig, ax = plt.subplots(figsize=(10,3))
ax.plot(date_cs2, cs2_fr_on_land*100, c='k', marker='o', 
        markersize=3, lw=1)
ax.set_ylabel("%")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.tight_layout()

#ENV
env_on_land = ntot_env - num_env
env_fr_on_land = env_on_land / ntot_env

fig, ax = plt.subplots(figsize=(10,3))
ax.plot(date_env, env_fr_on_land*100, c='k', marker='o', 
        markersize=3, lw=1)
ax.set_ylabel("%")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.tight_layout()

#------------------------------------------------------
# total number of data every month and how they're split between modes/O-L

cs2_total_num = ma.sum(num_cs2)
cs2_fr_total = num_cs2/cs2_total_num

env_total_num = ma.sum(num_env)
env_fr_total = num_env/env_total_num

nmax = np.max(np.concatenate([num_cs2, num_env]))
cs2_fr_max = num_cs2/nmax
env_fr_max = num_env/nmax

#-------------------
#-------------------
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(date_cs2, num_cs2, 
       width=20,
       facecolor='lightgrey',
       alpha=0.7)
ax.plot(date_cs2, no_lrm, c='k', marker='o', markersize=3, lw=1, label='lrm-oc')
ax.plot(date_cs2, nl_lrm, c='k', marker='X', label='lrm-l')
ax.plot(date_cs2, no_sar, c='maroon', marker='o', markersize=3, lw=1, label='sar-oc')
ax.plot(date_cs2, nl_sar, c='darkorange', marker='x',  label='sar-l')
ax.plot(date_cs2, no_sarin, c='mediumblue', marker='o', markersize=3, label='sarin-oc')
ax.plot(date_cs2, nl_sarin, c='c', marker='x', label='sarin-l')

ax.legend(ncol=6, loc='upper right', bbox_to_anchor=(.95, 1.05))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.spines[['right', 'top']].set_visible(False)
plt.tight_layout()

#-------------------

fig, ax = plt.subplots(figsize=(10, 3))
ax.bar(date_env, num_env,
       width=20,
       facecolor='lightgrey',
       alpha=0.7)
ax.plot(date_env, num_o_env, c='k', label='oc')
ax.plot(date_env, num_l_env, c='c', label='l')
ax.legend(ncol=6, loc='upper right', bbox_to_anchor=(.95, 1.15))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.spines[['right', 'top']].set_visible(False)
plt.tight_layout()
