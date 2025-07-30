"""
Create lat/lon grid for SIC data
save in a file

Last modified: 3 Apr 2020
"""

import numpy as np
import numpy.ma as ma

import pandas as pd
import xarray as xr

import cartopy.crs as ccrs

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from datetime import datetime

import glob
import sys
import os

# Directories
maindir = '/Volumes/SamT5/PhD_data/'
workdir = maindir + 'data/'
icedir = workdir + 'NSIDC/sic/'
topodir = workdir + 'topog/'
altdir = workdir + 'SSH_SLA/'
sicfigdir = workdir + '../PhD_figures/Figures_SIC/'

# import my functions
fcdir = maindir + 'scripts/'
sys.path.append(fcdir)
import aux_stereoplot as st 

#-------------------------------------------------------------------
# read one file to get lat/lon from the stereographic coordinates
#-------------------------------------------------------------------
filepath = icedir + 'bin/nt_200809_f17_v1.1_s.bin'

fname = os.path.basename(filepath)
print(fname)

# extract the year/month of the file
year, month = int(fname[3:7]), int(fname[7:9])
time_i = datetime(year, month, 1)

with open(filepath, 'rb') as fr:
    hdr = fr.read(300)
    ice = np.fromfile(fr, dtype=np.uint8)

# recover the rows x cols format
rows, cols = 332, 316
ice = ice.reshape(rows, cols)
# convert from bytes and scale to 0-1
ice = ma.masked_less(ice / 250., 0.)
# mask out the land
ice = ma.masked_greater(ice, 1.0)

# just an initial check
#plt.contourf(ice[:,:,0].T)

# Polar Sterographic grid coordinates
dx = dy = 25000

x = np.arange(-3950000, +3950000, +dx)
y = np.arange(+4350000, -3950000, -dy)

# convert to lat/lon geodetic coords
xg, yg = np.meshgrid(x,y)
use_proj = ccrs.Geodetic()
kw = dict(central_latitude=-90, central_longitude=0, true_scale_latitude=-70)
out_xy = use_proj.transform_points(ccrs.Stereographic(**kw),xg,yg)
lon = out_xy[:,:,0]
lat = out_xy[:,:,1]

# ... and plot to check how it looks
cbar_range = [0., 1.]
cmap = cm.get_cmap('cividis_r', 10)
cbar_units = 'SIC'
cbar_extend = 'neither'

plt.ion()
fig, ax, m = st.spstere_plot(lon, lat, ice,
                       cbar_range, cmap, 
                       cbar_units, cbar_extend)
lp = m.contour(lon, lat, ice, levels=[.15], 
          colors='crimson', linewidths=1., 
          latlon=True)
lp_label = '15%'
lp.collections[0].set_label(lp_label)
ax.legend(loc='lower right', bbox_to_anchor=(.15, .85), fontsize=12)
ax.annotate("%s/%s" % (year, month), 
  xycoords='axes fraction', xy=(.45, .5), 
  bbox=dict(facecolor='lavender',
            edgecolor='lavender',
            boxstyle='round'))
#fig.savefig(sicfigdir + 'sic_'+fname[3:9]+'.png')
sys.exit()
#-------------------------------------------------------------------
# collate all SIC files into one
#-------------------------------------------------------------------
timdim = 196
sic = ma.zeros((rows, cols, timdim))
time_ls = []
# recover the rows x cols format
rows, cols = 332, 316

i = 0 
for filepath in glob.iglob(icedir+"bin/nt_*.bin"):

    fname = os.path.basename(filepath)
    print(i, fname)

    # extract the year/month of the file
    year, month = int(fname[3:7]), int(fname[7:9])
    time_ls.append(datetime(year, month, 1))

    with open(filepath, 'rb') as fr:
        hdr = fr.read(300)
        ice = np.fromfile(fr, dtype=np.uint8)

    ice = ice.reshape(rows, cols)
    # convert from bytes
    ice = ma.masked_less(ice / 250., 0.)
    # mas out the land
    sic[:,:,i] = ma.masked_greater(ice, 1.0)
    """
    print("Saving figure ..")
    plt.ioff()
    fig, ax, m = st.spstere_plot(lon, lat, sic[:,:,i],
                       cbar_range, cmap, 
                       cbar_units, cbar_extend)
    lp = m.contour(lon, lat, ice, levels=[.15], 
              colors='crimson', linewidths=1., 
              latlon=True)
    lp_label = '15%'
    lp.collections[0].set_label(lp_label)
    ax.legend(loc='lower right', bbox_to_anchor=(.15, .85), fontsize=12)
    ax.annotate("%s/%s" % (year, month), 
      xycoords='axes fraction', xy=(.45, .5), 
      bbox=dict(facecolor='lavender',
                edgecolor='lavender',
                boxstyle='round'))
    fig.savefig(sicfigdir + 'sic_'+fname[3:9]+'.png')
    plt.close()
    """
    i += 1
print("sorting by time ..")
# sort by time
time = np.asarray(time_ls)
tsort_idx = np.argsort(time)
sic_sort = sic[:,:,tsort_idx]
time_sort = sorted(time)

ds = xr.Dataset({'sic':(('r', 'c', 'time'), sic_sort), 
                 'lon': (('r', 'c'), lon),
                 'lat': (('r', 'c'), lat)},
  coords={'r': np.arange(rows),
          'c' : np.arange(cols),
          'time' : time_sort})

newfilename = 'sic0_raw.nc'
ds.to_netcdf(icedir + newfilename)

print("File: %s \nsaved to: %s" % (newfilename, icedir))

"""
plt.ion()
fig = plt.figure(figsize=(9, 9))
ax = plt.axes(projection=ccrs.SouthPolarStereo(central_longitude=0))

cs = ax.coastlines(resolution='110m', linewidth=0.8)

ax.gridlines()
ax.set_extent([-180, 180, -90, -40], crs=ccrs.PlateCarree())

kw = dict(central_latitude=-90, central_longitude=0, 
  true_scale_latitude=70)
cs = ax.pcolormesh(x, y, ice, cmap=plt.cm.Blues,
                   transform=ccrs.Stereographic(**kw))


ii = 0

for filepath in glob.iglob(icedir+"nt_*.txt"):
    fname = os.path.basename(filepath)
    print(fname)

    # extract the year/month of the file
    year, month = int(fname[3:7]), int(fname[7:9])

    data_i = np.loadtxt(filepath)
    time_i = datetime(year, month, 1)
    if ii == 0 :
        data = data_i
        time = time_i
    else:
        data = np.dstack((data, data_i))
        time = np.hstack((time, time_i))
    ii += 1

# sort by time
tsort_idx = np.argsort(time)
data_sort = data[:,:,tsort_idx]
time_sort = sorted(time)

ice= np.transpose(data_sort, (1,0,2))
ice = ice / 250.
ice= ma.masked_greater(ice, 1.0)
"""