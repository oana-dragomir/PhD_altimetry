"""
Plot one track to show the offset between O/L 

plot the mean/std of every track for every month and the distribution of them

Last modified: 2 Sep 2020
"""

import numpy as np
from numpy import ma

from netCDF4 import Dataset, num2date

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon as polyg
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rcParams

import xarray as xr

import sys 
import os
import glob
# --------------------------------------------------------
# Directories
voldir = '/Volumes/SamT5/PhD_data/'
ncdir = voldir + 'altimetry_cpom/1_raw_nc/'
bindir = voldir + 'altimetry_cpom/2a_mean/'
lmdir = voldir + 'land_masks/'
icedir = voldir + 'NSIDC/sic/'
coastdir = voldir + 'land_masks/holland_vic/'

filepath = ncdir + 'month0508.nc'

localdir = '/Users/ocd1n16/PhD_local/'
scriptdir = localdir + 'scripts/'
figdir = localdir + 'data_notes/Figures_v8seq/'

# --------------------------------------------------------
time_units = 'days since 1950-01-01 00:00:00.0'

# --------------------------------------------------------
# SIC file; on the same grid as the altimetry
sic0 = xr.open_dataset(icedir + 'sic_raw.nc')
print(sic0.keys())

#crop to the same time period
sic = sic0.sel(time='2005-08-01')

# --------------------------------------------------------
f = Dataset(filepath, 'r+')
elev = ma.masked_invalid(f['Elevation'][:])
mss = ma.masked_invalid(f['MeanSSH'][:])
track_num = f['track_num'][:]
surf = f['SurfaceType'][:]
lat = f['Latitude'][:]
lon = f['Longitude'][:]
time = f['Time'][:][0]
f.close()

sla = ma.masked_outside(elev-mss, -3, 3)

date = num2date(time, units=time_units, calendar='gregorian')

# --------------------------------------------------------
m = Basemap(projection='spstere', 
			boundinglat=-49,
			lon_0=180,
			round=True)
# extract segments north of 60 S
print("Getting coastlines north of 60 S ... \n")
coast = m.drawcoastlines(linewidth=0)
segments = coast.get_segments()
lat_seg, lon_seg = [], []

for j in range(len(segments)):
    xy = np.vstack(segments[j]) 
    lons_b, lats_b = m(xy[:, 0], xy[:, 1], inverse=True)
    lats_bm = ma.masked_outside(lats_b, -60, -50.)
    lons_bm = ma.masked_array(lons_b, lats_bm.mask)
    if lats_bm.count() > 0:
        lat_seg.append(lats_bm)
        lon_seg.append(lons_bm)

# -----------------------
# coastlines south of 60S
print("Importing Antarctic Digital Database shapefile. \n")
# add path to coastline files
sys.path.append(coastdir)
from coastline_Antarctica import coastline

# extract coastline points from files (Tiago/Paul Holland)
## it returns a list of lists
[ilon_land, ilat_land, ilon_ice, ilat_ice] = coastline()

# --------------------------------------------------------
# extract one track
tn = 65
track_sla = sla[track_num==tn]
track_lat = lat[track_num==tn]
track_surf = surf[track_num==tn]
track_lon = lon[track_num==tn]
# ---------------------- PLOT 1 -------------------------------
fig, ax = plt.subplots(figsize=(9, 2))
ax.plot(track_lat[track_surf==1], 
		track_sla[track_surf==1], lw=1, c='lightblue')
ax.plot(track_lat[track_surf==2], 
		track_sla[track_surf==2], lw=1, c='royalblue')
plt.tight_layout()
ax.set_xlabel('Latitude')

# ---------------------- PLOT 2 ---------------------------
# rcParams
params = {
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    #'figure.figsize': [9, 8]
}
rcParams.update(params)

savefigname = 'single_track_1011.png'
# - - - - - - - - - - - - - - - - - - - 
fig, ax = plt.subplots()

m.scatter(track_lon[track_surf==1],
		track_lat[track_surf==1],
		latlon=True, s=1,
		c='lightblue')
m.scatter(track_lon[track_surf==2],
		track_lat[track_surf==2],
		latlon=True, s=1,
		c='royalblue')
#  - - - SIC  - - - - 
lp = m.contour(sic.lon.values, sic.lat.values,
			sic.sic.values, levels=[.15],
			colors='deeppink', linewidths=1.,
			latlon=True, zorder=4)
lp_label = '15% SIC'
lp.collections[0].set_label(lp_label)
ax.legend(loc='lower right',
        bbox_to_anchor=(.15, .85),
        fontsize=12)
# - - - - - - - - - - - - - - - - - - - 
ax.annotate(('{}').format(date.strftime("%m/%Y")), 
            xy=(0.45, 0.5),
            xycoords='figure fraction',
            color='k', zorder=5, weight='bold') 
# - - - - - - - - - - - - - - - - - - - 
for k in range(len(ilon_land)):
    xf, yf = m(ilon_land[k], ilat_land[k])
    xyf = np.c_[xf, yf]
    poly = polyg(xyf, facecolor='w', 
                zorder=4, edgecolor='dimgrey',
                linewidth=0.5)
    ax.add_patch(poly)

for k in range(len(ilon_ice)):
    xf, yf = m(ilon_ice[k], ilat_ice[k])
    xyf = np.c_[xf, yf]
    poly = polyg(xyf, facecolor='lightgrey',
                zorder=4, edgecolor='dimgrey',
                linewidth=0.5)
    ax.add_patch(poly)

for k in range(len(lat_seg)):
    m.plot(lon_seg[k], lat_seg[k], 
           lw=0.5, latlon=True, 
           zorder=4, c='dimgrey')
# - - - - - - - - - - - - - - - - - - - 
ax.set_rasterization_zorder(0)

# don't clip the map boundary circle
circle = m.drawmapboundary(linewidth=1, color='k')
circle.set_clip_on(False)

fig.tight_layout(rect=[0, 0, 1, 1])
# savefig
fig.savefig(figdir + savefigname, bbox_inches='tight',
            dpi=fig.dpi)

# # --------------------------------------------------------
# # --------------------------------------------------------
# #               MEAN/STD of every track
# # --------------------------------------------------------
# # --------------------------------------------------------
# for filepath in glob.iglob(ncdir+"*.nc"):

#     filename = os.path.basename(filepath)
#     print(filename)

#     f = Dataset(filepath, 'r+')
#     elev = ma.masked_invalid(f['Elevation'][:])
#     mss = ma.masked_invalid(f['MeanSSH'][:])
#     track_num = f['track_num'][:]
#     surf = f['SurfaceType'][:]
#     lat = f['Latitude'][:]
#     lon = f['Longitude'][:]
#     time = f['Time'][:][0]
#     f.close()

#     sla = elev - mss

#     date = num2date(time, units=time_units, calendar='gregorian')
#     # --------------------------------------------------------

#     # compute mean and STD of every track, regardless of OCEAN/lead step in sla
#     tr_mean, tr_std = [],[]
#     for tr in range(track_num.max()):
#         tr_sla = sla[track_num==tr]
#         tr_mean.append(tr_sla.mean())
#         tr_std.append(tr_sla.std(ddof=1))
#     fig, ax = plt.subplots()
#     ax.hist(tr_mean, bins=30, label='mean')
#     ax.hist(tr_std, fill=False, bins=30, label='std')
#     ax.legend()
#     fig.savefig(figdir + 'tr_hist_' + filename[:-3] + '.png')

#     fig, ax = plt.subplots(figsize=(10, 3))
#     ax.plot(tr_mean, c='k', label='mean')
#     ax.legend(loc=2)
#     ax2 = ax.twinx()
#     ax2.plot(tr_std, c='m', label='StDev')
#     ax2.legend(loc=1)
#     ax.set_xlabel("track number")
#     ax.set_title("%s/%s" % (date.year, date.month), loc='left')
#     plt.tight_layout()
#     fig.savefig(figdir + 'tr_stats_' + filename[:-3] + '.png')



