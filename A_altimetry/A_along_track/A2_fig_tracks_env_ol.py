"""
Envisat

Plot along-track data split into Ocean-Leads
and add the 15% SIC contour

Last modified: 23 Mar 2023
"""

import numpy as np
from numpy import ma

from netCDF4 import Dataset, num2date

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon as polyg
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rcParams
import matplotlib

import datetime

import sys

import xarray as xr

workdir = '/Volumes/SamT5/PhD_data/'
icedir = workdir + 'NSIDC/sic/'
coastdir = workdir + 'land_masks/holland_vic/'
ncdir = workdir + 'altimetry_cpom/1_raw_nc/'

localdir = '/Volumes/SamT5/PhD_scripts/'
auxscriptdir = localdir + 'aux_func/'
sys.path.append(auxscriptdir)
from aux_1_filenames import env_id_list

time_units = 'days since 1950-01-01 00:00:00.0'
# --------------------------------------------------------
# --------------------------------------------------------
m = Basemap(projection='spstere',
            boundinglat=-52.,
            lon_0 = -180,
            resolution='i',
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
# SIC file; on the same grid as the altimetry
sic0 = xr.open_dataset(icedir + 'sic_raw.nc')
print(sic0.keys())

#corp to the same time period
sic = sic0.sel(time='2012-02-01')
# --------------------------------------------------------
i = 115

filename = ncdir + env_id_list[i] + '.nc'

ds = Dataset(filename, 'r+')
surf = ds['SurfaceType'][:]
time = ds['Time'][:][0]
lat = ds['Latitude'][:]
lon = ds['Longitude'][:]
dist = ds['distance_m'][:]
ds.close()

lon = lon[dist>1e4]
lat = lat[dist>1e4]
surf = surf[dist>1e4]

# split into O-L
# --------------------------------------------------------
o_lat = lat[surf==1]
o_lon = lon[surf==1]

l_lat = lat[surf==2]
l_lon = lon[surf==2]

date = num2date(time, units=time_units, calendar='gregorian')
# --------------------------------------------------------
# --------------------------------------------------------
savefigname = env_id_list[i] + '.png'
print("plotting %s" % savefigname)
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
# - - - - - - - - - - - - - - - - - - - 
# figure area
plt.ion()
fig, ax = plt.subplots(figsize=(6.5, 6.5))

m.scatter(o_lon, o_lat,
          c='lightblue', s=0.01,
          latlon=True,
          rasterized=True,
          zorder=2)
m.scatter(l_lon, l_lat,
          c='royalblue', s=0.01,
          latlon=True,
          rasterized=True,
          zorder=3)
# - - - - - - - - - - - - - - - - - - - 
ax.annotate(('{}').format(date.strftime("%m/%Y")), 
            xy=(0.45, 0.5),
            xycoords='figure fraction',
            color='k', zorder=5, weight='bold') 
#  - - - SIC  - - - - 
lp = m.contour(sic.lon.values, sic.lat.values,
            sic.sic.values, levels=[.15],
            colors='deeppink', linewidths=2.,
            latlon=True, zorder=4)
lp_label = '15% SIC'
lp.collections[0].set_label(lp_label)
ax.legend(loc='lower right',
        bbox_to_anchor=(.15, .85),
        fontsize=12)
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
# parallels and meridians
m.drawparallels(np.arange(-80., -50., 10), 
                zorder=10, linewdith=0.25, ax=ax)
m.drawmeridians(np.arange(0., 360., 30.), 
                zorder=10, labels=[1, 1, 1, 1],
                linewidth=0.25, ax=ax)

x1, y1 = m(190, -80.5)
ax.annotate(r"$80^\circ$S", xy=(x1, y1),
            xycoords='data', xytext=(x1, y1),
            textcoords='data', zorder=10)
x2, y2 = m(186, -70.5)
ax.annotate(r"$70^\circ$S", xy=(x2, y2),
            xycoords='data', xytext=(x2, y2),
            textcoords='data', zorder=10)
x3, y3 = m(184, -60.5)
ax.annotate(r"$60^\circ$S", xy=(x3, y3),
            xycoords='data', xytext=(x3, y3),
            textcoords='data', zorder=10)
# - - - - - - - - - - - - - - - - - - - 
ax.set_rasterization_zorder(0)

# don't clip the map boundary circle
circle = m.drawmapboundary(linewidth=1, color='k')
circle.set_clip_on(False)

fig.tight_layout(rect=[0, 0, 1, 1])
# savefig
fig.savefig(figdir + savefigname, bbox_inches='tight',
            dpi=fig.dpi*5)
