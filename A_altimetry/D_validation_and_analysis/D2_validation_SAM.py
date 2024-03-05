"""
SAM Index

Import SSHA data.
Compute correlation maps from the overlapping period 

Last modified: 4 June 2019
"""
# Import modules
import scipy.io as sio
import scipy.stats as ss
import scipy as sp

import numpy as np
from numpy import ma

import pandas as pd

from netCDF4 import Dataset

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm, ion
from matplotlib.offsetbox import AnchoredText 
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon as polyg
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap
from matplotlib import rcParams

from palettable.cartocolors.diverging import Tropic_7 

from scipy.stats.stats import pearsonr

import sys
import os

# Define directories
workdir = '/Users/ocd1n16/OneDrive - University of Southampton/PhD/data/'
sladir = workdir + 'SSH_SLA/'
lmdir = workdir + 'land_masks/'
coastdir = workdir + '/land_masks/holland_vic/'
figdir = workdir + '../data_notes/Figures_v4/'

# other useful variables
time_units = 'days since 1950-01-01 00:00:00.0'

# LAND MASK
#-------------------------------------------------------------------------------
# lon grid is -180/180, 0.5 lat x 1 lon
# lm shape=(mid_lon, mid_lat)
# land=1, ocean=0
#-------------------------------------------------------------------------------
lm = Dataset(lmdir+'land_mask_gridded_50S.nc', 'r+')
lmask = lm['landmask'][:]
lm.close()
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 1. SAM INDEX
# -----------------------------------------------------------------------------
# SAM index from
# https://climatedataguide.ucar.edu/climate-data/
# marshall-southern-annular-mode-sam-index-station-based 
# first row is aheader with month names (Jan-Dec)
filename = workdir+'sam_nerc_bas.txt'

# -----------------------------------------------------------------------------
# read lines and store only necessary columns in lists
# columns correspond to monthly index for Jan-Dec (0-12)
sam_years, sam_index = [], []
with open(filename,'r') as source:
    for line in source:
        a = line.split("\t")
        if a[0][0] != " ":  #remove headers
            aa = np.fromstring(a[0], dtype=float, sep='\t')
            sam_years.append(aa[0])
            sam_index.append(aa[1:])

sam_yrs = np.asarray(sam_years, dtype=int)
sam_ind = ma.ones((len(sam_yrs), 12))
for j in range(len(sam_yrs)):
    arr = np.asarray([item for item in sam_index[j]])
    ll = len(arr)
    sam_ind[j, :ll] = arr
    sam_ind[j, ll:] = ma.masked

# 1d array
sam_ind_1d = sam_ind.compressed()
sam_months = ma.ones(sam_ind.shape)*np.arange(1, 13)
sam_months[sam_ind.mask==True] = ma.masked
sam_months_1d = sam_months.compressed().astype(int)

sam_myrs = ma.ones(sam_ind.shape)*sam_yrs[:, np.newaxis]
sam_myrs[sam_ind.mask==True] = ma.masked
sam_yrs_1d = sam_myrs.compressed().astype(int)

# -----------------------------------------------------------------------------
# 2. ENVISAT + CS2
# -----------------------------------------------------------------------------
slafile = 'all_DOT_MDT_mean_20thresh_e0702_c0910.nc'
#slafile = 'all_DOT_MDT_mean_20thresh_e0702_c1110.nc'

dd = Dataset(sladir+slafile, 'r+') 

sla = ma.masked_invalid(dd['sla'][:])
elat = ma.masked_invalid(dd['edges_lat'][:])
elon = ma.masked_invalid(dd['edges_lon'][:])
time = ma.masked_invalid(dd['time'][:])
dd.close()

londim, latdim, timdim = sla.shape
glat, glon = np.meshgrid(elat, elon)

# -----------------------------------------------------------------------------
# 3. CORRELATION
# -----------------------------------------------------------------------------
# overlap period between SAM index and SSHA
# A. ALL DATA: 2002-07 to 2018-10
sam_ind_overlap = sam_ind_1d[45*12+6:-4]
sla_overlap = sla[:]
time_overlap = time[:]

# B. CS2 only: 2010-10 to 2018-10
#sam_ind_overlap = sam_ind_1d[53*12+9:-4]
#sla_overlap = sla[:, :, 1:]
#time_overlap = time[1:]

# DETREND DATA  
import func_trend as fc

print("Computing linear trends ..")
#------------------------------------------------------------------
slope, conf = [ma.ones((sla_overlap.shape[:-1])) for _ in range(2)]
for i in range(londim):
    for j in range(latdim):
        slope[i, j], conf[i, j] = fc.trend_ci(time_overlap[:], sla_overlap[i, j, :], 0.95)[1:]

total_days = time_overlap[-1] - time_overlap[0]
sla_trend_day = slope*total_days

sla_overlap -= sla_trend_day[:, :, np.newaxis]

slope, conf = fc.trend_ci(time_overlap, sam_ind_overlap, 0.95)[1:]
sam_trend_day = slope*total_days

sam_ind_overlap -= sam_trend_day

corr_sam = ma.zeros(sla.shape[:-1])
for i in range(londim):
    for j in range(latdim):
        sat_ts = ma.masked_invalid(sla_overlap[i, j, :])
        #get corr if more than 20% of the time there is data
        if sat_ts.count() > 0.2 * len(time_overlap):
            sam_ts = sam_ind_overlap[~sat_ts.mask]
            sat_ts = sat_ts[~sat_ts.mask]
            
            corr_sam[i, j] = pearsonr(sat_ts, sam_ts)[0]
        else:
            corr_sam[i, j] = ma.masked

# apply land mask
corr_sam[lmask==1] = ma.masked

#------------------------------------------------------------------
# 4. COASTLINES 
#------------------------------------------------------------------
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
#------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 5. PLOT
# -----------------------------------------------------------------------------
print("Preparing correlation map")
#figdir = input("Where to save the figure? ")
#figtitle = "Correlation map SSHA-SAM Index (2002-07 to 2018-10)"
savefigname='corr_SAM.png'

# rcParams
params = {
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    'figure.figsize': [9, 8]
}
rcParams.update(params)

# colormap
discrete_cmap = ListedColormap(Tropic_7.mpl_colors)
cont_cmap = Tropic_7.mpl_colormap

# figure area
ion()
fig, ax = plt.subplots()
cs = m.pcolormesh(glon, glat, corr_sam,
                  vmin=-1, vmax=1,
                  cmap=cont_cmap, #cm.PRGn_r,
                  latlon=True, 
                  rasterized=True,
                  zorder=1)
# colorbar
cbar = fig.colorbar(cs, ax=ax, shrink=0.5, pad=0.1)
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

#ax.annotate(' SAM Index \n 2002.07-2018.10',
#            xy=(.85, .18), xycoords='figure fraction',
#            ha='right', va='bottom')
ax.annotate(' SAM Index \n 2010.10-2018.10',
            xy=(.85, .18), xycoords='figure fraction',
            ha='right', va='bottom')


for k in range(len(ilon_land)):
    xf, yf = m(ilon_land[k], ilat_land[k])
    xyf = np.c_[xf, yf]
    poly = polyg(xyf, facecolor='w', 
                zorder=2, edgecolor='dimgrey',
                linewidth=0.5)
    ax.add_patch(poly)

for k in range(len(ilon_ice)):
    xf, yf = m(ilon_ice[k], ilat_ice[k])
    xyf = np.c_[xf, yf]
    poly = polyg(xyf, facecolor='lightgrey',
                zorder=2, edgecolor='dimgrey',
                linewidth=0.5)
    ax.add_patch(poly)

for k in range(len(lat_seg)):
    m.plot(lon_seg[k], lat_seg[k], 
           lw=0.5, latlon=True, 
           zorder=2, c='dimgrey')

# parallels and meridians
m.drawparallels(np.arange(-80., -50., 10), 
                zorder=3, linewdith=0.25, ax=ax)
m.drawmeridians(np.arange(0., 360., 30.), 
                zorder=3, labels=[1, 1, 1, 1],
                linewidth=0.25, ax=ax)
# don't clip the map boundary circle
circle = m.drawmapboundary(linewidth=1, color='k', ax=ax)
circle.set_clip_on(False)

x1, y1 = m(190, -80.5)
ax.annotate(r"$80^\circ$S", xy=(x1, y1), xycoords='data',
        xytext=(x1, y1),textcoords='data')
x2, y2 = m(186, -70.5)
ax.annotate(r"$70^\circ$S", xy=(x2, y2), xycoords='data',
        xytext=(x2, y2),textcoords='data')
x3, y3 = m(184, -60.5)
ax.annotate(r"$60^\circ$S", xy=(x3, y3), xycoords='data',
        xytext=(x3, y3),textcoords='data')

ax.set_rasterization_zorder(0)

# savefig
fig.savefig(figdir + savefigname, bbox_inches='tight')
print("Figure saved. Script done!")
