#! /usr/bin/env python3
"""
Plot coastline from the Antarctic Digital Database
> data from Clement Vic/Paul Holland

Last modified: 16 Mar 2021
"""

## Import modules
import numpy as np
from numpy import ma


import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as polyg

from mpl_toolkits.basemap import Basemap

import sys

# Define directories
#------------------------------------------------------------------
workdir = '/Volumes/SamT5/PhD/data/'
directorycoast = workdir + 'land_masks/holland_vic/'
figdir = workdir + '../data_notes/Figures/'


# COASTLINES
#------------------------------------------------------------------
#------------------------------------------------------------------
print("Defining map area ...\n")
# coastline resolution can vary from coarse to fine: c, l, i, h, f
m = Basemap(projection='spstere', 
            boundinglat=-49.5, 
            lon_0=-180, 
            resolution='f', 
            round=True)

print("extract segments north of 60 S from Basemap")

coast = m.drawcoastlines(linewidth=0)
segments = coast.get_segments()
lat_seg, lon_seg = [], []

for j in range(len(segments)):
    xy = np.vstack(segments[j]) 
    lons_b, lats_b = m(xy[:, 0], xy[:, 1], inverse=True)
    lats_bm = ma.masked_less(lats_b, -60)
    if lats_bm.count() > 0:
        lat_seg.append(lats_b)
        lon_seg.append(lons_b)

#------------------------------------------------------------------
print("Importing Antarctic Digital Database shapefile.. \n")
# add path to coastline files
sys.path.append(directorycoast)
from coastline_Antarctica import coastline


# extract coastline points from files (Tiago/Paul Holland)
## it returns a list of lists
[ilon_land, ilat_land, ilon_ice, ilat_ice] = coastline()

# combine both
ilon = np.hstack((ilon_land, ilon_ice))
ilat = np.hstack((ilat_land, ilat_ice))


plt.ion()
fig, ax = plt.subplots()

# map boundary
circle = m.drawmapboundary(color='k', linewidth=1)
circle.set_clip_on(False)

# ---- cropped Basemap region ----
for k in range(len(lat_seg)):
    xf, yf = m(lon_seg[k], lat_seg[k])
    m.plot(xf, yf, zorder=1, c='k', linewidth=0.25)

for k in range(len(ilon_land)):
    xf, yf = m(ilon_land[k], ilat_land[k])
    xyf = np.c_[xf, yf]
    poly = polyg(xyf, facecolor='lightgrey', edgecolor='k')
    plt.gca().add_patch(poly)

for k in range(len(ilon_ice)):
    xf, yf = m(ilon_ice[k], ilat_ice[k])
    xyf = np.c_[xf, yf]
    poly = polyg(xyf, facecolor='pink', edgecolor='k')
    plt.gca().add_patch(poly)


q = input("Save file? (Y/N): ")
if q =='Y':
    print('... saving file ...')
    fig.savefig(figdir + 'coastline_basemap_add.pdf',
                bbox_inches='tight')
else:
    print("Figure not saved.")


print('Script done!')
