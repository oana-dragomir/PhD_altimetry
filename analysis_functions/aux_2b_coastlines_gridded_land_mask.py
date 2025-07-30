"""
>> this script creates the files that are later used to: 
    (1) calculate the distance between pointwise data and nearest land contour and 
    (2) mask the land in the gridded data 

1. I define a coarse contour to the north of 60S that roughly follows the 
South American coast. This is because I couldn't figure out how to extract 
it from Basemap (this could be revised). 
To the south of 60S, I use the contours of coastline and ice shelves 
from the Vic/Holland product.
N.B.: Given the time, there could be a test to see if this step makes 
any difference at all.

2. Create a gridded land mask to use with the binned data.
[N.B.: This is not the code that created the land_mask_gridded_50s.nc file!!
I seem to have lost it but this should not make much of a difference and the 
approach to obtain it must have been along these lines. The file is more detailed 
than the result I get from the script so I would use the file or make your own land mask]

- if any of the corners of a grid cell are inside a land/ice shelf contour then
the grid cell is masked
- there is no set distance from the coast that is masked

Last modified: 12 Nov 2024
"""
# Import modules

import numpy as np
from numpy import ma

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, cm, ion 
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon as polyg

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from copy import deepcopy
import sys

from netCDF4 import Dataset

# Define directories
workdir = '/Volumes/SamT5/PhD/PhD_data/'
lmdir = workdir +  'land_masks/'
directorycoast = lmdir + 'holland_vic/'

# Define a GRID that matches the altimetry grid
#--------------------------------------------------------------
edges_lon = np.linspace(-180, 180, num=361, endpoint=True)
edges_lat = np.linspace(-82, -50, num=65, endpoint=True)

glat, glon = np.meshgrid(edges_lat, edges_lon)

mid_lon = 0.5*(edges_lon[1:] + edges_lon[:-1])
mid_lat = 0.5*(edges_lat[1:] + edges_lat[:-1])

gmlat, gmlon = np.meshgrid(mid_lat, mid_lon)
mlondim, mlatdim = gmlon.shape
londim, latdim = glon.shape

# grid of 0s that will have 1s where there is land
grid0 = ma.zeros(glon.shape)
mgrid0 = ma.zeros(gmlon.shape)

#--------------------------------------------------------------
#------------------ Land contour north of 60S -----------------
#--------------------------------------------------------------
# hardcode a contour around South America and the islands between 50S-60S
chile_lon = np.asarray([-75.60, -75.50, -75.63, -75.60, -75.48, -75.40, 
			-75.58, -75.55, -75.38, -75.30, -75.37, -75.40, 
			-75.38, -75.49, -75.48, -75.49, -75.52, -75.47, 
			-75.30, -75.06, -75.10, -75.15, -75.24, -75.22, 
			-75.31, -75.34, -75.26, -75.10, -75.14, -75.12, 
			-75.01, -74.93, -74.50, -74.68, -74.75, -74.40, 
			-74.33, -74.15, -73.93, -73.91, -73.87, -73.73, 
			-73.58, -73.33, -73.41, -73.39, -73.50, -73.27, 
			-73.22, -72.14, -72.14, -71.97, -71.44, -71.48, 
                        -70.92, -70.45, -70.16, -69.20, -68.29, -68.10, 
			-67.88, -67.88, -67.60, -67.20, -67.00, -67.15,
			-66.90, -66.40, -66.50, -65.30, -65.25, -64.80, 
			-64.70, -64.20, -63.70, -63.70, -64.30, -64.80,
			-65.25, -65.10, -66.00, -67.00, -68.10, -68.25,
			-68.75, -68.35, -68.95, -69.15, -69.10, -68.78,
			-68.15, -67.90, -67.70, -67.65, -67.50, -75.60])

chile_lat = np.asarray([-49.00, -49.08, -49.22, -49.25, -49.28, -49.55,
			-49.66, -49.85, -49.78, -49.90, -50.01, -50.04,
			-50.13, -50.31, -50.41, -50.60, -50.67, -50.78, 
			-50.82, -50.85, -51.03, -51.25, -51.35, -51.50, 
			-51.55, -51.63, -51.67, -51.74, -51.82, -51.92, 
			-52.15, -52.30, -52.65, -52.72, -52.76, -53.11, 
			-53.31, -53.39, -53.31, -53.55, -53.63, -53.73, 
			-53.79, -53.75, -53.85, -53.94, -54.07, -54.18, 
			-54.14, -54.60, -54.66, -54.78, -54.80, -54.93, 
                        -55.16, -55.21, -55.33, -55.63, -55.63, -55.72, 
			-55.83, -55.87, -55.92, -56.00, -55.86, -55.75,
			-55.35, -55.25, -55.05, -54.90, -54.80, -54.80, 
			-54.90, -54.85, -54.78, -54.65, -54.58, -54.80,
			-54.80, -54.60, -54.60, -54.16, -53.40, -52.98,
			-52.57, -52.35, -51.55, -51.00, -50.60, -50.30,
			-50.12, -50.00, -49.80, -49.50, -49.00, -49.00])
# Falklands/Malvinas
isl1_lon = np.asarray([-61.40, -61.40, -61.10, -60.93, -60.65, -60.45,
			-60.18, -59.90, -59.85, -59.70, -59.30, -58.4, 
			-57.66, -57.66, -57.90, -59.00, -59.15, -59.90,
			-61.25, -61.26, -60.80, -60.9, -61.4])
isl1_lat = np.asarray([-51.65, -51.82, -52.06, -52.18, -52.27, -52.25, 
			-52.05, -52.05, -52.37, -52.41, -52.37, -52.13, 
			-51.70, -51.50, -51.33, -51.23, -51.36, -51.19,
			-50.99, -51.03, -51.31, -51.63, -51.65])

isl2_lon = np.asarray([-38.10, -37.40, -36.15, -35.70, -36.20, -36.90, -38.10])
isl2_lat = np.asarray([-54.00, -54.30, -54.95, -54.80, -54.20, -54.00, -54.00])

isl3_lon = [-26.80, -27.70, -27.00, -26.30, -26.20, -26.20, -26.50, -27.40, -27.80, -27.20, -26.80]
isl3_lat = [-58.00, -59.50, -59.60, -59.00, -58.40, -57.70, -57.00, -56.20, -56.20, -57.00, -58.00]

isl4_lon = [3.5, 3.0, 3.5, 3.8, 3.5]
isl4_lat = [-54.6, -54.4, -54.2, -54.4, -54.6]

isl5_lon = [73.90, 73.30, 73.20, 73.50, 73.90]
isl5_lat = [-53.10, -52.95, -53.00, -53.22, -53.10]

#--------------------------------------------------------------
# ALL contours north of 60S
ilon_basemap, ilat_basemap = [], []
for llo, lla in zip([chile_lon, isl1_lon, isl2_lon, isl3_lon, isl4_lon, isl5_lon],
                    [chile_lat, isl1_lat, isl2_lat, isl3_lat, isl4_lat, isl5_lat]):
    ilon_basemap.append(llo)
    ilat_basemap.append(lla)

#--------------------------------------------------------------
#------------------ COASTLINES south of 60S -----------------
#--------------------------------------------------------------
print("Importing Antarctic Digital Database shapefile. \n")
# add path to coastline files
sys.path.append(directorycoast)
from coastline_Antarctica import coastline

# extract coastline points from files (Tiago/Paul Holland)
## it returns a list of lists
[ilon_land, ilat_land, ilon_ice, ilat_ice] = coastline()

# combine contours from both land and ice shelves
ilon = np.hstack((ilon_land, ilon_ice, ilon_basemap))
ilat = np.hstack((ilat_land, ilat_ice, ilat_basemap))

#--------------------------------------------------------------
# 1. file used to calculate distance to land 
#--------------------------------------------------------------
coast = np.vstack((ilon, ilat))

 with open(lmdir+'coastline_nested_lists.pkl', 'wb') as f:
    pickle.dump(coast, f)
#--------------------------------------------------------------
#--------------------------------------------------------------
#--------------------------------------------------------------
# 2. Gridded land mask  
#--------------------------------------------------------------
# pick land contour 10 (biggest area, main coastline):
clon = np.asarray([float(item) for item in ilon[10]])
clat = np.asarray([float(item) for item in ilat[10]])

# contour
lonlat_arr = np.column_stack((clon, clat))
polygon = Polygon(lonlat_arr)

# if any of the corners of a grid cell is inside a contour then mask the cell
for lo in range(1, londim-1):
    for la in range(1, latdim-1):
        point = Point(glon[lo, la], glat[lo, la])
        if polygon.contains(point) == True:
            mgrid0[lo-1, la-1] = mgrid0[lo-1, la] = 1
            mgrid0[lo, la-1] = mgrid0[lo, la] = 1


# indices along the lon/lat dimensions
ind_lon = np.arange(len(edges_lon))[:, np.newaxis]*ma.ones((glon.shape))
ind_lat = np.arange(len(edges_lat))*ma.ones((glat.shape))

ind_lon = ind_lon.flatten()
ind_lat = ind_lat.flatten()

# data lon are -180/180;
# need to change to +ve lons for the contour that crosses the 180/-180 meridian
glon1 = glon.flatten()
glon_pos = deepcopy(glon1)
glon_pos[glon_pos<0] = glon_pos[glon_pos<0] + 360

fglat = glat.flatten()

imask_lo, imask_la = [], []
# contours info:
# land: 10 = largest one, main grounding line - analyse separately because it
# needs to be broken down into smaller segments
# ice (+land, +208): 45 (253) = Ronne, 95 (303) = Ross
for i in range(len(ilon)):
    if i != 10:  # this contour will be dealt with separately
        # 1. contour lat/lon
        clon = np.asarray([float(item) for item in ilon[i]])
        clat = np.asarray([float(item) for item in ilat[i]])
    
        if i == 303: # Ross Sea (across the discontinuity -180/180) 
            clon[clon<0] = clon[clon<0] + 360
            fglon = glon_pos
        else:
            fglon = glon1
        
        # 3. create a polygon
        lonlat = np.column_stack((clon, clat))
        polygon = Polygon(lonlat)

        for j in range(len(ind_lon)):
            point = Point(fglon[j], fglat[j])
            if polygon.contains(point) == True:
                imask_lo.append(ind_lon[j])
                imask_la.append(ind_lat[j])

iilo = np.asarray([int(imask_lo[i]) for i in range(len(imask_lo))])
iila = np.asarray([int(imask_la[i]) for i in range(len(imask_la))])

ilo0 = ma. masked_outside(iilo, 1, 359)
ila0 = ma.masked_array(iila, ilo0.mask)
ila = ma.masked_outside(ila0, 1, 59)
ilo = ilo0[~ila.mask]
ila = ila[~ila.mask]

mgrid0[ilo, ila] = 1
mgrid0[ilo-1, ila-1] = 1
mgrid0[ilo-1, ila] = 1
mgrid0[ilo, ila-1] = 1

# save mask to a file /// commented this out so as not to replace the file I used
#newfile = 'land_mask_gridded_50s_v2.nc' # edited the name just in case
dataset = Dataset(lmdir+newfile, 'w')

# dimensions
dataset.createDimension('mlon', mlondim)
dataset.createDimension('mlat', mlatdim)
dataset.createDimension('elon', londim)
dataset.createDimension('elat', latdim)

# variables
landmask = dataset.createVariable('landmask', np.float64, ('mlon', 'mlat'))
mlati = dataset.createVariable('mid_latitude', np.float64, ('mlon', 'mlat'))
mlongi = dataset.createVariable('mid_longitude', np.float64, ('mlon', 'mlat'))
elati = dataset.createVariable('edge_latitude', np.float64, ('elon', 'elat'))
elongi = dataset.createVariable('edge_longitude', np.float64, ('elon', 'elat'))

dataset.description = ("Gridded land mask south of 50S (land=1, ocean=0). (0.5 lat x 1 lon)")
landmask[:] = mgrid0
mlati[:] = gmlon
mlongi[:] = gmlat
elati[:] = glat
elongi[:] = glon

dataset.close()

#--------------------------------------------------------------
# PLOT 
#--------------------------------------------------------------
# coastline resolution can vary from coarse to fine: c, l, i, h, f
m = Basemap(projection='spstere', 
            boundinglat=-50., 
            lon_0=180,
            round=True,
            resolution='h',
            ellps='WGS84')

basemap_coast = m.drawcoastlines(linewidth=0)
segments = basemap_coast.get_segments()
lat_seg, lon_seg = [], []

# get only contours to the north of 60S from basemap
for j in range(len(segments)):
    xy = np.vstack(segments[j]) 
    lons_b, lats_b = m(xy[:, 0], xy[:, 1], inverse=True)
    lats_bm = ma.masked_less(lats_b, -60)
    if lats_bm.count() > 0:
        lat_seg.append(lats_b)
        lon_seg.append(lons_b)

# --- begin figure
fig, ax = plt.subplots()
for k in range(len(ilon_land)):
    xf, yf = m(ilon_land[k], ilat_land[k])
    xyf = np.c_[xf, yf]
    poly = polyg(xyf, facecolor='lightgrey', 
                  zorder=3, edgecolor='k')
    ax.add_patch(poly)

for k in range(len(ilon_ice)):
    xf, yf = m(ilon_ice[k], ilat_ice[k])
    xyf = np.c_[xf, yf]
    poly = polyg(xyf, facecolor='pink',
                  zorder=3, edgecolor='k')
    ax.add_patch(poly)

# ---- cropped Basemap region ----
for k in range(len(lat_seg)):
    xf, yf = m(lon_seg[k], lat_seg[k])
    m.plot(xf, yf, zorder=3, c='k')

m.pcolormesh(glon, glat, mgrid0, alpha=0.6, cmap=cm.Blues, latlon=True, zorder=4)
m.drawparallels([-50, -60])
#m.scatter(glon, glat, c='grey', s=5, latlon=True, zorder=4)
#m.scatter(gmlon, gmlat, c='c', s=5, latlon=True, zorder=4)

