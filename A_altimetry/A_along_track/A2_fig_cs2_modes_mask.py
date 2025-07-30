"""
Plot mode masks for CryoSat-2
Geographical mode masks v 3.8 downloaded from: 
https://earth.esa.int/eogateway/instruments/siral/description#geographical-mask-mode

Last modified: 11 March 2021
"""
import numpy as np
from numpy import ma

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

import shapefile
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# directories
maskdir = '/Volumes/SamT5/PhD_data/altimetry_cpom/CS2_mode_mask/'

# define map area
m = Basemap(projection='spstere', 
            lon_0 = -180, 
            boundinglat = -50.,
            resolution='l',
            round=True)

# Shapefile with CS2 mode masks
sf = shapefile.Reader(maskdir+"mask3_8")

# * attributes
print(sf.fields)

# * bounding box
print("bounding box: ", sf.bbox)

# * get indices of shapes south of -50
shapes = sf.shapes()

idx = []
for i in range(len(shapes)):
    lon1, lat1, lon2, lat2 = shapes[i].bbox
    if lat1<-50 and lat2<-50:
        idx.append(i)

idx = np.asarray(idx)

# * print fields of selected polygons
for i in idx:
    print(sf.record(i))

# extract a shape and plot it
k = 67
# 67 - box NW or the Ant Pen
shapeRec = sf.shapeRecord(k)
x_wap = [i[0] for i in shapeRec.shape.points[:]]
y_wap = [i[1] for i in shapeRec.shape.points[:]]


"""
def plot_shape(k, cl):
    shapeRec = sf.shapeRecord(k)
    x = [i[0] for i in shapeRec.shape.points[:]]
    y = [i[1] for i in shapeRec.shape.points[:]]
    m.plot(x, y, c=cl, latlon=True)
"""
### ----------------------------------------------------------
# LRM vs. SAR
# shapes 0-23 are are boundaries between open ocean vs. sea-ice
# every two pairs are the same shape, so pick either to use
# assume 0/1 = Jan, ..22/23 = Dec

# indices of SAR/LRM shapes
SARidx = np.arange(0, 23, 2)
SARlon, SARlat = [], []
for k in SARidx:
    SARshape = sf.shapeRecord(k)
    lon = [i[0] for i in SARshape.shape.points[:]]
    lat = [i[1] for i in SARshape.shape.points[:]]
    SARlat.append(lat)
    SARlon.append(lon)

# SEA-ICE vs COAST/ICE Sheet (SAR-SARin) boundary has index 24
sarin_shape = sf.shapeRecord(24)
sarin_lon = [i[0] for i in sarin_shape.shape.points[:]]
sarin_lat = [i[1] for i in sarin_shape.shape.points[:]]


params = {
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    'figure.figsize': [9, 8]
}

from matplotlib import rcParams
rcParams.update(params)

plt.ion()
fig, ax = plt.subplots()
m.drawcoastlines(color='dimgrey', linewidth=0.5)
m.fillcontinents(color='lightgrey', alpha=0.8)
m.plot(sarin_lon, sarin_lat, 
       c='coral', lw=1.5, ls='--',
       latlon=True, label='SARin')
m.plot(x_wap, y_wap, c='coral',
       lw=1.5, ls='--', latlon=True)
m.plot(SARlon[2], SARlat[2], 
       c='teal', lw=1.5,
       latlon=True, label='SAR-Mar')
m.plot(SARlon[8], SARlat[8], 
       c='teal', lw=1.5, ls=':',
       latlon=True, label='SAR-Sep')
ax.legend(loc=4, ncol=1, bbox_to_anchor=[-.05, .0, .3, .4], framealpha=1)

# don't clip the map boundary circle
circle = m.drawmapboundary(linewidth=1, color='k', ax=ax)
circle.set_clip_on(False)

ax.set_rasterization_zorder(0)

### ----------------------------------------------------------

# construct SAR polygon; it depends on the month of the data
dmonth = 11 # 0-11 (Jan-Dec)
sar_lon, sar_lat = SARlon[dmonth], SARlat[dmonth]
sar_x, sar_y = m(sar_lon, sar_lat)
sar_xy = np.column_stack((sar_x, sar_y))
sar_poly = Polygon(sar_xy)

# construct SARin polygon
sarin_x, sarin_y = m(sarin_lon, sarin_lat)
sarin_xy = np.column_stack((sarin_x, sarin_y))
sarin_poly = Polygon(sarin_xy)
