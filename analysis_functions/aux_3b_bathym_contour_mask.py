"""
Creates a mask to extract a variable inside a chosen
bathymetry contour.
>> the contour keeps its original resolution; or can find a way to make it coarser

Last modified: 21 June 2025
"""
import numpy as np
from numpy import ma

import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from shapely.geometry.polygon import Polygon
from shapely.geometry import Point 

import analysis_functions.aux_stereoplot as st

#-------------------------------------------------------------------
# Directories
#-------------------------------------------------------------------
voldir = '/Volumes/SamT5/PhD/PhD_data/'
griddir = voldir + 'altimetry_cpom/3_grid_dot/'
topodir = voldir + 'topog/'

#-------------------------------------------------------------------
# Use the altimetry file to get the lat-lon grid
#-------------------------------------------------------------------
geoidtype = 'goco05c'#'eigen6s4v2' # 'goco05c', 'egm08'
satellite = 'all'
sigma = 3

altfile = 'dot_' + satellite + '_30bmedian_' + geoidtype + '_sig' + str(sigma) + '.nc'

#-------------------------------------------------------------------
# load altimetry file
alt = xr.open_dataset(griddir+altfile) 

# GRID coordinates
# at bin edges
alt_eglat, alt_eglon = np.meshgrid(alt.edge_lat, alt.edge_lon)
# at bin centres
alt_glat, alt_glon = np.meshgrid(alt.latitude, alt.longitude)

#-----------------------------------
# bathymetry file
topo = xr.open_dataset(topodir + 'coarse_gebco_p5x1_latlon.nc')

tglat, tglon = np.meshgrid(topo.lat, topo.lon)
#-----------------------------------

cbar_range = [0, .1]
cmap = cm.get_cmap('bone_r', 11)
cbar_units = ''
contour_color = 'm'

# intialize and array
shelf = ma.zeros(alt_glat.shape)


# fig, ax = plt.subplots()
# cs = plt.contour(tglat, tglon, topo.elevation, levels=[-2000]).collections


# for contour in cs:
#     paths = contour.get_paths()
#     for path in paths:
#         vertices = path.vertices
#         for x, y in vertices:
#             print(f"({x:.2f}, {y:.2f})")

plt.ion()
fig, ax, m = st.spstere_plot(alt_eglon, alt_eglat, shelf,
                       cbar_range, cmap, cbar_units, contour_color)
#bathymetry contours
lp = m.contour(tglon, tglat, topo.elevation,
          levels=[-2000],
          colors=['slategray'],
          latlon=True, zorder=2)
lp_labels = ['2000 m']
for i in range(len(lp_labels)):
    lp.collections[i].set_label(lp_labels[i])
ax.legend(loc=2, fontsize=9)

#------------------------------------------------
# extract contour coords from bathym contour plot
lpsegs = lp.allsegs[0][:]

# find the longest contours - this marks the shelf break
mm, idx = 0, 0
for i in range(len(lpsegs)):
    a = lpsegs[i]
    if mm < len(a):
        mm = len(a)
        idx = i
#extract that contour (first one for 2000m)        
cc = lpsegs[idx]
#convert into geophys coords and crop the Ant Peninsula
llon, llat = m(cc[:, 0], cc[:,1], inverse=True)

# crop Antarctic peninsula
#llat[(llon>-85)&(llon<-61)] = -70
#llat[(llon>-61)&(llon<-20)] = -74

#------------------------------------------------
m.plot(llon, llat, c='c',latlon=True, lw=3)
#figname = 'shelf_contour_3000m.png'
#fig2.savefig(figdir+figname, dpi=fig2.dpi*2)

#------------------------------------------------
# create a polygon from the contour's lat/lon
llon_c = np.hstack((llon, llon[-1]))
llat_c = np.hstack((llat, llat[-1]))
xc, yc = m(llon_c, llat_c)
xy_c = np.column_stack((xc, yc))
xy_topog_poly = Polygon(xy_c)

# check what grid points are inside
xd, yd = m(alt_glon, alt_glat)
r, c = xd.shape

for i in range(r):
    for j in range(c):
        point = Point(xd[i, j], yd[i, j])
        #if alt.land_mask.values[i, j] == 0: # not on land
        if xy_topog_poly.contains(point):
            shelf[i, j] = 1
#shelf[alt.land_mask==1]=0
#shelf = ma.masked_equal(shelf, 1)

# values inside the contour but outside the land mask are 1; 
# everything else is 0
#------------------------------------------------
alt_glon[shelf!=1] = np.nan
alt_glat[shelf!=1] = np.nan

contour_lat = np.nanmax(alt_glat, axis=1)
contour_lon = alt.longitude.values
#------------------------------------------------
plt.ion()
fig, ax, m = st.spstere_plot(alt_eglon, alt_eglat, shelf,
                       cbar_range, cmap, cbar_units, contour_color)
#bathymetry contours
lp = m.plot(llon, llat, c= 'm',
         latlon=True, zorder=2, label='2000m')
ax.legend(loc=2, fontsize=9)


#------------------------------------------------

ds = xr.Dataset({'bathy_mask' : (('lon', 'lat'), shelf),
                'llon' : ('c', llon),
                'llat' : ('c', llat)},
                coords={'lon' : alt.longitude.values,
                'lat' : alt.latitude.values,
                'c' : np.arange(len(llon))})
ds.bathy_mask.attrs["comments"] = "1=ocean region inside a 2000m bathymetry contour (GEBCO), 0=discard/mask"
ds.clon.attrs["long_name"] = "lon of the bathy contour"
ds.clat.attrs["long_name"] = "lat of the bathy contour"

ds.to_netcdf(topodir + 'bathy_mask_2km.nc')

#------------------------------------------------
# check file
with xr.open_dataset(topodir + 'bathy_mask_2km.nc') as nds:
  print(nds.keys())
