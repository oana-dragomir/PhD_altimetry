"""
Read binned ocean/lead files.
- discard bins that have less than 30 points
- compute and plot offset (O-L) - 3 plots side by side
- save figure 

Last modified: 10 Mar 2025
"""

# Import modules
import numpy as np
from numpy import ma

import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.basemap import Basemap

import sys
#-------------------------------------------------------------------
# Define directories
#-------------------------------------------------------------------
voldir = '/Volumes/SamT5/PhD/PhD_data/'
ncdir = voldir + 'altimetry_cpom/1_raw_nc/'
bindir = voldir + 'altimetry_cpom/2_grid_offset/'

scriptdir = '/Volumes/SamT5/PhD/PhD_scripts/ch2_amundsen/'
auxscriptdir = scriptdir + 'aux_func/'

 
sys.path.append(auxscriptdir)
import aux_stereoplot as st
#- - - - - - - - - - - - - -
#-------------------------------------------------------
# bin edges
edges_lon = np.linspace(-180, 180, num=361, endpoint=True)
edges_lat = np.linspace(-82, -50, num=65, endpoint=True)
eglat, eglon = np.meshgrid(edges_lat, edges_lon)

m = Basemap(projection='spstere', 
            boundinglat=-49.5, 
            lon_0=-180, 
            resolution='c', 
            round=True, 
            ellps='WGS84')
#-------------------------------------------------------

# # # # # # # # # # # # 
statistic = 'median'
satellite = 'env'
n_thresh = 30
# # # # # # # # # # # # 

print("- - - - - - - - - - - - - - ")
print("> > bin statistic: %s" % statistic)
print("> > satellite: %s" % satellite)
print("> > bin threshold: %s" % str(n_thresh))
print("- - - - - - - - - - - - - - \n")

#------------------------------------------------------------------
filename = 'b01_bin_ssha_OL_' + satellite + '_' + str(statistic) + '.nc'

with xr.open_dataset(bindir + filename) as bin0:
    print(bin0.keys)

bin0.ssha_o.values[bin0.npts_o<n_thresh] = np.nan
bin0.ssha_l.values[bin0.npts_l<n_thresh] = np.nan

#------------------------------------------------------------------
#                        STEREOGRAPHIC PLOT
print('\n PLOT: Stereographic projection of SLA - Ocean and Lead \n')
#------------------------------------------------------------------
# variables to plot
tim_idx = 17

var_O = bin0.ssha_o.isel(time=tim_idx)
var_L = bin0.ssha_l.isel(time=tim_idx)
var_dif = var_O - var_L

tim_month = bin0.time[tim_idx].dt.month.values
tim_year = bin0.time[tim_idx].dt.year.values

xx, yy = eglon, eglat

cmap = cm.seismic


plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(15, 6), ncols=3)

ax1.set_title("Ocean SLA (m)")
ax2.set_title("Lead SLA (m)")
ax3.set_title("Offset (O-L)")

# pcolormesh: 2D array has x=rows, y=cols; matches with the grid
cs1 = m.pcolormesh(xx, yy, var_L, 
                   cmap=cmap, 
                   latlon=True, zorder=2,
                   vmin=-0.5, vmax=0.5, 
                   rasterized=True, ax=ax1)
m.drawcoastlines(linewidth=0.25, zorder=4, ax=ax1)
m.drawparallels(np.arange(-80, -40, 10), 
                zorder=3, linewdith=0.25, ax=ax1)
m.drawmeridians(np.arange(30., 360., 30.), zorder=3,
               labels=[1, 1, 1, 1], linewidth=0.25, ax=ax1)
plt.colorbar(cs1, ax=ax1, orientation='horizontal', 
             extend='both', shrink=0.7)
# ----- #
cs2 = m.pcolormesh(xx, yy, var_O, 
                   latlon=True,
                   cmap=cmap, 
                   zorder=2, vmin=-0.5, vmax=0.5, 
                   rasterized=True, ax=ax2)
m.drawcoastlines(linewidth=0.25, zorder=4, ax=ax2)
m.drawparallels(np.arange(-80, -40, 10), 
                zorder=3, linewdith=0.25, ax=ax2)
m.drawmeridians(np.arange(30., 360., 30.), zorder=3,
               labels=[1, 1, 1, 1], linewidth=0.25, ax=ax2)
plt.colorbar(cs2, ax=ax2, orientation='horizontal', 
             extend='both', shrink=0.7)

# ----- #
cs3 = m.pcolormesh(xx, yy, var_dif, 
                   latlon=True,
                   cmap=cmap, 
                   zorder=2, 
                   vmin=-0.5, vmax=0.5, 
                   rasterized=True, ax=ax3)
m.drawcoastlines(linewidth=0.25, zorder=4, ax=ax3)
m.drawparallels(np.arange(-80., -40., 10), 
                zorder=3, linewdith=0.25, ax=ax3)
m.drawmeridians(np.arange(30., 360., 30.), zorder=3,
               labels=[1, 1, 1, 1], linewidth=0.25, ax=ax3)
plt.colorbar(cs3, ax=ax3, orientation='horizontal', 
             extend='both', shrink=0.7)

for ax in (ax1, ax2, ax3):
    # annotate parallels
    x1, y1 = m(180, -80)
    ax.annotate(r"$80^\circ S$", xy=(x1, y1), xycoords='data',
            xytext=(x1, y1),textcoords='data')
    x2, y2 = m(180, -70)
    ax.annotate(r"$70^\circ S$", xy=(x2, y2), xycoords='data',
            xytext=(x2, y2),textcoords='data')
    x3, y3 = m(180, -60)
    ax.annotate(r"$60^\circ S$", xy=(x3, y3), xycoords='data',
            xytext=(x3, y3),textcoords='data')
    x4, y4 = m(180, -50)
    ax.annotate(r"$50^\circ S$", xy=(x4, y4), xycoords='data',
            xytext=(x4, y4),textcoords='data')
    ax.set_rasterization_zorder(0)

fig.suptitle(('{}{}{}{}{}').format('M/Y: ', 
                                  tim_month, '/', 
                                  tim_year, '\n'))
