"""
Plot SLA [ref to monthly climatology]

Add sea-ice edge (from nsidc)

Last modified: 8 Wed 2020

"""
## libraries go here
import numpy as np
from numpy import ma

from datetime import datetime
today = datetime.today()

import time as runtime
t_start = runtime.process_time()

from mpl_toolkits.basemap import Basemap

from netCDF4 import Dataset, num2date, date2num
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm, ion, ioff
from matplotlib.offsetbox import AnchoredText 
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon as polyg
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap
from matplotlib import rcParams

from palettable.cartocolors.diverging import Tropic_7 

import os
import sys


# Define directories
#-------------------------------------------------------------------------------
workdir = '/Users/ocd1n16/OneDrive - University of Southampton/PhD/data/'
coastdir = workdir + 'land_masks/holland_vic/'
topodir = workdir + 'topog/'
sshdir = workdir + 'SSH_SLA/'
icedir = workdir + 'NSIDC/sic/'
figdir = workdir + '../data_notes/Figures_v7sla/2002_2018/'

time_units = 'days since 1950-01-01 00:00:00.0'
s1_lon, s1_lat = -116.358, -72.468

sla_file = sshdir + 'v7_2_DOT_egm08_20thresh_intersat_bin_e0702_c1110_sig3.nc'

with xr.open_dataset(sla_file) as f1:
    print(f1.keys())

#grid
glat, glon = np.meshgrid(f1.edge_lat, f1.edge_lon)

start_date = '2002-07-01'
end_date = '2018-10-01'
#crop dataset
f2 = f1.sel(time=slice(start_date, end_date))

# time dimension
strtim = f2['time'].dt.strftime("%y%m").values
strdate = f2['time'].dt.strftime("%Y/%m").values
timdim = len(f2.time)

#sla 
mdt_crop = f2.dot.mean('time')
f2['sla_crop'] = f2['dot'] - mdt_crop
f2['mdt_crop'] = mdt_crop

# COASTLINES 
#------------------------------------------------------------------
m = Basemap(projection='spstere',
            boundinglat=-50.,
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

# bathymetry file
with xr.open_dataset(topodir + 'coarse_gebco_p5x1_latlon.nc') as topo:
    print(topo.keys())
topo_glat, topo_glon = np.meshgrid(topo.lat, topo.lon)

# SIC file; on the same grid as the altimetry
sic0 = xr.open_dataset(icedir + 'sic_raw.nc')
print(sic0.keys())

#corp to the same time period
sic = sic0.sel(time=slice(start_date, end_date))

# PLOT
# -----------------------------------------------------------------------------
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
cont_cmap = cm.RdBu_r

for k in range(timdim):
    suffix = strtim[k]
    savefigname='sla_'+suffix+'_MDT_0207_1810.png'
    print("working on %s ..." % savefigname)

    # figure area
    plt.ioff()
    fig, ax = plt.subplots()
    cs = m.pcolormesh(glon, glat, f2.sla_crop[:, :, k],
                      vmin=-.1, vmax=0.1,
                      cmap=cont_cmap,
                      latlon=True, 
                      rasterized=True,
                      zorder=1)
    # colorbar
    cbar = fig.colorbar(cs, ax=ax, extend='both', shrink=0.5, pad=0.1)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    # - - - - - - - - - - - - - - - - - - - 
    lp = m.contour(sic.lon.values, sic.lat.values,
                sic.sic.values[:,:,k], levels=[.15],
                colors='k', linewidths=1.5,
                latlon=True, zorder=4)
    lp_label = '15% SIC'
    lp.collections[0].set_label(lp_label)
    ax.legend(loc='lower right',
            bbox_to_anchor=(.17, .85),
            fontsize=12)
    # - - - - - - - - - - - - - - - - - - - 
    ax.annotate('ENV+CS2\n SLA (m)\n 03.2010-\n12.2015 mean',
                xy=(.9, .15), xycoords='figure fraction',
                ha='right', va='bottom')

    ax.annotate(strdate[k], xy=(.4, .5), 
                xycoords='figure fraction', weight='bold',
               bbox=dict(facecolor='lavender', edgecolor='lavender',
                         boxstyle='round'))

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
    # bathymetry contours
    lp = m.contour(topo_glon, topo_glat, 
                   topo.elevation.values, 
                   levels=[-3000, -1000],latlon=True, 
                    colors=['slategray', 'm'],
                    linewidths=1.2,
                    linestyles='-')
    lp_labels = ['3000 m', '1000 m']
    for i in range(len(lp_labels)):
        lp.collections[i].set_label(lp_labels[i])
    ax.legend(loc='lower right',
            bbox_to_anchor=(.17, .85), fontsize=14)
    # location of S1
    #m.scatter(s1_lon, s1_lat, marker='*', latlon=True,
    #           c='gold', edgecolor='k', lw=.5, zorder=5)
    
    ax.set_rasterization_zorder(0)
    fig.tight_layout()
    plt.close(fig)

    # savefig
    fig.savefig(figdir + savefigname, bbox_inches='tight')
print("Figure saved. Script done!")

t_stop = runtime.process_time()
print("execution time: %.3f s " %(t_stop-t_start))
