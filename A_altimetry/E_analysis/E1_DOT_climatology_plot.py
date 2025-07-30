"""
Plot the monthly climatology of DOT

Last modified: 8 Apr 2020
"""
import numpy as np
from numpy import ma
from netCDF4 import num2date

import xarray as xr
import pandas as pd
from scipy.stats import pearsonr
from scipy.interpolate import interp2d

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.dates as mdates
import matplotlib
matplotlib.use("Agg")

from shapely.geometry.polygon import Polygon
from shapely.geometry import Point 

from pycurrents.num import interp1

import func_trend as ft
import matlab_func as matf
import stereoplot as st 

import sys

# Directories
workdir = '/Users/ocd1n16/OneDrive - University of Southampton/PhD/data/'
topodir = workdir + 'topog/'
altdir = workdir + 'SSH_SLA/'
icedir = workdir + 'NSIDC/sic/'
lmdir = workdir + 'land_masks/'
figdir = workdir + '../data_notes/Figures_v7sla/'

#-------------------------------------------------------------------
# - - - - - - - - - - - - - - 
# REMOVE MONTHLY CLIMATOLOGY (i.e seasonal cycle)
# - - - - - - - - - - - - - - 
date_start_clim = '2010-03-01'
date_end_clim = '2015-02-28'
# - - - - - - - - - - - - - - 

#-----------------------------------
# bathymetry file
with xr.open_dataset(topodir + 'coarse_gebco_p5x1_latlon.nc') as topo:
    print(topo.keys())
tglat, tglon = np.meshgrid(topo.lat, topo.lon)
# --------------------------------------------------------
# SIC file; on the same grid as the altimetry
sic0 = xr.open_dataset(icedir + 'sic_grid_p5latx1lon.nc')
print(sic0.keys())

#corp to the same time period
sic = sic0.sel(time=slice(date_start, date_end))

# compute an average for every month from all years
sic_month = sic.groupby("time.month").mean()
sic_month_vals = sic_month.sic.values

# --------------------------------------------------------
print("\n reading altimetry data..")
altfile = 'v7_2_DOT_egm08_20thresh_intersat_bin_e0702_c1110_sig3.nc'

with xr.open_dataset(altdir+altfile) as alt:
    print(alt.keys())

# GRID coordinates
# at bin edges
alt_eglat, alt_eglon = np.meshgrid(alt.edge_lat, alt.edge_lon)
# at bin centres
alt_glat, alt_glon = np.meshgrid(alt.latitude, alt.longitude)
# bin edges for velocity plots
alt_gmlat, alt_gmlon = np.meshgrid(alt.mlat, alt.mlon)

# DIMENSIONS
alondim, alatdim, atimdim = alt.dot.shape
print("DOT dimensions: %s lon x %s lat x %s time" %(alondim, alatdim, atimdim))

# - - - - - - - - - - - - - - - - - 
# 1. crop dataset
dot = alt.dot.sel(time=slice(date_start, date_end))
alt_date = dot.time.dt.strftime("%m/%Y").values
fig_date = dot.time.dt.strftime("%Y%m").values

# time dimension
timdim = len(dot.time)

ndays = mdates.date2num(list(dot.time.values))
dt = ndays - ndays[0] # for computing linear trend

# - - - - - - - - - - - - - - - - - 
# compute monthly climatology
dot_clim_crop = dot.sel(time=slice(date_start_clim, date_end_clim))
dot_clim = dot.groupby("time.month").mean()
sla = dot.groupby("time.month") - dot_clim

sla_month = sla.groupby("time.month").mean().values

print("\n computing correlations and plotting ..\n")
# -----------------------------------------------------------------------------
#   r-map: SLA & rotated U (S1) 
# --------------------------------------------------------------------------
cbar_range = [-.1, .1]
cmap = cm.get_cmap('RdBu_r', 17)
cbar_units = 'r'
cbar_extend = 'both'

mnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


for k in range(1, 13):
    # variables for correlation plot
    var1 = sla_month[:,:,k-1]  # SLA

    print("Processing data for %s" % mnames[k-1])

    fig1, ax1, m1 = st.spstere_plot(alt_eglon, alt_eglat, var1,
                           cbar_range, cmap, cbar_units, cbar_extend)
    #bathymetry contours
    lp = m1.contour(tglon, tglat, topo.elevation,
              levels=[-1000],
              colors=['slategray', 'fuchsia'],
              latlon=True, zorder=2)
    lp_labels = ['1000 m']
    for i in range(len(lp_labels)):
        lp.collections[i].set_label(lp_labels[i])
    ax1.legend(loc=2, fontsize=9)

    # location of S1 and Holland box
    m1.scatter(s1_lon, s1_lat,
              marker='*', s=60, 
              latlon=True,
              c='gold', edgecolor='k', 
              lw=.5, zorder=7)

    m1.plot(hbox_lon, hbox_lat, latlon=True,
           c='navy', lw=1.5, ls='-', zorder=3)

    ax1.annotate("SLA\n%s\n%s"%(alt_date[0], alt_date[-1]),
                xy=(.94, .1),
                xycoords='figure fraction',
                ha='right', va='bottom',
               bbox=dict(facecolor='powderblue',
                         edgecolor='powderblue',
                         boxstyle='round'))
    ax1.annotate("%s" % mnames[k-1],
                xy=(.45, .5),
                xycoords='axes fraction',
               bbox=dict(facecolor='powderblue',
                         edgecolor='powderblue',
                         boxstyle='round'))

        #------------------------------------------------

    # add SIC 15%
    lp = m1.contour(alt_glon, alt_glat, sic_month_vals[k-1,:,:], levels=[.15], 
          colors='deeppink', linewidths=2., 
          latlon=True, zorder=4)
    lp_label = '15% SIC'
    lp.collections[0].set_label(lp_label)
    ax1.legend(loc='lower right',
            bbox_to_anchor=(.15, .85),
            fontsize=12)

    fig1.suptitle(mnames[k-1]+" SLA composite \n \
                  (linear trend not removed)")
    # - - - 
    fig_month = 'sla_composite_' + mnames[k-1] + '.png'
    fig1.savefig(figdir + fig_month, dpi=fig1.dpi*2)
    print("Figure %s saved in %s " % (fig_month, figdir))
    # - - - 
 