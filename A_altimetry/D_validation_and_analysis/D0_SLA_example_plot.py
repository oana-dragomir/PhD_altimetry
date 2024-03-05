"""
Plot a stereographic projection of DOT/SLA 

> needs some tweaking

Last modified: 22 Oct 2019
"""

import numpy as np
from numpy import ma

import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from palettable.colorbrewer.diverging import RdBu_11, PuOr_11

import sys

# Define directories
voldir = '/Volumes/SamT5/PhD/data/'
griddir = voldir + 'altimetry_cpom/3_grid_dot/'

localdir = '/Users/ocd1n16/PhD_local/'
auxscriptdir = localdir + 'scripts/aux_func/'
figdir = localdir + 'data_notes/Figures_v8/'

sys.path.append(auxscriptdir)
import aux_stereoplot as st

# --------------------------------------------------------
geoidtype = '_goco05c' #'_egm08' #'_goco05c'
statistics = 'median'

altfile = 'dot_all_30b' + statistics + geoidtype + '.nc'

# this function reads and displays the content of the file
with xr.open_dataset(griddir+altfile) as alt0:
    print(alt0.keys())

# crop dataset for a specified period
date_start = '2003-01-01'
date_end = '2018-01-01'
alt = alt0.sel(time=slice(date_start, date_end))

# GRID coordinates
# at bin edges (useful for pcolormesh plots for SSH)
alt_eglat, alt_eglon = np.meshgrid(alt.edge_lat,
                                  alt.edge_lon)

# --------------------------------------------------------
sla = alt.dot - alt.dot.mean('time')

# --------------------------------------------------------
savefigname = 'sla_example.png'
cmap = RdBu_11.mpl_colormap #cm.seismic
cbar_units = "SLA (cm)"
cbar_range = [-10, 10]
cbar_extend = 'both'

tim_idx = 10

plt.ion()
fig, ax, m = st.spstere_plot(alt_eglon, alt_eglat,
	sla.isel(time=tim_idx)*100, cbar_range, 
	cmap, cbar_units, cbar_extend)
ax.annotate(('{}').format(sla.time[tim_idx].dt.strftime("%m.%Y").values), 
            xy=(0.45, 0.5),
            xycoords='figure fraction',
            color='k', zorder=5, weight='bold') 

fig.savefig(figdir+savefigname, dpi=fig.dpi*5)
