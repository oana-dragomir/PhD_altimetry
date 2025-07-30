"""
Plot all along tracks SSHA on a stereographic projection 
to check for any major gaps/unusual spread
- not using the land contour

Last modified: 10 Mar 2025
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

# don't display figures when running the script
matplotlib.use('Agg')

workdir = '/home/doc58/Documents/PhD_data/'
ncdir = workdir + 'nc_data/'
coastdir = workdir + 'land_masks/holland_vic/'
figdir = workdir + '../PhD_figures/Figures_v4/'

# filenames
filenamespath = '/Volumes/SamT5/PhD_scripts/'
sys.path.append(filenamespath)
from aux_1_filenames import env_id_list as id_list

time_units = 'days since 1950-01-01 00:00:00.0'

savefig = False
# --------------------------------------------------------
m = Basemap(projection='spstere',
            boundinglat=-52.,
            lon_0 = -180,
            resolution='i',
            round=True)

itt = len(id_list)
for i in range(itt):
    filename = ncdir + id_list[i] + '.nc'
    savefigname = id_list[i] + '.png'

    ds = Dataset(filename, 'r+')
    lat = ds['Latitude'][:]
    lon = ds['Longitude'][:]
    time = ds['Time'][:][0]
    ds.close()

    fig, ax = plt.subplots()
    m.scatter(lon, lat,
              s=0.1, c='grey',
              latlon=True)

    date = num2date(time, units=time_units, calendar='gregorian')
    fig.suptitle(('{}').format(date.strftime("%m/%Y")))
    
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

    ax.set_rasterization_zorder(0)

    # don't clip the map boundary circle
    circle = m.drawmapboundary(linewidth=1, color='k')
    circle.set_clip_on(False)

    # savefig
    if savefig:
        fig.savefig(figdir + savefigname, bbox_inches='tight',
                    dpi=fig.dpi*3)
