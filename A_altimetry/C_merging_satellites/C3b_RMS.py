"""
Compute RMS map and uncertainty estimate due to the intersatellite offset.

Intersat offset computed as area-weighted average of mean or median 
of SLA differences durign the overlap period between ENV-CS2:

[mean_30bin]   egm08   / goco05c
    > mean   = 3.37 cm / 3.40 cm
    > median = 3.87 cm / 3.89 cm

[median_30bin]
    > mean   = 3.41 cm / 3.44 cm
    > median = 3.86 cm / 3.88 cm

> apply the median-based offset because it reduces the rms of SLA residuals 
after correcting CS2 with the intersat offset

Last modified: 8 Apr 2021  
"""

import numpy as np
from numpy import ma

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import xarray as xr
import pandas as pd

import sys

#----------------------------------------------------------
# Intersatellite offset (median, metres)
#intersat_off = 0.038519 # for SLA (SSH-geoid)
#intersat_off = 0.041194 # for SSHA (SSH-MSS)
#intersat_off = 0.041465 #(4_)
#intersat_off = 0.037683 #(6_, median)

# Define directories
voldir = '/Volumes/SamT5/PhD/data/'
griddir = voldir + 'altimetry_cpom/3_grid_dot/'
figdir = voldir + '../PhD_figures/Figures_v8/'

scriptdir = '/Volumes/SamT5/PhD_scripts/'
auxscriptdir = scriptdir + 'scripts/aux_func/'


sys.path.append(auxscriptdir)
import aux_func_trend as fc
import aux_stereoplot as st

#----------------------------------------------------------
# # # # # # # # # # # # 
geoidtype = '_eigen6s4v2' #'_goco05c' #'_egm08' #'_goco05c'
statistics = 'median'
intersat_off = 0.0384
# # # # # # # # # # # # 

cs2_file = griddir + 'dot_cs2_30b' + statistics + geoidtype + '_sig3.nc'
env_file = griddir + 'dot_env_30b' + statistics + geoidtype + '_sig3.nc'

with xr.open_dataset(cs2_file) as cs2_dict:
    print(cs2_dict.keys())
with xr.open_dataset(env_file) as env_dict:
    print(env_dict.keys())

# overlap period
time = pd.date_range('2010-11-01', '2012-03-01', freq='1MS')

env = env_dict.sel(time=slice(time[0], time[-1]))
cs2 = cs2_dict.sel(time=slice(time[0], time[-1]))

print(env.keys())
print(cs2.keys())

# LAT/LON GRID
lat = env.latitude.values
lon = env.longitude.values
elat = env.edge_lat.values
elon = env.edge_lon.values

eglat, eglon = np.meshgrid(elat, elon)

lmask = env.land_mask.values

# DOT
cs2_dot = cs2.dot + intersat_off
env_dot = env.dot


# computing SLA relative to the overlap period mean SSH
#----------------------------------------------------------
cs2_mdt = cs2_dot.mean('time')
env_mdt = env_dot.mean('time')

cs2_sla = cs2_dot - env_mdt
env_sla = env_dot - env_mdt

# residuals
residuals = env_sla - cs2_sla
res_square = (residuals**2).sum('time')
n = len(residuals.time)
rms_map = np.sqrt(res_square/(n-1))


#----------------------------------------------------------
# area-weighted circumpolar SLA average - time series
area_grid = fc.grid_area(eglon, eglat)
area_grid[lmask==1] = np.nan # mask where land

# RMS at one point
aa = ma.masked_invalid(rms_map*area_grid)
total_area = ma.sum(area_grid[~aa.mask])
rms_pt = ma.sum(aa)/total_area

#----------------------------------------------------------
# don't display figures when running the script
#matplotlib.use('Agg')

savefigname = 'overlap_rms_intersat_bin.png'

cbar_range = [0, 6]
cbar_units = 'cm'
cmap = cm.get_cmap('YlGnBu', '6')

# # # # # # # # # # # # 
plt.ion()
fig, ax, m = st.spstere_plot(eglon, eglat, rms_map*1e2,
                cbar_range, cmap, cbar_units, 'both')
fig.tight_layout(rect=[0, -.1, 1, .95])

fig.suptitle('SLA RMS (cm) from overlap mission (11.2010-03.2012)')

ax.annotate('area-weighted \nRMS: %.3f cm' %(rms_pt*1e2), 
            xy=(0.04, 0.17),
            xycoords='figure fraction',
            color='indianred', weight='bold') 

