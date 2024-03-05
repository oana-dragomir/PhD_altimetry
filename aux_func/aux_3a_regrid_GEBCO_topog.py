"""
coarsen the topography grid from GEBCO

Last modified: 16 Mar 2021 
"""
import numpy as np

import xarray as xr

from scipy.interpolate import RegularGridInterpolator as rgi

import sys

# Directories
workdir = '/Vlumes/SamT5/PhD/data/'
altdir = workdir + 'altimetry_cpom/3_grid/'
lmdir = workdir + 'land_masks/'
topodir = workdir + 'topog/'

#-------------------------------------------------------------------
# bathymetry file
with xr.open_dataset(topodir + 'gebco_all.nc') as topo:
    print(topo.keys())

print("\n reading altimetry data..")
# --------------------------------------------------------
altfile = 'v7_1_DOT_egm08_20thresh_intersat_bin_e0702_c1110_sig2.nc'
with xr.open_dataset(altdir+altfile) as alt:
    print(alt.keys())

# topog grid only covers -50 to -78
# define the new grid within those bounds otherwise the interp fc complains
fc = rgi((topo.lon.values, topo.lat.values), topo.elevation.values.T)

# grid - lon goes from 0 to 360 in topog 
alat = alt.latitude.sel(latitude=slice(topo.lat.min(), topo.lat.max())).values
alon = alt.longitude.values
alon[alon<0] = alon[alon<0] + 360

glat, glon = np.meshgrid(alat, alon)

# coarser topography
coarse_elev = fc((glon, glat))

# --------------------------------------------------------
# save coarser topog in a new file
coarse_topog = xr.Dataset({'elevation': (['lon', 'lat'], coarse_elev)},
                          coords={'lon': alon, 'lat': alat})
fname = 'coarse_gebco_p5x1_latlon.nc'
coarse_topog.to_netcdf(topodir+ fname)
print("File saved in %s as %s" % (topodir, fname))
