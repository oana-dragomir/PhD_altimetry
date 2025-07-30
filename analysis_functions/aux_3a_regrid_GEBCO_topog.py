"""
Coarsen the topography grid from GEBCO to match the altimetry resolution

Last modified: 30 May 2025 
"""
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator as rgi

# Directories
workdir = '/Vlumes/SamT5/PhD/PhD_data/'
altdir = workdir + 'altimetry_cpom/3_grid/'
topodir = workdir + 'topog/'

#-------------------------------------------------------------------
# bathymetry file
with xr.open_dataset(topodir + 'gebco_all.nc') as topo:
    print(topo.keys())

print("\n reading altimetry data..")
# --------------------------------------------------------
# use any altimetry file that has been gridded
altfile = #'insert_here_the_altimetry_file.nc'
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
