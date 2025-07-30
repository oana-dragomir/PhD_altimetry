"""
Script 2/2 Sea Ice Motion
- combine all yearly files into one

Last modified: 7 April 2020
"""

import numpy as np
import numpy.ma as ma

import xarray as xr

# Directories
workdir = '/Volumes/SamT5/PhD_data/'
icedir = workdir + 'NSIDC/ice_drift/'

yr = np.arange(2002, 2019)

for k in range(len(yr)):
	fname = 'SIvel_monthly_wgrid_p5x1_' + str(yr[k]) + '.nc'
	if k == 0:
		ice = xr.open_dataset(icedir + fname)
	else:

		ice_i = xr.open_dataset(icedir + fname)
		ice = xr.merge([ice, ice_i])

newfname = 'SIvel_monthly_wgrid_p5x1_all.nc'
ice.to_netcdf(icedir + newfname)

print("File %s /n saved to %s" % (newfname, icedir))