"""
Plot of O/L offset climatology and Stdev 
> for each satellite compare the offset when 
using mean vs median as the bin statistic

Method: for every month we compute an area-weighted average offset
then we average the monthly means separately for every month 
to obtain a climatology.
StDev is based on the spread of the area-weighted means over every month.

- save climatology in a file

Last modified: 10 Mar 2025
"""

## Import modules
import numpy as np
from numpy import ma

import matplotlib.pyplot as plt 

import xarray as xr
import pandas as pd

import sys

#-------------------------------------------------------------------
# Define directories
#-------------------------------------------------------------------
voldir = '/Volumes/SamT5/PhD/PhD_data/'
ncdir = voldir + 'altimetry_cpom/1_raw_nc/'
bindir = voldir + 'altimetry_cpom/2_grid_offset/'

scriptdir = '/Volumes/SamT5/PhD/ch2_altimetry/PhD_scripts/'
auxscriptdir = scriptdir + 'aux_func/'

sys.path.append(auxscriptdir)
import aux_func as ft

#-------------------------------------------------------
# bin edges
edges_lon = np.linspace(-180, 180, num=361, endpoint=True)
edges_lat = np.linspace(-82, -50, num=65, endpoint=True)
eglat, eglon = np.meshgrid(edges_lat, edges_lon)

total_area = ft.grid_area(eglon, eglat)
#-------------------------------------------------------
 
def mclim(statistic, satellite, n_thresh):
	print("- - - - - - - - - - - - - - ")
	print("> > bin statistic: %s" % statistic)
	print("> > satellite: %s" % satellite)
	print("> > bin threshold: %s" % str(n_thresh))
	print("- - - - - - - - - - - - - - \n")

	#------------------------------------------------------------------
	filename = 'b01_bin_ssha_OL_' + satellite + '_' + str(statistic) + '.nc'
	print("File: %s" % filename)

	with xr.open_dataset(bindir + filename) as bin0:
	    print(bin0.keys)

	#------------------------------------------------------------------
	# OFFSET computation
	#------------------------------------------------------------------
	# discard bins with fewer points than a certain threshold
	bin0.ssha_o.values[bin0.npts_o<n_thresh] = np.nan
	bin0.ssha_l.values[bin0.npts_l<n_thresh] = np.nan

	# subtract leads from ocean
	offset = (bin0.ssha_o - bin0.ssha_l).transpose('longitude', 'latitude', 'time')
	offset.values[bin0.land_mask==1] = np.nan
	offset = offset.to_dataset(name='ol_dif')

	# create weights based on the surface area of each bin
	ones = np.ones(offset.ol_dif.shape)
	ones[np.isnan(offset.ol_dif.values)] = 0
	arr_area = total_area[:,:, np.newaxis]*ones

	# normalize weights
	sum_area = arr_area.sum(axis=(0,1))
	norm_area = arr_area/sum_area

	offset['weights'] = (('longitude', 'latitude', 'time'), norm_area) 

	#- - - - - - - - - - - - - - 
	print(offset.keys())
	#- - - - - - - - - - - - - - 

	# > > a. area weighted mean and StDev for every month
	# results are the same as my implementation of the weighted avg
	weighted_obj = offset.ol_dif.weighted(offset.weights)
	monthly_off_weighted = weighted_obj.mean(('longitude', 'latitude'))

	monthly_res_sq = (offset.ol_dif - monthly_off_weighted)**2
	weighted_obj_res_sq = monthly_res_sq.weighted(offset.weights)
	monthly_variance_weighted = weighted_obj_res_sq.mean(('longitude', 'latitude'))
	monthly_std_weighted = np.sqrt(monthly_variance_weighted)

	# a. climatology of monthly area-weighted mean offset
	monthly_off_clim = monthly_off_weighted.groupby('time.month').mean('time')

	# TWO ways to compute StDev!!! ????
	#monthly_off_clim_std = monthly_std_weighted.groupby('time.month').mean('time')
	monthly_off_clim_std = monthly_off_weighted.groupby('time.month').std('time')

	ds = xr.Dataset({'ol_dif' : ('month', monthly_off_clim.values),
		'ol_std' : ('month', monthly_off_clim_std.values)},
		coords={'month' : np.arange(1,13)})

	newfile = 'b02_OL_offset_' + satellite + '_' + str(n_thresh) + statistic +'.nc'
	ds.to_netcdf(bindir + newfile)
	print("File %s saved in %s" % (newfile, bindir))

	return ds

#-------------------------------------------------------
mean_env = mclim('mean', 'env', 30)
median_env = mclim('median', 'env', 30)

mean_cs2 = mclim('mean', 'cs2', 30)
median_cs2 = mclim('median', 'cs2', 30)

#-------------------------------------------------------
xtim = mean_env.month.values
fig, ax = plt.subplots(figsize=(7,3))

ax.plot(xtim,
		mean_env.ol_dif.values*1e2,
		c='k', marker='o', markersize=4,
		label='bin mean')
ax.errorbar(xtim,
			mean_env.ol_dif.values*1e2,
			yerr=mean_env.ol_std.values*1e2,
            capsize=3, ecolor='k',
            color='none', lw=1.)
ax.plot(xtim,
		median_env.ol_dif.values*1e2,
		c='coral', marker='o', markersize=4,
		label='bin median')
ax.errorbar(xtim,
			median_env.ol_dif.values*1e2,
			yerr=median_env.ol_std.values*1e2,
            capsize=3, ecolor='coral',
            color='none', lw=1.)

ax.set_xticks(xtim, minor=True)
ax.grid(True, which='major', lw=1., ls='-')
ax.grid(True, which='minor', lw=1., ls=':')
ax.set_ylabel("O/L offset Envisat (cm)")
ax.axhline(0, ls=':', c='k')
ax.legend()
plt.tight_layout()

#-------------------------------------------------------
xtim = mean_env.month.values
fig, ax = plt.subplots(figsize=(7,3))

ax.plot(xtim,
		mean_cs2.ol_dif.values*1e2,
		c='k', marker='o', markersize=4,
		label='bin mean')
ax.errorbar(xtim,
			mean_cs2.ol_dif.values*1e2,
			yerr=mean_cs2.ol_std.values*1e2,
            capsize=3, ecolor='k',
            color='none', lw=1.)
ax.plot(xtim,
		median_cs2.ol_dif.values*1e2,
		c='coral', marker='o', markersize=4,
		label='bin median')
ax.errorbar(xtim,
			median_cs2.ol_dif.values*1e2,
			yerr=median_cs2.ol_std.values*1e2,
            capsize=3, ecolor='coral',
            color='none', lw=1.)

ax.set_xticks(xtim, minor=True)
ax.grid(True, which='major', lw=1., ls='-')
ax.grid(True, which='minor', lw=1., ls=':')
ax.set_ylabel("O/L offset CryoSat-2 (cm)")
ax.legend()
plt.tight_layout()

