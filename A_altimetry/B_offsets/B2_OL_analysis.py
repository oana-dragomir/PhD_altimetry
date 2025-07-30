"""
Compute O/L offset for envisat from binned data
- area-weighted average of overlapping cells
- monthly climatology of all months

- uses the one file with all gridded data instead of individual files

Last modified: 30 Mar 2021
"""    
#------------------------------------------------------------------
"""
# TEMPORAL VARIABILITY
- compute monthly spatial mean/median/stdev of the the bias
- make time series of the monthly spatial means + stdev
- is there a seasonal signal?

# SEASONAL VARIABILITY/CLIMATOLOGY
- average spatial mean by month
- look at StDev by month
- similar to Tiago's method

# SAPTIAL VARIABILITY
- divide into 30 -60 deg long regions
- look at time series of the monthly spatial averages for that region
- check how stdev compares with the climatology
"""
#------------------------------------------------------------------

## Import modules
import numpy as np
from numpy import ma

import matplotlib.pyplot as plt 
from mpl_toolkits.basemap import Basemap
from matplotlib import cm

import xarray as xr
import pandas as pd

from scipy.stats import kurtosis, skew

import sys

#-------------------------------------------------------------------------------

def skew_kurt_hist(var, month):
	skew0 = kurtosis(var.flatten(), nan_policy='omit')
	kurt0 = skew(var.flatten(), nan_policy='omit')
	print('Kurtosis: ', skew0)
	print('Skewnesss: ', kurt0)

	fig, ax = plt.subplots()
	ax.hist(var.flatten()*1e2, bins=17, 
	    label='n_thresh=%s'% str(n_thresh))
	ax.annotate('skew = %.4f' % skew0,
	    xy=(.15, .8),
	    xycoords='figure fraction')
	ax.annotate('kurt = %.4f' % kurt0,
	    xy=(.15, .75),
	    xycoords='figure fraction')

	ax.legend(loc=2)
	ax.set_title("%s O-L offset (cm) %s" % (satellite, month), loc='left')
	plt.tight_layout()

#-------------------------------------------------------------------
# Define directories
#-------------------------------------------------------------------
voldir = '/Volumes/SamT5/PhD/PhD_data/'
ncdir = voldir + 'altimetry_cpom/1_raw_nc/'
bindir = voldir + 'altimetry_cpom/2_grid_offset/'

scriptdir = '/Volumes/SamT5/PhD/PhD_scripts/'
auxscriptdir = scriptdir + 'aux_func/'

sys.path.append(auxscriptdir)
import aux_func as ft
import aux_stereoplot as st
#- - - - - - - - - - - - - - 

#-------------------------------------------------------
# bin edges
edges_lon = np.linspace(-180, 180, num=361, endpoint=True)
edges_lat = np.linspace(-82, -50, num=65, endpoint=True)
eglat, eglon = np.meshgrid(edges_lat, edges_lon)

total_area = ft.grid_area(eglon, eglat)
#-------------------------------------------------------

# # # # # # # # # # # # 
statistic = 'mean'
satellite = 'cs2'
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

#------------------------------------------------------------------
#- - - - - - - - - - - - - -
#- - - - - - - - - - - - - -
# > > a. area weighted mean and StDev for every month
# results are the same as my implementation of the weighted avg
#- - - - - - - - - - - - - -
#- - - - - - - - - - - - - -
weighted_obj = offset.ol_dif.weighted(offset.weights)
monthly_off_weighted = weighted_obj.mean(('longitude', 'latitude'))

# a. climatology of monthly area-weighted mean offset
monthly_off_clim = monthly_off_weighted.groupby('time.month').mean('time')

#- - - - - - - - - - - - - - 

# standard deviation - weighted
monthly_res_sq = (offset.ol_dif - monthly_off_weighted)**2
weighted_obj_res_sq = monthly_res_sq.weighted(offset.weights)
monthly_variance_weighted = weighted_obj_res_sq.mean(('longitude', 'latitude'))
monthly_std_weighted = np.sqrt(monthly_variance_weighted)

# standard deviation - not weighted - ALMOST SIMILAR
monthly_variance = monthly_res_sq.mean(('longitude', 'latitude'))
monthly_std = np.sqrt(monthly_variance)

# spread of monthly area-weighted mean for every month
monthly_off_clim_std = monthly_off_weighted.groupby('time.month').std('time')

# error addition (sqrt of sum of square std)
comb_std_sum = (monthly_std**2).groupby('time.month').sum()
comb_std_clim = np.sqrt(comb_std_sum)

#- - - - - - - - - - - - - -
#- - - - - - - - - - - - - -
# > > b. for every month, average offset in time in every bin 
#- - - - - - - - - - - - - -
#- - - - - - - - - - - - - -
offset_month = offset.ol_dif.groupby('time.month').mean('time')
offset_month_std = offset.ol_dif.groupby('time.month').std('time', ddof=1)

# keep only bins where std <= 0.7 m
offset_month.values[offset_month_std>0.7] = np.nan
offset_month.values[np.isnan(offset_month_std)] = np.nan

offset_month = offset_month.to_dataset(name='ol_m')

# > > b. area weighted avg for every time-averaged month
# create weights based on the surface area of each bin
ones = np.ones(offset_month.ol_m.shape)
ones[np.isnan(offset_month.ol_m.values)] = 0
arr_area = total_area[:,:, np.newaxis]*ones

offset_month['weights'] = (('longitude', 'latitude', 'month'), arr_area) 

# climatology - weights for every calendar month avg (12)
off_obj = offset_month.ol_m.weighted(offset_month.weights)
off_mclim = off_obj.mean(('longitude', 'latitude')) 

res = (offset_month.ol_m - off_mclim)**2
res_obj = res.weighted(offset_month.weights)
res_var = res_obj.mean(('longitude', 'latitude'))
off_mclim_std = np.sqrt(res_var)


#------------------------------------------------------------------
# stereographical PLOT for individual months
#------------------------------------------------------------------
cbar_range = [-30, 30]
cbar_units = 'cm'
cmap = cm.seismic

k = 1
fig, ax, m = st.spstere_plot(eglon, eglat, 
				offset.ol_dif.isel(time=k)*100,
				cbar_range, cmap, cbar_units, None)

#- - - - - - - - - - - - - -
var = offset.ol_dif.values.flatten()
skew_kurt_hist(var, '')

#- - - - - - - - - - - - - -
xtim = monthly_off_weighted.time.values
fig, ax = plt.subplots(figsize=(12,3))

ax.plot(xtim,
		monthly_off_weighted.values*1e2,
		c='k', marker='o', markersize=4)
ax.errorbar(xtim,
			monthly_off_weighted.values*1e2,
			yerr=monthly_std.values*1e2,
            capsize=3, ecolor='k',
            color='none', lw=1.)
ax.set_xticks(xtim, minor=True)
ax.grid(True, which='major', lw=1., ls='-')
ax.grid(True, which='minor', lw=1., ls=':')
ax.set_ylabel("O/L offset %s (cm)" % satellite)
ax.axhline(0, ls=':', c='k')
plt.tight_layout()

#- - - - - - - - - - - - - -
xtim = monthly_off_clim.month.values
fig, ax = plt.subplots(figsize=(7,3))

ax.plot(xtim,
		monthly_off_clim.values*1e2,
		c='k', marker='o', markersize=4)
ax.errorbar(xtim,
			monthly_off_clim.values*1e2,
			yerr=monthly_off_clim_std.values*1e2,
            capsize=3, ecolor='k',
            color='none', lw=1.)
ax.set_xticks(xtim, minor=True)
ax.grid(True, which='major', lw=1., ls='-')
ax.grid(True, which='minor', lw=1., ls=':')
ax.set_ylabel("O/L offset %s (cm)" % satellite)
ax.axhline(0, ls=':', c='k')
plt.tight_layout()

#------------------------------------------------------------------
# stereographical PLOT - maps of monthly avg offset & StDev
#------------------------------------------------------------------
cbar_range = [-40, 40]
cbar_units = 'Dec (cm)'
cmap = cm.seismic

k=1 	# 0-11 Jan-Dec
fig, ax, m = st.spstere_plot(eglon, eglat, 
                offset_month.ol_m.isel(month=k)*100,
                cbar_range, cmap, cbar_units, None)
fig, ax, m = st.spstere_plot(eglon, eglat, 
                offset_month_std.isel(month=k)*100,
                cbar_range, cm.cool, cbar_units + ' StDev', None)

#- - - - - - - - - - - - - -
var = offset_month.ol_m.isel(month=k).values
skew_kurt_hist(var, 'Jan')

var = offset_month_std.isel(month=k).values
skew_kurt_hist(var, 'Jan')

# monthly climatology
#- - - - - - - - - - - - - -
xtim = off_mclim.month.values
fig, ax = plt.subplots(figsize=(7,3))

ax.plot(xtim,
		off_mclim.values*1e2,
		c='k', marker='o', markersize=4)
ax.errorbar(xtim,
			off_mclim.values*1e2,
			yerr=off_mclim_std.values*1e2,
            capsize=3, ecolor='k',
            color='none', lw=1.)
ax.set_xticks(xtim, minor=True)
ax.grid(True, which='major', lw=1., ls='-')
ax.grid(True, which='minor', lw=1., ls=':')
ax.set_ylabel("O/L offset %s (cm)" % satellite)
ax.axhline(0, ls=':', c='k')
plt.tight_layout()
