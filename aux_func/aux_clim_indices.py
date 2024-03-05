"""
Process climate indices (asl, sam, soi, enso)*
* data are in two files, one for asl and one for the other indices 

> crop to a set period
> re-standardise to that period (mean=0, std=1)
> remove seasonal cycle
> plot the re-standardised time series
> return the anomaly (mean=0), standardised anomaly, and 
the stdandardised anomaly without the seas cycle; 
the latter is what Armitage et al (2018) use

Last modified: 7 Nov 2022
"""


import numpy as np
from numpy import ma

import xarray as xr
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

import datetime

import sys

plt.ion()
# - - - - - - - - local plotting function
def plot_index(var, time, label):
	fig, ax = plt.subplots(figsize=(10, 2))
	ax.plot(time, var, label=label, c='k', zorder=3)
	ax.scatter(time[var<-1], var[var<-1], c='b', zorder=4)
	ax.scatter(time[var>1], var[var>1], c='crimson', zorder=4)
	ax.axhline(0, c='grey', ls=':', zorder=1)
	ax.axvline(pd.to_datetime('2010-03-01'), c='violet')
	ax.axvline(pd.to_datetime('2015-12-31'), c='violet')

	# spines
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)

	# date axis
	years = mdates.YearLocator(1, month=1, day=1)   # every year
	months = mdates.MonthLocator()  # every month
	years_fmt = mdates.DateFormatter('%Y')

	# format the ticks
	ax.xaxis.set_major_locator(years)
	ax.xaxis.set_major_formatter(years_fmt)
	ax.xaxis.set_minor_locator(months)

	# round to nearest years
	ax.set_xlim(time[0]-np.timedelta64(60, 'D'), 
	    time[-1] + np.timedelta64(60, 'D'))

	ax.format_xdata = mdates.DateFormatter('%Y')
	fig.autofmt_xdate()

	ax.grid(True, which='major', lw=1., ls='-')
	ax.grid(True, which='minor', lw=1., ls=':')
	
	plt.tight_layout()
	ax.legend()

#----------------------------------------------
# user input variables
#----------------------------------------------
# Directories
voldir = '/Users/ocd1n16/PhD_local/data/'
idxdir = voldir + 'climate_indices/'

# set periods to crop data
date_start = '2002-11-01'
date_end = '2018-10-31'

# files
asl_csv = 'asli_era5_v3.csv'
other_idx_file = 'climate_indices_79_19.nc'
# eNSO index without the positive trend (ocean index, SOI is only atmospheric)
relative_enso34 = 'iersst_nino34a_rel_2000_2020.nc'

time = pd.date_range('2000-01-01', '2020-12-01', freq='MS')

def climate_index(index, date_start, date_end, plots):
	#----------------------------------------------
	# Amundsen Sea Low
	#----------------------------------------------
	if index == 'asl':
		asl_pd = pd.read_csv(idxdir + asl_csv)

		asl0 = xr.Dataset({'lon' : ('time', asl_pd.lon.values),
		  'lat' : ('time', asl_pd.lat.values),
		  'ActCenPres' : ('time', asl_pd.ActCenPres.values),
		  'SectorPres' : ('time', asl_pd.SectorPres.values),
		  'RelCenPres' : ('time', asl_pd.RelCenPres.values)},
		  coords={'time' : pd.to_datetime(asl_pd.time.values)})

		# crop to a set time
		asl = asl0.sel(time=slice(date_start, date_end))

		# standardise and remove seasonal cycle (Armitage et al, 2018)
		asl_anom = asl - asl.mean('time')
		asl_stand = asl_anom / asl_anom.std(ddof=1)
		asl_mclim_stand = (asl_stand.groupby('time.month')
			- asl_stand.groupby('time.month').mean('time'))

		# PLOTS
		if plots == 'y':
			x = asl_mclim_stand.time
			plot_index(asl_mclim_stand.lon, x, 'ASL lon')
			plot_index(asl_mclim_stand.lat, x, 'ASL lat')
			plot_index(asl_mclim_stand.ActCenPres, x, 'ASL ActCenPres')
			plot_index(asl_mclim_stand.SectorPres, x, 'ASL SectorPres')
			plot_index(asl_mclim_stand.RelCenPres, x, 'ASL RelCenPres')

		return asl_anom, asl_stand, asl_mclim_stand
	#----------------------------------------------
	# ENSO 3.4 without linear trend
	#----------------------------------------------
	elif index == 'enso34':
		with xr.open_dataset(idxdir + relative_enso34, decode_times=False) as a:
		  print(a)

		a['time'] = time

		a_crop = a.sel(time=slice(date_start, date_end))

		# standardise data (remove mean, divide by std; mean=0, std=1)
		a_anom = a_crop - a_crop.mean()
		a_stand = a_anom / a_anom.std(ddof=1)
		a_mclim_stand = (a_stand.groupby('time.month') 
							- a_stand.groupby('time.month').mean('time'))

		# PLOTS
		if plots == 'y':
			xtim = a_crop.time.values
			plot_index(a_mclim_stand["Nino3.4r"], xtim, 'Nino 3.4')

		return a_anom, a_stand, a_mclim_stand
	#----------------------------------------------
	# SAM, SOI, Nino (3, 3.4, 4)
	#----------------------------------------------
	elif index == 'other':
		with xr.open_dataset(idxdir + other_idx_file) as clim_idx:
		  print(clim_idx)

		indx = clim_idx.sel(time=slice(date_start, date_end))

		# standardise data (remove mean, divide by std; mean=0, std=1)
		indx_anom = indx - indx.mean()
		indx_stand = indx_anom / indx.std(ddof=1)
		indx_mclim_stand = (indx_stand.groupby('time.month') 
											- indx_stand.groupby('time.month').mean('time'))

		# PLOTS
		if plots == 'y':
			xtim = indx.time.values
			plot_index(indx_mclim_stand.sam, xtim, 'SAM')
			plot_index(indx_mclim_stand.soi_ncep*(-1), xtim, 'SOI (NCEP) x (-1)')
			plot_index(indx_mclim_stand.soi_ncar*(-1), xtim, 'SOI (NCAR) x (-1)')
			plot_index(indx_mclim_stand.nino3, xtim, 'Nino 3')
			plot_index(indx_mclim_stand.nino34, xtim, 'Nino 3.4')

		return indx_anom, indx_stand, indx_mclim_stand

	else: 
		print("Typo! index must be 'asl' or 'other' (sam, soi, nino). ")
		sys.exit()
