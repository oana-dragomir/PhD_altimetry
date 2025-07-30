"""
Script 1/2 Sea Ice Motion (u, v, error) 

> Compute monthly averages from daily data
> Interpolate data from 25x25km to wind grid and then SSH grid (0.5 lat x 1 lon)
	>> the interpolation to wind grid first makes it less coarse
> save a file for every year with the monthly data

Last modified: 10 Mar 2025
"""

import numpy as np
import numpy.ma as ma

import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.basemap import Basemap

# prevent figure from popping up 
# non-interactive backend
import matplotlib
matplotlib.use('Agg')

from scipy.interpolate import griddata

import sys
import os

# Directories
maindir = '/Volumes/SamT5/PhD_data/'
workdir = maindir + 'data/'
icedir = workdir + 'NSIDC/ice_drift/'
topodir = workdir + 'topog/'
altdir = workdir + 'altimetry_cpom/3_grid_dot/'
winddir = workdir + 'reanalyses/'
sicfigdir = workdir + '../PhD_figures/Figures_SIC/'

# import my functions
scriptdir = '/Volumes/SamT5/PhD_scripts/'
auxscriptdir = scriptdir + 'aux_func/'
sys.path.append(auxscriptdir)
import aux_stereoplot as st


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# 				ERA Interim data
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
with xr.open_dataset(winddir + 'eraint_1979_2018.nc') as wind:
    print(wind.keys())

# grid
wind_glat, wind_glon = np.meshgrid(wind.latitude, 
                                   wind.longitude)

dtim = ice.time.size
dwlon, dwlat = wind_glat.shape

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
#				Altimetry data
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
geoidtype = 'goco05c'#'eigen6s4v2' # 'goco05c', 'egm08'
satellite = 'all'
sigma = 3

altfile = 'dot_' + satellite + '_30bmedian_' + geoidtype + '_sig' + str(sigma) + '.nc'

saving_figures = False
#-------------------------------------------------------------------
# load altimetry file
with xr.open_dataset(altdir+altfile) as alt:
    print(alt.keys())

# GRID coordinates
# at bin edges
alt_eglat, alt_eglon = np.meshgrid(alt.edge_lat.values,
								 alt.edge_lon.values)
# at bin centres
alt_glat, alt_glon = np.meshgrid(alt.latitude.values,
								 alt.longitude.values)
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
#				SIC data
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# number of years
yr = np.arange(2002, 2019)
fn0 = 'icemotion_daily_sh_25km_'

cbar_range = [-3, 3]
cmap = cm.get_cmap('bwr', 31)
cbar_units = 'cm/s'
cbar_extend = 'both'

for k in range(len(yr)):
	filename = fn0 + str(yr[k]) + '0101_' + str(yr[k]) + '1231_v4.1.nc'
	print("Processing %s \n" % filename)

	with xr.open_dataset(icedir + filename) as ds:
	    print(ds.keys)

	lon = ds.longitude.values
	lat = ds.latitude.values

	# rotate the along-x/-y components to east/north
	vel_zonal = ds.u * np.cos(ds.longitude * np.pi / 180.) + ds.v * np.sin(ds.longitude * np.pi / 180.)
	vel_merid = (-1) * ds.u * np.sin(ds.longitude * np.pi / 180.) + ds.v * np.cos(ds.longitude * np.pi / 180.)
	
	r, c = vel_zonal.shape[1:]
	months = vel_zonal.time.dt.month.values

	# compute monthly means - only to use the time coord
	mv_zonal = vel_zonal.resample(time='1MS').mean()
	#mv_merid = vel_merid.resample(time='1MS').mean()

	timdim = len(mv_zonal.time)
	figdate = mv_zonal.time.dt.strftime("%Y%m").values

	# compute monthly averages
	# mask cells that have data less than 20/30 days
	um, vm = [np.zeros((12, r, c)) for _ in range(2)]
	for i in range(1, 13):
		u = ma.masked_invalid(vel_zonal.values[months==i])
		u_mean = u.mean(axis=0)
		u_num = u.count(axis=0)
		u_mean[u_num<20] = ma.masked
		um[i-1] = u_mean.filled(np.nan)

		v = ma.masked_invalid(vel_merid.values[months==i])
		v_mean = v.mean(axis=0)
		v_num = v.count(axis=0)
		v_mean[v_num<20] = ma.masked
		vm[i-1] = v_mean.filled(np.nan)


	print("Interpolate every month on wind and then altimetry grid .. \n")
	# interpolate monthly data onto wind grid then altimetry
	for i in range(timdim):
		wgrid_u = griddata((lon.flatten(), lat.flatten()), 
		                um[i,:,:].flatten(),
		                (wind_glon, wind_glat), method='nearest')
		wgrid_v = griddata((lon.flatten(), lat.flatten()), 
		                vm[i,:,:].flatten(),
		                (wind_glon, wind_glat), method='nearest')

		agrid_u = griddata((wind_glon.flatten(), wind_glat.flatten()), 
		                wgrid_u.flatten(),
		                (alt_glon, alt_glat), method='linear')
		agrid_v = griddata((wind_glon.flatten(), wind_glat.flatten()), 
		                wgrid_v.flatten(),
		                (alt_glon, alt_glat), method='linear')
		
		if saving_figures:
			print("Saving figures ..")

			fig, ax, m = st.spstere_plot(alt_glon, alt_glat, itp_u,
										 cbar_range, cmap, 
	                       				 cbar_units, cbar_extend)
			ax.annotate("U %s/%s" % (yr[k], mv_zonal.time.dt.month.values[i]), 
		      xycoords='axes fraction', xy=(.43, .5), 
		      bbox=dict(facecolor='lavender',
		                edgecolor='lavender',
		                boxstyle='round'))
			fig.savefig(sicfigdir + 'seaice_u_'+figdate[i]+'.png')
			plt.close()

			fig, ax, m = st.spstere_plot(alt_glon, alt_glat, itp_v,
										 cbar_range, cmap, 
	                       				 cbar_units, cbar_extend)
			ax.annotate("V %s/%s" % (yr[k], mv_zonal.time.dt.month.values[i]), 
		      xycoords='axes fraction', xy=(.43, .5), 
		      bbox=dict(facecolor='lavender',
		                edgecolor='lavender',
		                boxstyle='round'))
			fig.savefig(sicfigdir + 'seaice_v_'+figdate[i]+'.png')
			plt.close()
		
		if i==0:
			u_grid, v_grid = agrid_u, agrid_v
		else:
			u_grid = np.dstack((u_grid, agrid_u))
			v_grid = np.dstack((v_grid, agrid_v))

	gds = xr.Dataset({'u' : (('lon', 'lat', 'time'), u_grid),
					'v' : (('lon', 'lat', 'time'), v_grid)},
					coords={'lon' : alt.longitude.values,
							'lat' : alt.latitude.values,
							'time' : mv_zonal.time.values})
	gds.u.attrs['long_name'] = "sea_ice_zonal_velocity"
	gds.u.attrs['units'] = "cm/s"
	gds.u.attrs['comments'] = "positive eastward"
	gds.v.attrs['long_name'] = "sea_ice_meridional_velocity"
	gds.v.attrs['units'] = "cm/s"
	gds.v.attrs['comments'] = "positive northward"

	newfilename = 'SIvel_monthly_wgrid_p5x1_' + str(yr[k]) + '.nc'
	gds.to_netcdf(icedir + newfilename)



	

