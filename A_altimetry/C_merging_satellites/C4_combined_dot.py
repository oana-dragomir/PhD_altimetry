"""
Combine env and cs2 binned files to cover the full period;
- intersat offset is computed from binned data from the overlap of CS2 and ENV
- intersatellite offset is added to binned data to reference to Envisat.
- compute geostrophic velocity
- save all in a file using xarray

# Intersatellite offset (median, metres, without negative values)
# v6
#intersat_off = 0.03873
# v7, sigma=2
#intersat_off = 0.037405

# v7, sigma=3
#intersat_off = 0.038668 # bmedian, mean
#intersat_off = 0.03372 # bmean, mean

Last modified: 8 Apr 2021

~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
[mean_30bin]   egm08   / goco05c
    > mean   = 3.37 cm / 3.40 cm
    > median = 3.87 cm / 3.89 cm

[median_30bin]
    > mean   = 3.41 cm / 3.44 cm
    > median = 3.86 cm / 3.88 cm

> apply the median-based offset because it reduces the rms of SLA residuals 
after correcting CS2 with the intersat offset

Last edited to merge two-weekly averages of DOT maps: 17 Oct 2022
"""

import numpy as np
from numpy import ma

from datetime import datetime
today = datetime.today()

import xarray as xr
import pandas as pd
import gsw

import sys

# Define directories
voldir = '/Volumes/SamT5/PhD_data/'
griddir = voldir + 'altimetry_cpom/3_grid_dot/'

#----------------------------------------------------------
# # # # # # # # # # # # 
statistics = 'median'
geoidtype = '_goco05c'#'_eigen6s4v2' #'_egm08'  # '_goco05c'
# Gaussian filter - sigma and corresponding radius
gauss_sig = 3
# # # # # # # # # # # # 

if statistics == 'median':
    if gauss_sig == 2:
        gauss_radius = '100km'
        if geoidtype == '_goco05c':
            intersat_off = 0.0386
        elif geoidtype == '_eigen6s4v2':
            intersat_off = 0.0393
    elif gauss_sig == 3:
        gauss_radius = '150km'
        if geoidtype == '_egm08':
            intersat_off = 0.0387
        elif geoidtype == '_goco05c':
            intersat_off = 0.0342 #0.0389 (3.89cm is for monthly averages; 3.42 cm is for 2-weekly avg)
        elif geoidtype == '_eigen6s4v2':
            intersat_off = 0.0384 

print(gauss_radius, intersat_off)
# # # # # # # # # # # # 

cs2_file = griddir + '2week_dot_cs2_30b' + statistics + geoidtype + '_sig' + str(gauss_sig) + '.nc'
env_file = griddir + '2week_dot_env_30b' + statistics + geoidtype + '_sig' + str(gauss_sig) + '.nc'

with xr.open_dataset(cs2_file) as cs2_dict:
    print(cs2_dict.keys())
with xr.open_dataset(env_file) as env_dict:
    print(env_dict.keys())

# LAT/LON GRID
lat = env_dict.latitude.values
lon = env_dict.longitude.values
elat = env_dict.edge_lat.values
elon = env_dict.edge_lon.values

# GRID
eglat, eglon = np.meshgrid(elat, elon)
glat, glon = np.meshgrid(lat, lon)
gmlat, gmlon = np.meshgrid(elat[1:-1], elon[1:-1])

lmask = env_dict.land_mask.values


# CS2 correction
cs2_dot = cs2_dict.dot + intersat_off

#----------------------------------------------------------
#time = pd.date_range('2002-07-01', '2018-10-01', freq='1MS')
#time_env = pd.date_range('2002-07-01', '2010-10-01', freq='1MS')
#time_overlap = pd.date_range('2010-11-01', '2012-03-01', freq='1MS')
#time_cs2 = pd.date_range('2012-04-01', '2018-10-01', freq='1MS')

time_env = env_dict.time.values
time_cs2 = cs2_dict.time.values

# time coord for the combined satellites
time = np.unique(np.concatenate((time_env, time_cs2)))
# find the intersection of the two time arrays; returns the sorted unique values
time_overlap = np.intersect1d(time_env, time_cs2)

# crop
dot_env = env_dict.dot.sel(time=slice(time_env[0], time_overlap[0])).values
dot_env_overlap = env_dict.dot.sel(time=slice(time_overlap[0], time_overlap[-1])).values
dot_cs2_overlap = cs2_dot.sel(time=slice(time_overlap[0], time_overlap[-1])).values
dot_cs2 = cs2_dot.sel(time=slice(time_overlap[-1], time_cs2[-1])).values
#----------------------------------------------------------
# combine satellites
dot_overlap = 0.5*(dot_env_overlap + dot_cs2_overlap)
# drop the last and first time slices because they are part of the overlap
dot = np.dstack((dot_env[:,:,:-1], dot_overlap, dot_cs2[:,:,1:]))

# apply land mask
dot[lmask==1] = np.nan

#----------------------------------------------------------
# dimensions
#londim, latdim, timdim = dot.shape

# forward and backward vel; use forward and backward vel for endpoints 
# and avg mean for the in-between

# given the circumpolar/circular nature of the data, 
# extend longitude by one grid step and append the first col to the end
lon_ext = np.hstack((lon, lon[-1] + 1))
dot_ext = np.concatenate((dot, dot[0:1,:,:]), axis=0)

#since map is circumpolar append first col to the end
glat_ext, glon_ext = np.meshgrid(lat, lon_ext)

# scale by f and g
f = gsw.f(glat)
g =  gsw.grav(glat, 0)

print("Computing gvel .. ")
# distance between grid points
# given the grid is uniform, no need to flip the arrays
dx = gsw.distance(glon_ext, glat_ext, axis=0)  # lon
dy = gsw.distance(glon, glat, axis=1)  # lat

# ---------- u vel ------------------
# a. forward diff
dnx_forw = dot_ext[1:] - dot_ext[:-1]
grad_x_forw = dnx_forw / dx[:, :, np.newaxis]

grad_x_avg = 0.5 * (grad_x_forw[1:] + grad_x_forw[:-1])
grad_x = np.concatenate((grad_x_forw[0:1, : , :], grad_x_avg), axis=0)

vg = grad_x * (g/f)[:, :, np.newaxis]

# ---------- v vel ------------------
# a. forward diff
dny_forw = dot[:, 1:, :] - dot[:, :-1, :]
grad_y_forw = dny_forw / dy[:, :, np.newaxis]

grad_y_avg = 0.5 * (grad_y_forw[:,1:,:] + grad_y_forw[:,:-1,:])
grad_y = np.concatenate((grad_y_forw[:, 0:1, :], 
                        grad_y_avg, 
                        grad_y_forw[:, -1, :][:,np.newaxis,:]), axis=1)

ug = (-1) * grad_y * (g/f)[:, :, np.newaxis]


#speed = np.sqrt(ug**2 + vg**2)
#mean_speed = speed.mean(axis=2)

#mean_u = np.mean(ug, axis=2)


# dsla_x = dot[1:] - dot[:-1]
# dsla_y = dot[:, 1:, :] - dot[:, :-1, :]

# # take differences along one axis ..
# dsla_dx = dsla_x/dx[:, :, np.newaxis]
# dsla_dy = dsla_y/dy[:, :, np.newaxis]

# # average along the other direction
# dsla_dx_mid = 0.5*(dsla_dx[:, 1:, :]+dsla_dx[:, :-1, :])
# dsla_dy_mid = 0.5*(dsla_dy[1:, :, :]+dsla_dy[:-1, :, :])

# # scale by f and g
# f = gsw.f(glat)
# g =  gsw.grav(glat, 0)

# vg = dsla_dx_mid * (g/f)[:, :, np.newaxis]
# ug = (-1) * dsla_dy_mid * (g/f)[:, :, np.newaxis]

# vellon = elon[1:-1]
# vellat = elat[1:-1]
#----------------------------------------------------------
# .nc file
newfile = '2week_dot_all_30b' + statistics + geoidtype +'_sig' + str(gauss_sig) + '.nc'

ds = xr.Dataset({'dot' : (('longitude', 'latitude', 'time'), dot), 
                 'ug' : (('longitude', 'latitude', 'time'), ug),
                 'vg' : (('longitude', 'latitude', 'time'), vg),
                 'land_mask' : (('longitude', 'latitude'), lmask),
                 'intersat_offset' : intersat_off},
                coords={'longitude' : lon,
                        'latitude' : lat,
                        'time' : time,
                        'edge_lat' : elat,
                        'edge_lon' : elon})

ds.attrs['history'] = "Created " + today.strftime("%d/%m/%Y, %H:%M%S" )
ds.dot.attrs['units']='meters'
ds.ug.attrs['units']='metres/second'
ds.vg.attrs['units']='metres/second'
ds.ug.attrs['long_name']='zonal_surface_geostrophic_velocity'
ds.vg.attrs['long_name']='meridional_surface_geostrophic_velocity'
ds.dot.attrs['long_name']='dynamic_ocean_topography'
ds.intersat_offset.attrs['units']='metres'

description_text = (("ENVISAT + CryoSat2 altimetry (geoid:%s) \n"
"env: 07.2002-03.2012 || cs2:11.2010-10.2018 \n"
"> Lat, Lon at bin centre and edges(for pcolormesh) \n"
"> DOT (bin_statistic: %s, |DOT| < 3m) \n"
"> time (bi-weekly from the start of each month; the mean is over 1-15/15-end days of every month) \n"
"> compute value only where bins have more than 30 points \n"
"> land mask (land=1, ocean=0) \n"
"> *intersat offset = area-weighted average of "
"median of binned SLA_env-SLA_cs2 in the overlap period referenced to MDT_env \n"
" Intersatellite offset of %s m has been applied to binned CS2."
" Gaussian filter (sig=%s, %s) applied to binned data ") %
(geoidtype, statistics, intersat_off, gauss_sig, gauss_radius))

ds.attrs['description'] = description_text
ds.to_netcdf(griddir+newfile)

sys.exit()

# b. backwards diff
#xflip_dot = np.flip(dot_ext, axis=0)
#dnx_back = np.flip(xflip_dot[1:] - xflip_dot[:-1], axis=0) 
#grad_x_back = dnx_back / dx[:, :, np.newaxis]

# c. average of forward + backwards
#grad_x_avg = 0.5 * (grad_x_forw[:-1] + grad_x_back[1:])

# use first forward vel as the first entry
#grad_x = np.concatenate((grad_x_forw[0:1, : , :], grad_x_avg), axis=0)


# # b. backwards diff
# yflip_dot = np.flip(dot, axis=1)
# dny_back = np.flip(yflip_dot[:, 1:, :] - yflip_dot[:, :-1, :], axis=1) 
# grad_y_back = dny_back / dy[:, :, np.newaxis]

# # c. average of forward + backwards
# grad_y_avg = 0.5 * (grad_y_forw[:, :-1, :] + grad_y_back[:, 1:, :])

# # use first forward vel as the first entry
# grad_y = np.concatenate((grad_y_forw[:, 0:1 , :], grad_y_avg, grad_y_back[:,-1:-2, :]), axis=1)

"""

ds = Dataset(datadir + newfile, 'w')

# dimensions
lo, la, itt = dot.shape

ds.createDimension('lon', lo)
ds.createDimension('elon', lo+1)
ds.createDimension('lat', la)
ds.createDimension('elat', la+1)
ds.createDimension('idx', itt)
ds.createDimension('n1', 1)

# variables
lati = ds.createVariable('latitude', np.float64, ('lat'))
longi = ds.createVariable('longitude', np.float64, ('lon'))
elati = ds.createVariable('edges_lat', np.float64, ('elat',))
elongi = ds.createVariable('edges_lon', np.float64, ('elon',))
SSH = ds.createVariable('dot', np.float64, ('lon', 'lat', 'idx'))
SLA = ds.createVariable('sla', np.float64, ('lon', 'lat', 'idx'))
MDT = ds.createVariable('mdt', np.float64, ('lon', 'lat'))
numpts = ds.createVariable('num_pts', np.float64, ('lon', 'lat', 'idx'))
tim = ds.createVariable('time', np.float64, ('idx'))
land_mask = ds.createVariable('landmask', np.int32, ('lon', 'lat'))
intersat_mask = ds.createVariable('intersat_mask', np.int32, ('lon', 'lat'))
intersat = ds.createVariable('intersatellite_offset', np.float32, ('n1',))

ds.description = ("ENV + CryoSat2: \
*Lat, Lon at bin centre and edges \
*DOT (corrected, median of bin)\
*SLA (ref to mean over all period)\
*MDT\
*time (beginning of month)\
env:07.2002-03.2012; cs2:11.2010-10.2018\
*number of points in bins\
*land mask (land=1, ocean=0).\
*intersatellite mask (0 where offset < 0, 1 otherwise)\
Data have been filtered by sea-ice type\
and have the land mask applied.\
Intersatellite offset has been applied\
to along-track CS2 (3.7405 cm).")

ds.history = "Created " + today.strftime("%d/%m/%Y, %H:%M%S" )

tim.units = time_units 
SSH.units='meters'
lati.units='degrees_north'
longi.units='degrees_east'
elati.units='degrees_north'
elongi.units='degrees_east'
SSH.long_name='dynamic_ocean_topography'
SLA.long_name='sea_level_anomaly'
MDT.long_name='mean_dynamic_ocean_topography'

lati[:] = lat
longi[:] = lon
elongi[:] = elon
elati[:] = elat
SSH[:] = dot.filled(np.nan)
SLA[:] = sla.filled(np.nan)
MDT[:] = mdt.filled(np.nan)
numpts[:] = num_pts
tim[:] = time
land_mask[:] = lmask.filled(np.nan)
intersat_mask[:] = offsetmask.filled(np.nan)
intersat = intersat_off

ds.close()


u = ma.zeros((londim, latdim-1, timdim))
v = ma.zeros((londim-1, latdim, timdim))
for t in range(timdim):
    for x in range(londim):
        u[x, :, t] = gsw.geostrophic_velocity(dot[x, :, t], glon[x, :], glat[x, :],
                                           p=0)[0]
    for y in range(latdim):
        v[:, y, t] = gsw.geostrophic_velocity(dot[:, y, t], glon[:, y], glat[:, y],
                                           p=0)[0]

# zonal vel is +ve eastward and meridional vel is +ve northward
u = (-1)*u
v = v
# reduce the other dimension by 1
ug_vel = 0.5*(u[1:, :, :] + u[:-1, :, :])
vg_vel = 0.5*(v[:, 1:, :] + v[:, :-1, :])

# scale by g
ug_vel *= g[:, :, np.newaxis]
vg_vel *= g[:, :, np.newaxis]

t_stop = runtime.process_time()
print("execution time: %.3f s " %(t_stop-t_start))
"""
