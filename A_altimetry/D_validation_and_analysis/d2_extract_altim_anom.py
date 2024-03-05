"""
Function that crops and detrends DOT
- also compute geostrophic velocity after detrending

Last modified: 13 Apr 2021
"""

import numpy as np
from numpy import ma

import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.dates as mdates

import sys
import gsw

# Directories
voldir = '/Users/ocd1n16/PhD_local/data/'
griddir = voldir + 'altimetry_cpom/3_grid_dot/'

localdir = '/Users/ocd1n16/PhD_local/'

auxscriptdir = localdir + 'scripts/aux_func/'
sys.path.append(auxscriptdir)
import aux_func_trend as fc
import aux_stereoplot as st

plt.ion()

from palettable.scientific.diverging import Vik_20
# - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - 
def detrend_fc(var, var_eglon, var_eglat):
  """
  - remove linear trend at every grid point
  - var must have shape (lon, lat, time)
  - (var_eglon, var_eglat) = meshgrid with lat/lon values at bin edges for pcolormesh
  
  Returns detrended var.
  """
  print("\n>>> Computing linear trend at every grid point ..")

  date = var.time.dt.strftime("%m.%Y").values
  ndays = mdates.date2num(list(var.time.values))
  dt = ndays - ndays[0]

  # DIMENSIONS
  londim, latdim, timdim = var.shape

  # compute linear trend at every grid point
  interc, slope, ci, pval = [ma.zeros((londim, latdim)) for _ in range(4)]
  for r in range(londim):
      for c in range(latdim):
          arr_trend, _ = fc.trend_ci(var[r,c,:], 0.95)
          interc[r, c] = arr_trend.intercept.values
          slope[r, c] = arr_trend.slope.values
          pval[r, c] = arr_trend.p_val.values

  # trend in mm/yr with the GMSLR 
  slope_mm_yr = slope * 1e3 * 365.25

  # apply a mask to plot only trend values where p_val < 0.1
  slope_mm_yr[pval>0.1] = np.nan

  cbar_range = [-10, 10]
  #cmap = cm.get_cmap('coolwarm', 31)
  cmap = cm.get_cmap(Vik_20.mpl_colormap, 20)
  cbar_units = ('Linear trend (mm/yr) where p-val<0.1 (%s-%s)' % (date[0], date[-1]))
  
  fig, ax, m = st.spstere_plot(var_eglon, var_eglat, slope_mm_yr,
    cbar_range, cmap, cbar_units, "m")
  # p-value contours [<0.1]
  # cs1 = m.contourf(alt_glon, alt_glat, dot_pval,
  #                  levels=[0., 0.1], colors='none', 
  #                  hatches=['////', None], zorder=3, latlon=True)
  # cs2 = m.contour(alt_glon, alt_glat, dot_pval,
  #                 levels=[0., 0.1], colors='k',
  #                 zorder=3, linewidths=.8, latlon=True)

  # ** subtract the global mean sea level   
  #gmslr = alt_dt * 0.0032/365.25

  # detrend the time series at every point
  # gmslr/linear trend would be affected by the number of EOFs used in the recosntruction
  # thus subtract the linear trend before decomposition
  trend = slope[:,:,np.newaxis] * ndays[np.newaxis, np.newaxis, :] + interc[:,:,np.newaxis]
  var_det = var - trend #- gmslr
  return var_det

def geos_vel(dot, lat, lon):
  """
  dot: shape (lon, lat, time)
  """
  lon_ext = np.hstack((lon, lon[-1] + 1))
  dot_ext = np.concatenate((dot, dot[0:1,:,:]), axis=0)

  #since map is circumpolar append first col to the end
  glat, glon = np.meshgrid(lat, lon)
  glat_ext, glon_ext = np.meshgrid(lat, lon_ext)

  # scale by f and g
  f = gsw.f(glat)
  g =  gsw.grav(glat, 0)

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
  return ug, vg
# - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - 
def extract_dot_maps(geoidtype, satellite, sig, date_start, date_end):

  altfile = 'dot_' + satellite + '_30bmedian_' + geoidtype + '_sig' + str(sig) + '.nc'
  with xr.open_dataset(griddir+altfile) as alt:
      print(alt.keys())

  print("Crop altimetry to \n\n > > %s - %s\n\n" % (date_start, date_end))

  alt_crop = alt.sel(time=slice(date_start, date_end))
  dot = alt_crop.dot
  lat = alt_crop.latitude.values
  lon = alt_crop.longitude.values

  # GRID coordinates
  eglat, eglon = np.meshgrid(alt_crop.edge_lat, alt_crop.edge_lon)
  #-----------------------------------
  # DE-TREND function
  #-----------------------------------
  dot_det = detrend_fc(dot, eglon, eglat)


  print("Computing gvel .. ")
  ug_det, vg_det = geos_vel(dot_det.values, lat, lon)
  #-----------------------------------

  if satellite == 'all': 
    altim = xr.Dataset({'dot' : (('longitude', 'latitude', 'time'), dot.values),
      'dot_det' : (('longitude', 'latitude', 'time'), dot_det.values),
      'ug' : (('longitude', 'latitude', 'time'), alt_crop.ug.values),
      'vg' : (('longitude', 'latitude', 'time'), alt_crop.vg.values),
      'ug_det' : (('longitude', 'latitude', 'time'), ug_det),
      'vg_det' : (('longitude', 'latitude', 'time'), vg_det)},
      coords={'time' : dot.time.values,
      'longitude' : dot.longitude.values,
      'latitude' : dot.latitude.values,
      'elat' : alt_crop.edge_lat.values,
      'elon' : alt_crop.edge_lon.values})
  else:
    altim = xr.Dataset({'dot' : (('longitude', 'latitude', 'time'), dot.values),
      'dot_det' : (('longitude', 'latitude', 'time'), dot_det.values),
      'ug_det' : (('longitude', 'latitude', 'time'), ug_det),
      'vg_det' : (('longitude', 'latitude', 'time'), vg_det)},
      coords={'time' : dot.time.values,
      'longitude' : dot.longitude.values,
      'latitude' : dot.latitude.values,
      'elat' : alt_crop.edge_lat.values,
      'elon' : alt_crop.edge_lon.values})

  return altim

def extract_dot_ug(geoidtype, satellite, sigma, date_start, date_end):

  altfile = 'dot_' + satellite + '_30bmedian_' + geoidtype + '_sig' + str(sigma) + '.nc'
  with xr.open_dataset(griddir+altfile) as alt:
      print(alt.keys())

  print("Crop altimetry to \n\n > > %s - %s\n\n" % (date_start, date_end))

  alt_crop = alt.sel(time=slice(date_start, date_end))
  dot = alt_crop.dot

  print("Computing gvel .. ")
  ug, vg = geos_vel(dot.values, dot.longitude.values, dot.latitude.values)

  # - - - - - - - - - - - - - - - - - 
  altim = xr.Dataset({'dot' : (('lon', 'lat', 'time'), dot.values),
    'ug' : (('lon', 'lat', 'time'), ug),
    'vg' : (('lon', 'lat', 'time'), vg)},
    coords={'time' : dot.time.values,
    'lon' : dot.longitude.values,
    'lat' : dot.latitude.values,
    'elat' : alt_crop.edge_lat.values,
    'elon' : alt_crop.edge_lon.values})

  return altim


"""
Old way of computing geos vel - forward diff only
  # dx = gsw.distance(alt_glon, alt_glat, axis=0)
  # dy = gsw.distance(alt_glon, alt_glat, axis=1)

  # dsla_x = dot_det.values[1:] - dot_det.values[:-1]
  # dsla_y = dot_det.values[:, 1:, :] - dot_det.values[:, :-1, :]

  # # take differences along one axis ..
  # dsla_dx = dsla_x/dx[:, :, np.newaxis]
  # dsla_dy = dsla_y/dy[:, :, np.newaxis]

  # # average along the other direction
  # dsla_dx_mid = 0.5*(dsla_dx[:, 1:, :]+dsla_dx[:, :-1, :])
  # dsla_dy_mid = 0.5*(dsla_dy[1:, :, :]+dsla_dy[:-1, :, :])

  # # scale by f and g
  # f = gsw.f(alt_gmlat)
  # g =  gsw.grav(alt_gmlat, 0)

  # vg = dsla_dx_mid * (g/f)[:, :, np.newaxis]
  # ug = (-1) * dsla_dy_mid * (g/f)[:, :, np.newaxis]
"""
  # - - - - - - - - - - - - - - - - - 