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
from palettable.scientific.diverging import Vik_20

from eofs.xarray import Eof

import sys
import gsw

import analysis_functions.aux_func as fc
import analysis_functions.aux_stereoplot as st

# Directories
voldir = '/Volumes/SamT5/PhD/PhD_data/'
griddir = voldir + 'altimetry_cpom/3_grid_dot/'


plt.ion()
# - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - 
def detrend_fc(var, var_eglon, var_eglat, plot=False):
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

  if plot:
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

  Extends the longitudinal array to handle circumpolar data
  Uses forward differences and averages them;
  compute distances on the circle using gsw.distance
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

def geos_vel_gradients(var):
  """
  Another method to compute geostrophic velocity but I did not choose this one
  because it computes the velocity using SLA gradients and it calculates 
  the velocity at the mid-point location so I would have to use a different grid 
  for velocity or interpolate it.
  """
    
  glat, glon = np.meshgrid(var.lat.values, var.lon.values)
  gmlat = 0.5*(glat[1:, 1:] + glat[:-1, :-1])
  gmlon = 0.5*(glon[1:, 1:] + glon[:-1, :-1])

  dx = gsw.distance(glon, glat, axis=0)
  dy = gsw.distance(glon, glat, axis=1)

  dsla_x = var.values[1:] - var.values[:-1]
  dsla_y = var.values[:, 1:, :] - var.values[:, :-1, :]

  # take differences along one axis ..
  dsla_dx = dsla_x/dx[:, :, np.newaxis]
  dsla_dy = dsla_y/dy[:, :, np.newaxis]

  # average along the other direction
  dsla_dx_mid = 0.5*(dsla_dx[:, 1:, :]+dsla_dx[:, :-1, :])
  dsla_dy_mid = 0.5*(dsla_dy[1:, :, :]+dsla_dy[:-1, :, :])

  # scale by f and g
  f = gsw.f(gmlat)
  g =  gsw.grav(gmlat, 0)

  vg_rec = dsla_dx_mid * (g/f)[:, :, np.newaxis]
  ug_rec = (-1) * dsla_dy_mid * (g/f)[:, :, np.newaxis]

  return ug_rec, vg_rec, gmlon, gmlat
# - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - 
def detrend_dot_geosvel(geoidtype, satellite, sig, date_start, date_end, plot):
  """
  This function uses the filenaming convention for the altimetry data to remove 
  a linear trend from DOT and geostrophic velocity components.
  Can specify a start and end date to crop it (between 2002.07 and 2018.10) and
  can also specify to plot the linear trend.

  Returns:
  xarray dataset with the original (if combined product is used) and 
  detrended variables
  """

  altfile = 'dot_' + satellite + '_30bmedian_' + geoidtype + '_sig' + str(sig) + '.nc'
  print("Processing file: %s" % altfile)
  alt = xr.open_dataset(griddir+altfile) 

  print("\nCrop altimetry to \n > > %s - %s" % (date_start, date_end))

  alt_crop = alt.sel(time=slice(date_start, date_end))
  dot = alt_crop.dot
  lat = alt_crop.latitude.values
  lon = alt_crop.longitude.values

  # sea level anomaly
  sla = dot - dot.mean("time")

  # GRID coordinates
  eglat, eglon = np.meshgrid(alt_crop.edge_lat, alt_crop.edge_lon)
  #-----------------------------------
  # DE-TREND function
  #-----------------------------------
  sla_det = detrend_fc(sla, eglon, eglat, plot)


  print("\nComputing gvel .. ")
  ug_det, vg_det = geos_vel(sla_det.values, lat, lon)
  #-----------------------------------

  if satellite == 'all': 
    altim = xr.Dataset({'dot' : (('longitude', 'latitude', 'time'), dot.values),
      'sla' : (('longitude', 'latitude', 'time'), sla.values),
      'sla_det' : (('longitude', 'latitude', 'time'), sla_det.values),
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
      'sla' : (('longitude', 'latitude', 'time'), sla.values),
      'sla_det' : (('longitude', 'latitude', 'time'), sla_det.values),
      'ug_det' : (('longitude', 'latitude', 'time'), ug_det),
      'vg_det' : (('longitude', 'latitude', 'time'), vg_det)},
      coords={'time' : dot.time.values,
      'longitude' : dot.longitude.values,
      'latitude' : dot.latitude.values,
      'elat' : alt_crop.edge_lat.values,
      'elon' : alt_crop.edge_lon.values})

  print('Ready!')

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



def extract_eof(n_eof, sla, lat, lon):
    alt_var = sla

    glat, glon = np.meshgrid(lat, lon)

    # apply weighting to account for meridians converging at high lat
    coslat = np.cos(np.deg2rad(lat)).clip(0., 1.)
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(alt_var.T, weights=wgts)

    #eof = solver.eofsAsCorrelation(neofs=15)
    eof = solver.eofsAsCovariance(neofs=10)
    pc = solver.pcs(npcs=10, pcscaling=1)
    variance_frac = solver.varianceFraction()

    # - - - - - - - - - - - - - - - - - 
    # > > reconstruct DOT using the first 3 modes
    rec_sla = solver.reconstructedField(n_eof)

    ug_rec, vg_rec = geos_vel(rec_sla.T.values,
                            rec_sla.latitude.values,
                            rec_sla.longitude.values)

    # - - - - - - - - - - - - - - - - - 
    rec = xr.Dataset({'sla' : (('longitude', 'latitude', 'time'), rec_sla.T.values),
                      'ug' : (('longitude', 'latitude', 'time'), ug_rec),
                      'vg' : (('longitude', 'latitude', 'time'), vg_rec)},
                      coords={'longitude' : rec_sla.longitude.values,
                              'latitude' : rec_sla.latitude.values,
                              'time' : rec_sla.time.values})
    return rec


def plot_sla_ts(sla_var, uc_var, clim_idx, clim_label, title):
    xtim1 = sla_var.time.values
    xtim2 = uc_var.time.values
    xtim3 = clim_idx.time.values
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(xtim1, sla_var.values, c='m', label='SLA')
    ax.plot(xtim2, uc_var.values, c='k', label='UC')
    ax.plot(xtim3, clim_idx.values, c='b', label=clim_label)
    ax.set_title(title, loc='left')
    ax.legend()
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

    ax.format_xdata = mdates.DateFormatter('%Y')
    fig.autofmt_xdate()

    ax.grid(True, which='major', lw=1., ls='-')
    ax.grid(True, which='minor', lw=1., ls=':')
    plt.tight_layout()