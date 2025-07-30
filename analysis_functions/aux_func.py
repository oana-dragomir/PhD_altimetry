"""
Last modified: 30 May 2025
"""
import numpy as np
from numpy import ma

import xarray as xr


import pandas as pd

import scipy.stats as ss
import scipy as sp
from scipy.stats.stats import pearsonr
from scipy.interpolate import RegularGridInterpolator as rginterp

import matplotlib.dates as mdates

import scipy.interpolate as itp
from scipy.ndimage.filters import gaussian_filter as gfilt

import sys
import os
import gsw

from area import area


def decode_cf_time(var, units):
  """
  Use xr.decode_cf to convert time as a number with a reference
  to datetime64[ns]. Uses a units attribute like 'days since 0001-01-01'

  > returns an array of datetime64[ns]
  """
  attrs = {"units": units}
  t = xr.Dataset({'time':('time', var, attrs)})
  t_cf = xr.decode_cf(t)
  return t_cf.time.values

def detrend_DataArray(arr, confidence):
    """
    Calculate a linear least-squares regression and confidence
    interval for a given xarray.DataArray time series.
    The time array is converted to number of days from a datetime.
    The linear trend hence has units of array_units/day.

    arr : xarray.DataArray, 1D
          time series, must contain coordinate 'time'
    confidence : float
                 confidence interval, e.g. 0.95

    returns: de-trended xarray.DataArray 
    """

    # If input is a DataArray, proceed with the original logic
    ndays = mdates.date2num(list(arr.time.values))
    dt = ndays - ndays[0]  # for computing linear trend

    # mask nans if present
    arr_data = ma.masked_invalid(arr.data)
    # number of valid observations
    obs = ma.count(arr_data)

    # extract only valid data and their time indices from masked arrays
    arr_vals = arr_data[~arr_data.mask].squeeze()
    tim = dt[~arr_data.mask].squeeze()

    if obs > 2:  # time series must have at least 2 points
        slope, interc, r, pp, stderr = ss.linregress(tim, arr_vals)
        # confidence interval
        ci = stderr * sp.stats.t.ppf((1 + confidence) / 2., obs - 1)
    else:
        slope = interc = r = pp = ci = np.nan
        arr_det = arr

    arr_det = arr - (dt * slope + interc)

    return arr_det


def detrend_Dataset(arr, confidence):
    """
    Calculate a linear least-squares regression and confidence
    interval for a given xarray time series.
    The time array is converted to number of days from a datetime.
    The linear trend hence has units of array_units/day.

    arr : xarray.Dataset or xarray.DataArray, 1D
          time series, must contain coordinate 'time'
    confidence : float
                 confidence interval, e.g. 0.95

    returns: xarray.Dataset with intercept, slope, r-value, p-value, conf interval
            units of the array/day
    """

    # Check if input is a Dataset or DataArray
    if isinstance(arr, xr.Dataset):
        detrended = {}
        for var in arr.data_vars:
            var_det = detrend_DataArray(arr[var], confidence)
            detrended[var] = var_det

        # Combine results into a single Dataset
        arr_det = xr.Dataset(detrended)
    else:
        arr_det = detrend_dataArray(arr, confidence)

    return arr_det

def trend_ci(arr, confidence):
    """
    Calculate a linear least-squares regression and confidence 
    interval for a given xarray time series. 
    The time array is converted to number of dayas from a datetime.
    The linear trend hence has units of array_units/day.

    arr : array, 1D
          time series, must contain coordinate 'time'
    confidence : float
                 confidence interval, e.g. 0.95
    
    returns: xarray.Dataset with intercept, slope, r-value, p-value, conf interval
            units of the array/day
    """
    ndays = mdates.date2num(list(arr.time.values))
    dt = ndays - ndays[0] # for computing linear trend, regularly sampled array

    # mask nans if present
    arr_data = ma.masked_invalid(arr)
    # number of valid observations
    obs = ma.count(arr_data)

    # extract only valid data and their time indices from masked arrays
    arr_vals = arr_data[~arr_data.mask].squeeze()
    tim = dt[~arr_data.mask].squeeze()

    if obs > 2: # time serie must have at least 2 points
        slope, interc, r, pp, stderr = ss.linregress(tim, arr_vals)
        # confidence interval
        ci = stderr * sp.stats.t.ppf((1+confidence)/2., obs-1)
        
        # residuals
        #trend = interc + slope*tim
        #res = arr_vals - trend
        #mean_res, sem_res = ma.mean(res), res.std(ddof=1)/np.sqrt(obs-1)

    else:
        slope = interc = r = pp = ci = np.nan
        arr_det = arr
    ds = xr.Dataset({'slope' : slope,
                    'intercept' : interc,
                    'r_coeff' : r,
                    'p_val' : pp,
                    'ci' : ci})

    ds.slope.attrs["long_name"] = 'slope of regression line'
    ds.r_coeff.attrs["long_name"] = 'correlation coefficient'
    ds.p_val.attrs["long_name"] = ("two-sided p-value for a hypothesis "
        "test whose null hypothesis is that the slope is zero," 
        "using Wald Test with t-distribution of the test statistic")
    ds.ci.attrs["long_name"] = 'confidence interval'

    arr_det = arr  - (dt * slope + interc)

    return ds, arr_det

def trend_ci_np(ndays, arr, confidence):
    """
    Calculate a linear least-squares regression and confidence 
    interval for a given xarray time series. 
    The time array is converted to number of dayas from a datetime.
    The linear trend hence has units of array_units/day.

    arr : array, 1D
          time series, must contain coordinate 'time'
    confidence : float
                 confidence interval, e.g. 0.95
    
    returns: xarray.Dataset with intercept, slope, r-value, p-value, conf interval
            units of the array/day
    """
    dt = ndays - ndays[0] # for computing linear trend, regularly sampled array

    # mask nans if present
    arr_data = ma.masked_invalid(arr)
    # number of valid observations
    obs = ma.count(arr_data)

    # extract only valid data and their time indices from masked arrays
    arr_vals = arr_data[~arr_data.mask].squeeze()
    tim = dt[~arr_data.mask].squeeze()

    if obs > 2: # time serie must have at least 2 points
        slope, interc, r, pp, stderr = ss.linregress(tim, arr_vals)
        # confidence interval
        ci = stderr * sp.stats.t.ppf((1+confidence)/2., obs-1)
        
        # residuals
        #trend = interc + slope*tim
        #res = arr_vals - trend
        #mean_res, sem_res = ma.mean(res), res.std(ddof=1)/np.sqrt(obs-1)

    else:
        slope = interc = r = pp = ci = np.nan
        arr_det = arr
    ds = xr.Dataset({'slope' : slope,
                    'intercept' : interc,
                    'r_coeff' : r,
                    'p_val' : pp,
                    'ci' : ci})

    ds.slope.attrs["long_name"] = 'slope of regression line'
    ds.r_coeff.attrs["long_name"] = 'correlation coefficient'
    ds.p_val.attrs["long_name"] = ("two-sided p-value for a hypothesis "
        "test whose null hypothesis is that the slope is zero," 
        "using Wald Test with t-distribution of the test statistic")
    ds.ci.attrs["long_name"] = 'confidence interval'

    arr_det = arr  - (dt * slope + interc)

    return ds, arr_det

def detrend2d(arr, tim):
    """
    remove linear trend from 2d array (time x depth)
    arr : shape(r, c)
    tim : shape(r)
    """
    t, d = arr.shape

    slope, interc, ci = [np.ones((d,)) for _ in range(3)]
    for i in range(d):
      arr_trend, _ = trend_ci(dt, arr[:, i], 0.95)
      slope[i] = arr_trend.slope.values
      interc[i] = arr_trend.intercept.values
      ci[i] = arr_trend.ci.values

    trend = tim[:, np.newaxis] * slope[np.newaxis, :] + interc[np.newaxis, :]
    arr_det = arr - trend

    return arr_det, slope, ci

def detrend3d(var):
    """
    remove linear trend (95% CI) from 3d array (lon, lat, tim)
    returns detrended array, slope, pval
    """
    
    date = var.time.dt.strftime("%m.%Y").values
    ndays = mdates.date2num(list(var.time.values))
    dt = ndays - ndays[0]

    # DIMENSIONS
    londim, latdim, timdim = var.shape

    # compute linear trend at every grid point
    interc, slope, ci, pval = [ma.zeros((londim, latdim)) for _ in range(4)]
    for r in range(londim):
        for c in range(latdim):
            arr_trend, _ = trend_ci(var[r,c,:], 0.95)
            interc[r, c] = arr_trend.intercept.values
            slope[r, c] = arr_trend.slope.values
            pval[r, c] = arr_trend.p_val.values

    var_det = var  - (dt[np.newaxis, np.newaxis, :] * slope[:, :, np.newaxis] + interc[:, :, np.newaxis])
            
    return var_det, slope, pval


def grid_area(glon, glat):
    """
    Compute area of every grid cell.
    
    glat, glon: lat/lon at bin edges
    returns an array of the same size as the grid
    """
    lo, la = glon.shape
    area_grid = np.ones((lo-1, la-1))

    for i in range(lo-1):
        for j in range(la-1):
            lon0, lon1 = glon[i, j], glon[i+1, j]
            lat0, lat1 = glat[i, j], glat[i, j+1] 
            obj = {'type':'Polygon', 'coordinates':[[[lon0, lat0], [lon0, lat1],
                                                     [lon1, lat1], [lon1, lat0],
                                                     [lon0, lat0]]]}
            area_grid[i, j] = area(obj)
    return area_grid

def area_weighted_avg(arr, lon, lat):
    """
    arr : 2d array, can be masked,
        shape(r, c)
    lon, lat : 1d array with coordinates at bin edges
    
    returns a 2d array of size (r, c)
    """
    var = ma.masked_invalid(arr)
    glat, glon = np.meshgrid(lat, lon) 
    area_var_tot = grid_area(glon, glat)
    #apply mask 
    area_var = area_var_tot[~var.mask]
    aw_avg = (var*area_var_tot).sum()/area_var.sum()
    return aw_avg

#-------------------------------------------------------------------------------

def interp_nan(var,x,y):
    """
    interpolate non-nan values from neighbours at nan loci (e.g. to be used
    before Gaussian filtering)

    From Clement Vic (30/11/2018)
    """
    var  = np.ma.masked_invalid(var)
    xi   = x[~var.mask]
    yi   = y[~var.mask]
    vari = var[~var.mask]
    varnew = itp.griddata((xi,yi),vari.ravel(),(x,y),method='nearest')
    return varnew
#-------------------------------------------------------------------------------
# GAUSSIAN FILTERING
#-------------------------------------------------------------------------------
## 2D Gaussian filter
# Choose a sigma = 2 (i.e. 100km) (Armitage 2016, Dotto 2018)



def gaussian_filt(initial_arr, sigma, mode):
    """
    Multidimensional Gaussian filter from 
    scipy.ndimage.filters.

    initial_arr : array_like
        must not contain nans or be a masked array
    sigma : scalar
        standard deviation for the Gaussian kernel
    mode : str
        'reflect', 'wrap', 'constant', nearest', 'mirror'

    Returns:
        filt_arr_cut : the filtered array
    """    
    # append a few grid points at each longitude end
    # with data from the opposite end
    # similar to 'wrap' mode but avoiding wrapping
    # data at latitude ends
    n_append = 10*sigma
    extended_arr = np.vstack((initial_arr[-n_append:, :], initial_arr,
                              initial_arr[:n_append, :]))
    filt_arr = gfilt(extended_arr, sigma=sigma, order=0, 
                     mode=mode)
    filt_arr_cut = filt_arr[n_append:-n_append, :]
    
    return filt_arr_cut

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    # crop the ends to match the size of the initial signal
    yy = y[int(window_len/2):-int(window_len/2)]
    #print(yy.shape)
    return yy

def rotate_vec(u, v, alpha_deg, dir):
    """
    rotate a 2D vector clockwise by an angle
    parameters:
        u, v : velocity components
        alpha_deg : angle in degrees

   returns: rotated u, v 
    """
    alpha = np.radians(alpha_deg)
    if dir == 'clockwise':
        u_rot = u * np.cos(alpha) + v * np.sin(alpha)
        v_rot = (-1)*u * np.sin(alpha) + v * np.cos(alpha)
    elif dir == 'anticlockwise':
        u_rot = u * np.cos(alpha) - v * np.sin(alpha)
        v_rot = u * np.sin(alpha) + v * np.cos(alpha)        

    return u_rot, v_rot

def rotate_vec_xr(u, v, alpha_deg, dir):
    """
    rotate a 2D vector clockwise by an angle from a fixed frame
    parameters:
        u, v : velocity components
        alpha_deg : angle in degrees, +ve clockwise from zonal/meridional dir

   returns: rotated u, v in an xarray dataset
    """
    alpha = np.radians(alpha_deg)
    time = u.time
    u = u.values
    v = v.values

    if dir == 'clockwise':
        u_rot = u * np.cos(alpha) + v * np.sin(alpha)
        v_rot = (-1)*u * np.sin(alpha) + v * np.cos(alpha)
    elif dir == 'anticlockwise':
        u_rot = u * np.cos(alpha) - v * np.sin(alpha)
        v_rot = u * np.sin(alpha) + v * np.cos(alpha)   

    ds = xr.Dataset({"u" : ('time', u_rot),
        "v" : ("time", v_rot)},
        coords={"time" : time})
    return ds

def rotate_frame(u, v, alpha_deg, direction):
    """
    Return orthogonal vector components projected onto a rotated frame.
    (this is the correct function to use for obtaining alon/across trough components)
    
    Parameters:
        u, v : zonal/meridional velocity components (xarray)
        alpha_deg : angle in degrees
        direction : 'clockwise' or 'anticlockwise'

   returns: u, v projected onto the rotated coord frame
    """
    alpha = np.radians(alpha_deg)
    
    time = u.time
    u = u.values
    v = v.values

    if direction == 'clockwise':
        u_rot = u * np.cos(alpha) - v * np.sin(alpha)
        v_rot = u * np.sin(alpha) + v * np.cos(alpha)
    elif direction == 'anticlockwise':
        u_rot = u * np.cos(alpha) + v * np.sin(alpha)
        v_rot = (-1)* u * np.sin(alpha) + v * np.cos(alpha)        
        
    ds = xr.Dataset({"u" : ('time', u_rot),
        "v" : ("time", v_rot)},
        coords={"time" : time})
    return ds

def remove_nan_xr(arr1, arr2):
    """
    remove nans from two arrays. arrays can have different time start/end points
    but they should have some overlap.

    var1, var2: xarray datasets with time coordinate named 'time'; 
                monthly time series
    v1, v2: arrays with matching length and no nans

    returns v1, v2, time
    """
    var1 = arr1.copy()
    var2 = arr2.copy()

    v1_start, v1_end = var1.time[0], var1.time[-1]
    v2_start, v2_end = var2.time[0], var2.time[-1]
    
    t_start, t_end = v1_start, v1_end
    if v1_start < v2_start:
        t_start = v2_start
    if v1_end > v2_end:
        t_end = v2_end

    var1 = var1.sel(time=slice(t_start, t_end))
    var2 = var2.sel(time=slice(t_start, t_end))
    
    var1 = var1.values
    var2 = var2.values
    
    var1[np.isnan(var2)] = np.nan
    var2[np.isnan(var1)] = np.nan
    
    v1 = var1[~np.isnan(var1)]
    v2 = var2[~np.isnan(var2)]
    
    return v1, v2

def pearsonr_nan(arr1, arr2):
    """
    Compute the pearson correlation coefficient and significance 
    for two xarrays. arrays can have different time start/end points
    but they should have some overlap.

    var1, var2: xarray datasets with time coordinate named 'time'

    returns r, p
    """
    var1 = arr1.copy()
    var2 = arr2.copy()

    v1, v2 = remove_nan_xr(var1, var2)

    if np.count_nonzero(v1) < 2 or np.count_nonzero(v2) < 2:
        r, p = np.nan, np.nan 
        print("the arrays are either null or have less than 2 valid values")
    
    else:
        r, p = pearsonr(v1, v2)
    return r, p

def add_monthly_lag(var, lag):
    var_lag = var.shift(time=lag)    
    return var_lag

def add_daily_lag(var, lag):
    var_lag = var.shift(time=lag)    
    return var_lag

def mad(arr):
    """
    Compute the median absolute deviation of an array.
    """
    data = ma.masked_invalid(arr)
    data = data[~data.mask].squeeze()

    median = np.median(data)
    residuals = data - median
    mad = np.median(abs(residuals))
    return mad


def seasonal_avg(arr_months, arr, stats):
    # initialise arrays
    avg, spread = [ma.ones((12)) for _ in range(2)]
    arr = ma.masked_invalid(arr)
    if stats=='median':
        for i in range(12):
            mm = ma.masked_not_equal(arr_months, i+1)
            arr_m = arr[~mm.mask]
            avg[i] = np.median(arr_m)
            spread[i] = mad(arr)
    elif stats=='mean':
        for i in range(12):
            mm = ma.masked_not_equal(arr_months, i+1)
            arr_m = arr[~mm.mask]
            avg[i] = np.mean(arr_m)
            spread[i] = ma.std(arr_m, ddof=1)
    else:
        avg[:] = spread[:] = ma.masked
        print("typo")
    return avg, spread