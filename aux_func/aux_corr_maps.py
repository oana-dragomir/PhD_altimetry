"""
> compute correlation maps
> plot correlation maps

Last modified: 13 Apr 2021
"""
import numpy as np
from numpy import ma

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import xarray as xr

import scipy
from scipy.stats import pearsonr
import scipy.special as special

import sys

# - - - - - - - - - - - - - - - - -
voldir = '/Users/ocd1n16/PhD_local/data/'
#voldir = '/Users/ocd1n16/PhD_local/temporary/'
topodir = voldir + 'topog/'

localdir = '/Users/ocd1n16/PhD_local/'

auxscriptdir = localdir + 'scripts/aux_func/'
sys.path.append(auxscriptdir)
import aux_stereoplot as st

#-------------------------------------------------------------------
# bathymetry file
#-------------------------------------------------------------------
with xr.open_dataset(topodir + 'coarse_gebco_p5x1_latlon.nc') as topo:
    print(topo.keys())
tglat, tglon = np.meshgrid(topo.lat, topo.lon)

# - - - - - - - - - - - - - - - - -
# S1 mooring location
s1_lon = -116.358
s1_lat = -72.468

plt.ion()
#-------------------------------------------------------------------

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

def r_map_ts(arr, ts):
    """
    ts must have size tim
    arr must have the shape (lon, lat, tim)
    """
    arr = ma.masked_invalid(arr)
    ts = ma.masked_invalid(ts)
    
    corr_map, pval_map = [ma.zeros(arr.shape[:-1]) for _ in range(2)]
    londim, latdim = arr.shape[:-1]
    for i in range(londim):
        for j in range(latdim):
            map_ij = arr[i, j, :]
            
            ts_m = ma.masked_array(ts, map_ij.mask)
            map_ij = ma.masked_array(map_ij, ts_m.mask)

            if ts.count() > 3 and map_ij.count() >3:
                corr_map[i, j], pval_map[i, j] = pearsonr(ts_m.compressed(),
                                                          map_ij.compressed())
            else:
                corr_map[i, j] = pval_map[i, j] = ma.masked
    return corr_map, pval_map

def r_map_ts_filt(arr, ts, nfilt):
    """
    ts must have size tim
    arr must have the shape (lon, lat, tim)
    """
    TINY = 1.0e-20
    alternative='two-sided'

    arr = ma.masked_invalid(arr)
    ts = ma.masked_invalid(ts)
    
    corr_map, pval_map = [ma.zeros(arr.shape[:-1]) for _ in range(2)]
    londim, latdim = arr.shape[:-1]
    for i in range(londim):
        for j in range(latdim):
            map_ij = arr[i, j, :]
            
            ts_m = ma.masked_array(ts, map_ij.mask)
            map_ij = ma.masked_array(map_ij, ts_m.mask)

            if ts.count() > 3 and map_ij.count() >3:
                corr_map[i, j],_ = pearsonr(ts_m.compressed(),
                                            map_ij.compressed())
                #ab = (len(ts)/2 - 1)/nfilt
                #pval_map[i, j] = 2*special.btdtr(ab, ab, 0.5*(1 - abs(np.float64(corr_map[i, j]))))

                #r = 0.041
                #n = 24
                df = (len(ts)-2)/nfilt  # Number of degrees of freedom
                # n-2 degrees of freedom because 2 has been used up
                # to estimate the mean and standard deviation
                t = corr_map[i,j] * np.sqrt(df / ((1.0 - corr_map[i,j] + TINY)*(1.0 + corr_map[i,j] + TINY)))
                #t, pval_map[i,j] = scipy.stats.stats._ttest_finish(df, t, alternative)
                pval_map[i,j] = scipy.stats.t.sf(np.abs(t), df)*2  # two-sided pvalue = Prob(abs(t)>tt)

            else:
                corr_map[i, j] = pval_map[i, j] = ma.masked
    return corr_map, pval_map


def pearsonr_filt(ts1, ts2, nfilt):
    """
    two xarray time series
    """
    TINY = 1.0e-20
    alternative='two-sided'

    ts1, ts2 = remove_nan_xr(ts1, ts2)

    if np.count_nonzero(~np.isnan(ts1)) > 3 and np.count_nonzero(~np.isnan(ts2)) >3:
        corr,_ = pearsonr(ts1, ts2)
        df = (len(ts1)-2)/nfilt  # Number of degrees of freedom
        # to estimate the mean and standard deviation
        t = corr * np.sqrt(df / ((1.0 - corr + TINY)*(1.0 + corr + TINY)))
        pval = scipy.stats.t.sf(np.abs(t), df)*2  # two-sided pvalue = Prob(abs(t)>tt)
    else:
        corr = pval = np.nan
    return corr, pval


def set_start_end_dates(arr1, arr2):
    # initialize times from array
    date_start, date_end = arr1.time[0], arr1.time[-1]
    arr2_start, arr2_end = arr2.time[0], arr2.time[-1]
    
    if arr2_start > date_start:
        date_start = arr2_start
    if arr2_end < date_end:
        date_end = arr2_end
        
    if date_end < date_start:
        print("Incorrect date interval - end date precedes start date!")
    print("Time period: ")
    print(date_start.dt.strftime("%Y-%m-%d").values,
        date_end.dt.strftime("%Y-%m-%d").values)
    return date_start, date_end

def crop_arrays(arr1, arr2):
    date_start, date_end = set_start_end_dates(arr1, arr2)
    arr1_crop = arr1.sel(time=slice(date_start, date_end))
    arr2_crop = arr2.sel(time=slice(date_start, date_end))
    
    return arr1_crop, arr2_crop

def rmap_altim_moor(arr0, alt0, nfilt):
    # PLOTS    
    arr, alt = crop_arrays(arr0, alt0)
    
    # compute corr maps
    if nfilt>0:
        corr, pval = r_map_ts_filt(alt.values,
                              arr.values, nfilt)  
    else:      
        corr, pval = r_map_ts(alt.values,
                              arr.values)
    ds = xr.Dataset({"corr" : (("longitude", "latitude"), corr),
                "pval" : (("longitude", "latitude"), pval)},
                   coords={"longitude": alt.longitude.values,
                          "latitude": alt.latitude.values})
    
    return ds

#-------------------------------------------------------------------
def plot_corr(var, pval_var, var_glon, var_glat, cbar_range, cmap, color):
  cbar_units = 'r'

  fig, ax, m = st.spstere_plot(var_glon, var_glat, var,
                         cbar_range, cmap, cbar_units, color)
  cs1 = m.contourf(var_glon, var_glat, pval_var,
                   levels=[0., 0.05], colors='none', 
                   hatches=['////', None], zorder=2, latlon=True)
  cs2 = m.contour(var_glon, var_glat, pval_var,
                  levels=[0., 0.05], colors='k',
                  zorder=2, linewidths=.8, latlon=True)

  # location of S1
  #m.scatter(s1_lon, s1_lat,
  #          marker='*', s=60, 
  #          latlon=True,
  #          c='gold', edgecolor='k', 
  #          lw=.5, zorder=7)

  #bathymetry contours
  #lp = m.contour(tglon, tglat, topo.elevation,
  #          levels=[-4000, -1000],
  #          colors=['slategray', 'lightgreen'],
  #          latlon=True, zorder=2)
  #lp_labels = ['4000 m', '1000 m']
  #for i in range(len(lp_labels)):
  #    lp.collections[i].set_label(lp_labels[i])
  #ax.legend(loc='lower right', fontsize=9,
  #  bbox_to_anchor=(.1, .95))

  return fig, ax , m

def plot_signif_corr(var, pval_var, var_glon, var_glat, cbar_range, cmap):
  cbar_units = 'r'
  cbar_extend = 'both'

  fig, ax, m = st.spstere_plot(var_glon, var_glat, var,
                         cbar_range, cmap, cbar_units, cbar_extend)
  # location of S1
  m.scatter(s1_lon, s1_lat,
            marker='*', s=60, 
            latlon=True,
            c='gold', edgecolor='k', 
            lw=.5, zorder=7)

  #bathymetry contours
  lp = m.contour(tglon, tglat, topo.elevation,
            levels=[-4000, -1000],
            colors=['slategray', 'mediumvioletred'],
            latlon=True, zorder=2)
  lp_labels = ['4000 m', '1000 m']
  for i in range(len(lp_labels)):
      lp.collections[i].set_label(lp_labels[i])
  ax.legend(loc='lower right', fontsize=9,
    bbox_to_anchor=(.1, .95))

  return fig, ax , m