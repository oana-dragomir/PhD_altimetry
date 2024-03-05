"""
Climate inddices
- processing of ASL
- plotting time series: ASL, SAM, SOI, Nino

Last modified: 2 Dec 2020
"""

import numpy as np
from numpy import ma

import xarray as xr
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import detrend

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

import datetime
# - - - - - - - - 
# Directories
voldir = '/Users/ocd1n16/PhD_local/data/'
idxdir = voldir + 'climate_indices/'

localdir = '/Users/ocd1n16/PhD_local/'
scriptdir = localdir + 'scripts/'

figdir = localdir + 'data_notes/Figures_v8/'
#----------------------------------------------
date_start = '2002-10-01'
date_end = '2018-10-31'
#----------------------------------------------
# - - - - - - - - 
#----------------------------------------------
# Amundsen Sea Low
#----------------------------------------------
asl_csv = 'asli_era5_v3.csv'

asl_pd = pd.read_csv(idxdir + asl_csv)
asl = xr.Dataset({'lon' : ('time', asl_pd.lon.values),
  'lat' : ('time', asl_pd.lat.values),
  'ActCenPres' : ('time', asl_pd.ActCenPres.values),
  'SectorPres' : ('time', asl_pd.SectorPres.values),
  'RelCenPres' : ('time', asl_pd.RelCenPres.values)},
  coords={'time' : pd.to_datetime(asl_pd.time.values)})

# crop to a set time
asl_crop = asl.sel(time=slice(date_start, date_end))

# standardise
asl_crop_anom = asl_crop - asl_crop.mean('time')
asl_crop_stand = asl_crop_anom / asl_crop_anom.std(ddof=1)

asl_crop_mclim = asl_crop.groupby('time.month') - asl_crop.groupby('time.month').mean('time')
asl_crop_mclim_stand = asl_crop_mclim.groupby('time.month') / asl_crop_anom.groupby('time.month').std(ddof=1)

# - - - - - - - - 
def plot_asl(var, time, label):
	fig, ax = plt.subplots(figsize=(12, 2))
	ax.plot(x, var, label=label, c='k', zorder=3)
	ax.scatter(x[var<-1], var[var<-1], c='b', zorder=4)
	ax.scatter(x[var>1], var[var>1], c='r', zorder=4)
	ax.axhline(0, c='grey', ls=':', zorder=1)
	ax.axvline(pd.to_datetime('2010-03-01'), c='cornflowerblue')
	ax.axvline(pd.to_datetime('2015-12-01'), c='cornflowerblue')

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
	ax.set_xlim(x[0]-np.timedelta64(60, 'D'), 
	    x[-1] + np.timedelta64(60, 'D'))

	ax.format_xdata = mdates.DateFormatter('%Y')
	fig.autofmt_xdate()

	ax.grid(True, which='major', lw=1., ls='-')
	ax.grid(True, which='minor', lw=1., ls=':')
	
	plt.tight_layout()
	ax.legend()

x = asl_crop_stand.time
plot_asl(asl_crop_stand.lon, x, 'lon')
plot_asl(asl_crop_stand.lat, x, 'lat')
plot_asl(asl_crop_stand.ActCenPres, x, 'ActCenPres')
plot_asl(asl_crop_stand.SectorPres, x, 'SectorPres')
plot_asl(asl_crop_stand.RelCenPres, x, 'RelCenPres')

#----------------------------------------------
# SAM, SOI, Nino (3, 3.4, 4)
#----------------------------------------------
with xr.open_dataset(idxdir + 'climate_indices_79_19.nc') as clim_idx:
  print(clim_idx)

indx = clim_idx.sel(time=slice(date_start, date_end))

# - - - - - - - - 
# standardise data (remove mean, divide by std; mean=0, std=1)
indx_anom = indx - indx.mean()
indx_stand = indx_anom / indx.std(ddof=1)

indx_mclim = indx.groupby('time.month') - indx.groupby('time.month').mean('time')
indx_mclim_stand = indx_mclim / indx_mclim.std('time', ddof=1)

xtim = indx.time.values
plot_asl(indx_stand.sam, xtim, 'SAM')
plot_asl(indx_stand.soi_ncep, xtim, 'SOI-NCEP')
plot_asl(indx_stand.nino3, xtim, 'Nino 3')
plot_asl(indx_stand.nino34, xtim, 'Nino 3.4')

#----------------------------------------------
# SLA time series
#----------------------------------------------
sla = xr.open_dataset(idxdir + 'timeseries_sla_coast_off.nc')

# cross-spectrum
from scipy import signal

f, Pxy = signal.csd(sla.shelf_anom_stand.values, indx_anom.soi_ncep.values)

fig, ax = plt.subplots()
ax.semilogy(f, np.abs(Pxy))


from pycurrents.num import spectra

def spectrum1(h, dt=1):
    """
    First cut at spectral estimation: very crude.
    
    Returns frequencies, power spectrum, and
    power spectral density.
    Only positive frequencies between (and not including)
    zero and the Nyquist are output.
    """
    nt = len(h)
    npositive = nt//2
    pslice = slice(1, npositive)
    freqs = np.fft.fftfreq(nt, d=dt)[pslice] 
    ft = np.fft.fft(h)[pslice]
    psraw = np.abs(ft) ** 2
    # Double to account for the energy in the negative frequencies.
    psraw *= 2
    # Normalization for Power Spectrum
    psraw /= nt**2
    # Convert PS to Power Spectral Density
    psdraw = psraw * dt * nt  # nt * dt is record length
    return freqs, psraw, psdraw

freqs1, ps1, psd1 = spectrum1(sla.shelf_anom_stand.values, dt=1)
freqs2, ps2, psd2 = spectrum1(indx_stand.sam.values, dt=1)


fig, axs = plt.subplots(ncols=2, sharex=True)
axs[0].loglog(freqs1, psd1, 'r',
              freqs2, psd2, 'b', alpha=0.5)
axs[1].loglog(freqs1, ps1, 'r', 
              freqs2, ps2, 'b', alpha=0.5)
axs[0].set_title('Power Spectral Density')
axs[1].set_title('Power Spectrum')
axs[1].axis('tight', which='x')


def spectrum2(h, dt=1, nsmooth=5):
    """
    Add simple boxcar smoothing to the raw periodogram.
    
    Chop off the ends to avoid end effects.
    """
    freqs, ps, psd = spectrum1(h, dt=dt)
    weights = np.ones(nsmooth, dtype=float) / nsmooth
    ps_s = np.convolve(ps, weights, mode='valid')
    psd_s = np.convolve(psd, weights, mode='valid')
    freqs_s = np.convolve(freqs, weights, mode='valid')
    return freqs_s, ps_s, psd_s
    

freqs1, ps1, psd1 = spectrum2(sla.off_anom_std.values, dt=1)
freqs2, ps2, psd2 = spectrum2(indx_stand.soi_ncep.values, dt=1)
fig, axs = plt.subplots(ncols=2, sharex=True)
axs[0].loglog(freqs1, psd1, 'r',
              freqs2, psd2, 'b', alpha=0.5)
axs[1].loglog(freqs1, ps1, 'r', 
              freqs2, ps2, 'b', alpha=0.5)
axs[0].set_title('Power Spectral Density')
axs[1].set_title('Power Spectrum')
axs[1].axis('tight', which='x');





