#! /usr/bin/env python3
"""


Last modified: 28 Aug 2019
"""


## libraries go here
from __future__ import division
import numpy as np
from numpy import ma

import datetime


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, draw, show  #interactive plots
from matplotlib.pyplot import cm
from matplotlib.offsetbox import AnchoredText
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.basemap import Basemap

from netCDF4 import Dataset, num2date, date2num

import scipy.stats as ss
from scipy.interpolate import interp1d
import scipy.interpolate as itp

from pycurrents.plot.maptools import llticks
from pycurrents.plot import maptools

import os
import sys

# my functions for computing the trend
import func_trend as fc

import pycwt as wavelet
from pycwt.helpers import find

#sys.path.append('/home/ocdragomir/Documents/python_extra/wavelets-master/wave_python')
#from waveletFunctions import wavelet, wave_signif

# rcParams
params = {
    'axes.labelsize': 12,
    'font.size': 13,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False
}
rcParams.update(params)

print(".... libraries read successfully")

#-----------------------------------------------------------------------------
# Define directories
# where data files are
workdir ='/home/ocdragomir/Documents/PhD/data/'
dirdata = workdir + 'SSH_SLA/'
offsetdir = workdir +  'offset/'
lmdir = workdir + 'land_masks/'
figdir = workdir + '../data_notes/Figures_v5/'

time_units = 'days since 1950-01-01 00:00:00.0'

#-----------------------------------------------------------------------------
filename = '4_all_DOT_egm08_mean_20thresh_e0702_c1110.nc'
#filename = 'CS2_DOT_MDT_mean_20thresh.nc'
#filename = '4_ENV_CS2_SLA_MSS_20thresh.nc'
#filename = '5_all_DOT_egm08_20thresh_e0702_c1110.nc'

data = Dataset(dirdata+filename, 'r+')
elon = data['edges_lon'][:]
elat = data['edges_lat'][:]
sla = ma.masked_invalid(data['sla'][:])
time = data['time'][:]
data.close()

# GRID
glat, glon = np.meshgrid(elat, elon)
mid_lon = 0.5*(elon[1:] + elon[:-1])
mid_lat = 0.5*(elat[1:] + elat[:-1])
gmlat, gmlon = np.meshgrid(mid_lat, mid_lon)

londim, latdim, timdim = sla.shape

# grid area
area_grid = fc.grid_area(glon, glat)

date = num2date(time, units=time_units, calendar='gregorian')

x, y = np.meshgrid(date, elat)

# LAND MASK
#-------------------------------------------------------------------------------
# lon grid is -180/180, 0.5 lat x 1 lon
# lm shape=(mid_lon, mid_lat)
# land=1, ocean=0
#-------------------------------------------------------------------------------
lm = Dataset(lmdir+'land_mask_gridded.nc', 'r+')
lmask = lm['landmask'][:]
lm.close()

# subtract the global SLR, i.e., 3.1 mm/yr
dt = time-time[0]
global_trend = dt*0.0031/365.25
sla = sla-global_trend


print("Computing linear trends ..")
#------------------------------------------------------------------
slope, conf = [ma.ones((sla.shape[:-1])) for _ in range(2)]
for i in range(londim):
    for j in range(latdim):
        slope[i, j], conf[i, j] = fc.trend_ci(time[:], sla[i, j, :], 0.95)[1:]

trend_mm_yr = slope*1e3*365.25
trend_mm_yr[lmask==1] = ma.masked

# 2. MAP area and GRID
#------------------------------------------------------------------
print("Defining map area ...\n")
# coastline resolution can vary from coarse to fine: c, l, i, h, f
m = Basemap(projection='spstere',
            boundinglat=-49.5,
            lon_0=180,
            resolution='c',
            round=True)

#------------------------------------------------------------------
# regional SLA means/trends
llon1 = [0, 80]
llon2 = [80, -170]
llon3 = [-95, -170]
llon4 = [-95, -60]
llon5 = [-60, 0]

# # # # # #
# monthly avg of the regional SLA
def regional_sla(lon_min, lon_max):
    reg_glon = ma.masked_outside(gmlon, lon_min, lon_max)
    reg_sla = ma.ones((sla.shape))
    for i in range(len(time)):
        reg_sla[: ,:, i] = ma.masked_array(sla[:, :, i], reg_glon.mask)

    mean_sla = ma.zeros((len(time)))
    for i in range(len(time)):
        aa = ma.masked_invalid(reg_sla[:, :, i] * area_grid)
        total_area = ma.sum(area_grid[~aa.mask])
        mean_sla[i] = ma.sum(aa)/total_area

    interc, trend, ci = fc.trend_ci(time, mean_sla, 0.95)
    mm_yr = trend*1e3*365.25
    print("region: %s - %s" % (lon_min, lon_max))
    print("Linear trend %s mm/yr \n" % round(mm_yr, 3))

    # trend line
    gtrend = interc+time*trend
    return mean_sla

msla1 = regional_sla(llon1[0], llon1[1])
msla2 = regional_sla(llon2[0], llon2[1])
msla3 = regional_sla(llon3[0], llon3[1])
msla4 = regional_sla(llon4[0], llon4[1])
msla5 = regional_sla(llon5[0], llon5[1])

"""
ion()
fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(date, msla1+0.25, label='{}'.format('%.0f,%.0f' % (llon1[0], llon1[1])))
ax.plot(date, msla2+0.2, label='{}'.format('%.0f,%.0f' % (llon2[0], llon2[1])))
ax.plot(date, msla3+0.13, label='{}'.format('%.0f,%.0f' % (llon3[0], llon3[1])))
ax.plot(date, msla4+0.06, label='{}'.format('%.0f,%.0f' % (llon4[0], llon4[1])))
ax.plot(date, msla5, label='{}'.format('%.0f,%.0f' % (llon5[0], llon5[1])))

ax.legend(loc=4, ncol=5)
ax.set_ylabel("SLA (m)")
plt.tight_layout()

for i in range(len(time)):
    if date[i].month==1:
        ax.axvline(date[i], c='grey', ls=':', lw=1)
# time labels
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
#
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
#ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%b"))
#plt.setp(ax.xaxis.get_minorticklabels(), rotation=90, ha="center")

# expand x axis to -/+ several days of the time list endpoints
start_time = datetime.datetime(date[0].year, 
                               date[0].month-1, 10)
end_time = datetime.datetime(date[-1].year,
        date[-1].month, 30)
ax.set_xlim(start_time, end_time)

fig.suptitle("ENV + CS2")
#fig.savefig(figdir+'trend_area_avg_filt_cs2.pdf', bbox_inches='tight')

# draw separating meridians among regions

fig, ax = plt.subplots(figsize=(8, 7))
cs = m.pcolormesh(glon, glat, trend_mm_yr,
                  vmin=-10, vmax=10,
                  cmap=cm.seismic,
                  latlon=True, 
                  rasterized=True)
m.drawcoastlines(linewidth=0.25)
m.fillcontinents(color='w')
cb = fig.colorbar(cs, ax=ax, shrink=0.6, extend='both')
cb.ax.set_title("mm/yr")

m.drawparallels(np.arange(-80., -40., 10), 
                zorder=4, linewdith=0.1, ax=ax)
m.drawmeridians(np.arange(0., 360., 30.), zorder=4,
               labels=[1, 1, 1, 1], linewidth=0.1, ax=ax)
m.drawmeridians([0, 80, -170, -95, -60], linewidth=2, ax=ax)

# annotate parallels
x1, y1 = m(180, -80)
ax.annotate(r"$80^\circ S$", xy=(x1, y1), xycoords='data',
        xytext=(x1, y1),textcoords='data')
x2, y2 = m(180, -70)
ax.annotate(r"$70^\circ S$", xy=(x2, y2), xycoords='data',
        xytext=(x2, y2),textcoords='data')
x3, y3 = m(180, -60)
ax.annotate(r"$60^\circ S$", xy=(x3, y3), xycoords='data',
        xytext=(x3, y3),textcoords='data')
x4, y4 = m(180, -50)
ax.annotate(r"$50^\circ S$", xy=(x4, y4), xycoords='data',
        xytext=(x4, y4),textcoords='data')
ax.set_rasterization_zorder(0)


fig.suptitle("Trend anomaly relative to the global-mean rate of SLR \n\
             (2002.07-2018.10)")
"""

dat = msla1

t0 = 2002.0
dt = 1/12
n = len(dat)
t = np.arange(0, n)*dt + 6/12 + t0

# detrend data and normalize by std
p = np.polyfit(t-t0, dat, 1)
dat_notrend = dat - np.polyval(p, t-t0)
std = dat_notrend.std(ddof=1)
var = std ** 2
dat_norm = dat_notrend/std #normalized dataset

mother = wavelet.Morlet(6)
s0 = 1 * dt #starting scale - 6 months
dj = 1/12 # 4 suboctaves per octave
j1 = 7/dj 
alpha, _, _ = wavelet.ar1(dat) # lag-1 autocorrelation for red noise


wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, j1,
                                                     mother)
iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

# normalized wavelet and Fourier power spectra
# Fourier equivalent periods for each wavelet scale
power = (np.abs(wave))**2
fft_power = np.abs(fft)**2
period = 1/freqs

# rectify the power spectrum
power /= scales[:, None]

#Significance test; signif where power/sig95 > 1
signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                         significance_level=0.95,
                                        wavelet=mother)
sig95 = np.ones([1, n]) * signif[:, None]
sig95 = power/sig95

# global wavelet spectrum and signi level
glbl_power = power.mean(axis=1)
dof = n-scales
glbl_signif, tmp = wavelet.significance(var, dt, scales, 1, alpha,
                                        significance_level=0.95, dof=dof,
                                        wavelet=mother)
# scale avg between 2 and 8 years and signif level
sel = find((period>=2) & (period < 8))
Cdelta = mother.cdelta
scale_avg = (scales*np.ones((n, 1))).transpose()
scale_avg = power / scale_avg
scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
scale_avg_signif, tmp = wavelet.significance(var, dt, scales, 2, alpha,
					significance_level=0.95,
					dof=[scales[sel[0]], scales[sel[-1]]],
					wavelet=mother)

plt.ion()

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(t, dat_notrend, c='k', lw=1.5, label='detrended SLA')
ax.plot(t, iwave, c='c', lw=2, label='inverse wavelet transform')
ax.set_title('area-weighted average SLA (m)')
ax.legend(loc=4, ncol=2)
plt.tight_layout()


fig, ax = plt.subplots(figsize=(8, 8))
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
ax.contourf(t, np.log2(period), np.log2(power), np.log2(levels), extend='both',
           cmap=cm.viridis)
extent = [t.min(), t.max(), 0, max(period)]
ax.contour(t, np.log2(period), sig95, [-99, 1], colors='k', linewidths=2,
           extent=extent)
#ax.fill(np.concatenate([t, t[-1:]+dt, t[-1:]+dt, t[:1]-dt, t[:-1]-dt]),
#        np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),
#                        np.log2(period[-1:]), [1e-9]]),
#        'k', alpha=0.3, hatch='x')
ax.set_title('Wavelet Power Spectrum ({})'.format(mother.name))
ax.set_ylabel('Period (log2(years))')
Yticks = 2**np.arange(np.round(np.log2(period.min())), np.round(np.log2(period.max())))
ax.set_yticks(np.log2(Yticks))
ax.set_yticklabels(Yticks)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(5, 4))
ax.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
ax.plot(t, scale_avg, 'k-', linewidth=1.5)
ax.set_title('{}--{} year scale-averaged power'.format(2, 8))
ax.set_xlabel('Time (year)')
ax.set_ylabel(r'Average variance [{}]'.format('m'))
ax.set_xlim([t.min(), t.max()])
plt.tight_layout()

fig, ax = plt.subplots(figsize=(6, 9))
ax.plot(glbl_signif, np.log2(period), 'k--', label='global wavelet significance')
ax.plot(var*fft_theor, np.log2(period), '--', c='c', label='Fourier theoretical noise spectra')
ax.plot(var*fft_power, np.log2(1./fftfreqs), '-', c='m', lw=1., label='Fourier power spectra')
ax.plot(var*glbl_power, np.log2(period), 'k-', lw=1.5, label='global power spectra')
ax.set_title('Global Wavelet Spectrum')
ax.set_xlabel(r'Power [{}$^2$]'.format('m'))
#ax.set_xlim([0, glbl_power.max() * var])
#ax.set_ylim(np.log2([period.min(), period.max()]))
ax.set_yticks(np.log2(Yticks))
ax.set_yticklabels(Yticks)
ax.legend()
plt.tight_layout()
"""

dat = msla1

t0 = 2002.0
dt = 1/12
n = len(dat)
t = np.arange(0, n)*dt + 6/12 + t0

# detrend data and normalize by std
p = np.polyfit(t-t0, dat, 1)
dat_notrend = dat - np.polyval(p, t-t0)
std = dat_notrend.std(ddof=1)
variance = std ** 2
dat_norm = dat_notrend/std #normalized dataset

xlim = ([2002, 2018])
pad = 1 # pad the time series with 0
dj = 1/12 # 4 suboctaves per octave
s0 = 6 * dt #starting scale - 6 months
j1 = 7/dj 
lag1 = 0.72 # lag 1 autocorr for red noise background

#alpha, _, _ = wavelet.ar1(dat) # lag-1 autocorrelation for red noise

print("lag1 = ", lag1)

mother = 'MORLET'

# Wavelet transform
wave, period, scale, coi = wavelet(dat, dt, pad, dj, s0, j1, mother)
power = (np.abs(wave))**2 # wavelet power spectrum
global_ws = (np.sum(power, axis=1) / n) # time-average over all times

#Significance levels
signif = wave_signif(([variance]), dt=dt, sigtest=0, scale=scale, lag1=lag1,
                     mother=mother)
sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :]) # expand signif
sig95 = power / sig95

# Global wavelet spectrum & signif levels
dof = n - scale
global_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=1, lag1=lag1,
                           dof=dof, mother=mother)

# Scale-average between El Nino periods of 2-8 years
avg = np.logical_and(scale >= 2, scale < 8)
Cdelta = 0.776  # for the Morlet wavelet
scale_avg = scale[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
scale_avg = power / scale_avg
scale_avg = dj * dt / Cdelta *sum(scale_avg[avg, :])
scaleavg_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=2,
                              lag1=lag1, dof=([2, 7.9]), mother=mother)

#------------------------------------------------------ Plotting
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

time = t

#--- Plot time series
fig = plt.figure(figsize=(9, 10))
gs = GridSpec(3, 4, hspace=0.4, wspace=0.75)
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95, wspace=0, hspace=0)
plt.subplot(gs[0, 0:3])
plt.plot(time, dat_notrend, 'k')
plt.xlim(xlim[:])
plt.xlabel('Time (year)')
plt.ylabel('detrended SLA (m)')
plt.title('a) Sea level anomaly (monthly area-weighted average)')

plt.text(time[-1] + 35, 0.5,'Wavelet Analysis\nC. Torrence & G.P. Compo\n' +
    'http://paos.colorado.edu/\nresearch/wavelets/',
    horizontalalignment='center', verticalalignment='center')

#--- Contour plot wavelet power spectrum
# plt3 = plt.subplot(3, 1, 2)
plt3 = plt.subplot(gs[1, 0:3])
levels = [0, 0.5, 1, 2, 4, 999]
CS = plt.contourf(time, period, power, len(levels))  #*** or use 'contour'
im = plt.contourf(CS, levels=levels, colors=['white','bisque','orange','orangered','darkred'])
plt.xlabel('Time (year)')
plt.ylabel('Period (years)')
plt.title('b) Wavelet Power Spectrum (contours at 0.5,1,2,4\u00B0m$^2$)')
plt.xlim(xlim[:])
# 95# significance contour, levels at -99 (fake) and 1 (95# signif)
plt.contour(time, period, sig95, [-99, 1], colors='k')
# cone-of-influence, anything "below" is dubious
plt.plot(time, coi, 'k')
# format y-scale
plt3.set_yscale('log', basey=2, subsy=None)
plt.ylim([np.min(period), np.max(period)])
ax = plt.gca().yaxis
ax.set_major_formatter(ticker.ScalarFormatter())
plt3.ticklabel_format(axis='y', style='plain')
plt3.invert_yaxis()
# set up the size and location of the colorbar
# position=fig.add_axes([0.5,0.36,0.2,0.01]) 
# plt.colorbar(im, cax=position, orientation='horizontal') #, fraction=0.05, pad=0.5)

# plt.subplots_adjust(right=0.7, top=0.9)

#--- Plot global wavelet spectrum
plt4 = plt.subplot(gs[1, -1])
plt.plot(global_ws, period)
plt.plot(global_signif, period, '--')
plt.xlabel('Power (\u00B0m$^2$)')
plt.title('c) Global Wavelet Spectrum')
plt.xlim([0, 1.25 * np.max(global_ws)])
# format y-scale
plt4.set_yscale('log', basey=2, subsy=None)
plt.ylim([np.min(period), np.max(period)])
ax = plt.gca().yaxis
ax.set_major_formatter(ticker.ScalarFormatter())
plt4.ticklabel_format(axis='y', style='plain')
plt4.invert_yaxis()

# --- Plot 2--8 yr scale-average time series
plt.subplot(gs[2, 0:3])
plt.plot(time, scale_avg, 'k')
plt.xlim(xlim[:])
plt.xlabel('Time (year)')
plt.ylabel('Avg variance (\u00B0C$^2$)')
plt.title('d) 2-8 yr Scale-average Time Series')
plt.plot(xlim, scaleavg_signif + [0, 0], '--')

plt.show()
"""
