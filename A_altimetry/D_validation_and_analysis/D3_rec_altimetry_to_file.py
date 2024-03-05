"""
compute reconstructed SLA from first 4 eofs (for mooring rmaps) and save it to a file
"""

import numpy as np
from numpy import ma

import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from palettable.cmocean.diverging import Balance_19

import sys

plt.ion()
#-------------------------------------------------------------------
# Directories
#-------------------------------------------------------------------
voldir = '/Volumes/SamT5/PhD/data/'
griddir = voldir + 'altimetry_cpom/3_grid_dot/'

localdir = '/Users/ocd1n16/PhD_git/'
figdir = '/Users/ocd1n16/PhD_local/overleaf_figures/ch2/'

auxscriptdir = localdir + 'aux_func/'
sys.path.append(auxscriptdir)
import aux_func_trend as fc
import aux_stereoplot as st
import aux_corr_maps as rmap

# function that extracts altimetry data
sys.path.append(localdir + 'A_altimetry/D_validation_and_analysis/')
import d2_extract_altim_anom as altim
#-------------------------------------------------------------------
# time window
date_start = '2010-02-01'
date_end = '2016-01-01'


# ALTIMETRY data
geoidtype = 'goco05c'#'eigen6s4v2' # 'goco05c', 'egm08'
satellite = 'all'
sigma = 3
# -----------------------------------------------------------------
#            ~ ~ ~     ALTIMETRY DATA      ~ ~ ~
# -----------------------------------------------------------------
alt = altim.extract_dot_maps(geoidtype, satellite, sigma, date_start, date_end)
glat, glon = np.meshgrid(alt.latitude.values, alt.longitude.values)
eglat, eglon = np.meshgrid(alt.elat.values, alt.elon.values)

#-----------------------------------
# EOF (eofs)
#-----------------------------------
rec = fc.extract_eof(3, alt.dot_det, alt.latitude.values, alt.longitude.values)
rec_anom = rec - rec.mean('time')

rec_anom.coords["elon"] = alt.elon.values
rec_anom.coords["elat"] = alt.elat.values
#rec_anom["description"] = "geoid: goco05c, SLA reconstructed from first 4 EOFs, sigma=3 (gaussian filter)"
rec_anom.to_netcdf(griddir + "sla3eof_2010feb_2016jan.nc")





