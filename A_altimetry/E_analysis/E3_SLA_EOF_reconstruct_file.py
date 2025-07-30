"""
Compute reconstructed SLA from selected EOF modes and save it to a file.

Last modified: 30 May 2025
"""

import numpy as np
from numpy import ma
import pandas as pd
import xarray as xr

from datetime import datetime

import analysis_functions.aux_altimetry as altim

#-------------------------------------------------------------------
# Directories
#-------------------------------------------------------------------
voldir = '/Volumes/SamT5/PhD/PhD_data/'
griddir = voldir + 'altimetry_cpom/3_grid_dot/'

#-------------------------------------------------------------------
# time window
date_start = '2009-02-01'
date_end = '2016-01-31'

n_eof = 6
# -----------------------------------------------------------------
#            ~ ~ ~     ALTIMETRY DATA      ~ ~ ~
# -----------------------------------------------------------------
# ALTIMETRY filename - use goco05c sigma=3 as the main file
geoidtype = 'goco05c'#'eigen6s4v2' # 'goco05c', 'egm08'
satellite = 'all'
sigma = 3

alt = altim.detrend_dot_geosvel(geoidtype, satellite, sigma, date_start, date_end, plot=False)

#-----------------------------------
# EOF (eofs)
#-----------------------------------
rec = altim.extract_eof(n_eof, alt.sla_det, alt.latitude.values, alt.longitude.values)

# prepare file to be saved
rec.coords["elon"] = alt.elon.values
rec.coords["elat"] = alt.elat.values

rec.attrs["description"] = "SLA reconstructed from first %s EOF modes." % str(n_eof)
rec.attrs['creation_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Convert to datetime objects
start_time_dt = pd.Timestamp(rec.time.min().values)
end_time_dt = pd.Timestamp(rec.time.max().values)

# Format as "MonthYear"
start_month_year = start_time_dt.strftime("%b%Y")
end_month_year = end_time_dt.strftime("%b%Y")

# Create the final string
time_range = f"{start_month_year}_{end_month_year}_"

new_filename = 'sla_' + str(n_eof) + "eof_" + time_range + geoidtype + ".nc"
rec.to_netcdf(griddir + new_filename)

print("File %s saved in %s" % (new_filename, griddir))