#! /usr/bin/env python3
"""
Monthly means from daily sampled AVISO SLA

Time coverage:
- 1993-01 to 2018-05 (reprocessed, L4)
- 2018-06 to 2018-10 (near real time)

- load data (SLA relative to a 20-yr mean: 1993-2012)
- compute monthly means of the SLA
- save in a new file

Last modified: 27 Mar 2019
"""
# Import modules
import numpy as np
import xarray as xr

import sys

# Define directories
datadir = '/home/ocdragomir/Documents/PhD/data/aviso/'

# -----------------------------------------------------------------------------
# AVISO files
filenames = np.asarray(['93_96', '97_00', '01_04', '05_08', 
                        '09_12', '13_16', '17_18',
                        '2018jun_2018oct_nrt'])
nfiles = len(filenames)

# -----------------------------------------------------------------------------
print("Computing monthly means .. \n")
# read the first file and compute the monthly means
slafile = datadir+'sla_'+filenames[4]+'.nc'
daily_aviso = xr.open_dataset(slafile)

# time uses the first day of the month
monthly_aviso = daily_aviso.resample(time='1MS').mean()

monthly_aviso.to_netcdf(datadir+'monthly_sla_aviso_09_12.nc', mode='w')

sys.exit()
# merge datasets from all files above
for k in range(1, nfiles-1):
    slafile = datadir+'sla_'+filenames[k]+'.nc'
    ds = xr.open_dataset(slafile)
    
    # monthly averages
    monthly_ds = ds.resample(time='1MS').mean()
    monthly_aviso = monthly_aviso.combine_first(monthly_ds)

# -----------------------------------------------------------------------------
print("Saving dataset in a new file ... \n")
# save dataset in a new file
monthly_aviso.to_netcdf(datadir+'monthly_sla_aviso.nc', mode='w',
                        format='NETCDF4')

print("Script done!")
