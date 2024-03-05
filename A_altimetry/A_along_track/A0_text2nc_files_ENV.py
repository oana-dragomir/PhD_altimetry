'''
ENVISAT

This script applies some corrections to the raw data and converts the
text files (.elev) into netCDF format.

New .nc files only have the fields: 
    surface type (1=ocean, 2=lead) 
    time 
    lat, lon, 
    elevation, 
    mean SSH,
    percentage ice concentration (sic, outliers removed (-999))
    sea ice type (sit, keep all types)
    confidence in sea ice type (csit, keep only good (4) and excellent (5))
    atnum (along-track number),
    atdir (direction: 1=ascending [S-N], -1=descending)

Post-processing steps:
    1. keep valid data (validity: 0=no, 1=yes)
    2. keep ocean and lead type only (surface: 0=unknown, 1=Ocean, 2=Lead, 3=Flow)
    3. discard |SSHA| > 3 m, SSHA = SSH - MSS
    4. assign a number to tracks starting from 1 based on the fact that
    consecutive tracks are separated by roughly 180 deg
    5. determine direction of track and label it (i.e. ascending=1/descending=-1)
    
    [not applied anymore, too strict]6. discard SLA < median-3*MAD and SLA > median + 3*MAD

Last modified: 11 Mar 2021
'''
# Import modules
import numpy as np
from numpy import ma

from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset, num2date
import gsw

import os
import sys 

from datetime import datetime
import time as runtime

# list with the names of files
from filenames import env_id_list as filenames
#------------------------------------------------------------------
# estimate script running time
t_start = runtime.process_time()

# Define directories
# raw data are stored in a mounted dir
# >> cd ~/storage_soton/
# >> sudo ./mount_ODM ocd1n16

workdir = '/Volumes/SamT5/PhD/data/altimetry_cpom/'
datadir = workdir + '0_raw_elev/'
ncdir = workdir + '1_raw_nc/'

time_units = 'days since 1950-01-01 00:00:00.0'

today = datetime.today()

numfiles = len(filenames)

# Check all files have been created
for i in range(numfiles):
    filename = datadir + filenames[i] + '.elev'
    if (not os.path.isfile(filename)) or (not os.path.exists(filename)):
        print(filename, " - not found \n")
print(numfiles, " files found. \n")

print ("Preparing map area ... \n")
m = Basemap(projection='spstere', 
            boundinglat=-49.5, 
            lon_0=-180, 
            resolution='f', 
            round=True, 
            ellps='WGS84')

#------------------------------------------------------------------
for k in range(numfiles):
    print("parsing file "+filenames[k])
    filename = datadir + filenames[k] + '.elev'

    cols=(0, 1, 4, 5 ,6, 7, 8, 11, 12, 13)
    data = ma.masked_array(np.loadtxt(filename, usecols=cols))
    
    surf = data[:, 0]
    valid = data[:, 1]
    time = data[:, 2]
    lat = data[:, 3]
    lon = data[:, 4]
    ssh = data[:, 5]
    mss = data[:, 6]
    sic = data[:, 7]
    sit = data[:, 8]
    csit = data[:, 9]

    # 1. flagged data
    surf[valid!=1] = ma.masked
    surf[csit<4] = ma.masked
    surf[sit<1] = ma.masked
    surf[sic<0] = ma.masked

    # 2. keep only Ocean and Lead data
    surf = ma.masked_outside(surf, 1, 2)

    # apply mask to the rest of arrays    
    surf_02 = surf[~surf.mask]
    time_02 = time[~surf.mask]
    lat_02 = lat[~surf.mask]
    lon_02 = lon[~surf.mask]
    ssh_02 = ssh[~surf.mask]
    mss_02 = mss[~surf.mask]
    sic_02 = sic[~surf.mask]
    sit_02 = sit[~surf.mask]
    csit_02 = csit[~surf.mask]

    print("3. remove |SLA|>3 m")
    # 3. remove |SLA| > 3
    sla_02 = ssh_02 - mss_02
    sla_02 = ma.masked_outside(sla_02, -3, 3) #does not mask the limits 

    surf_03 = surf_02[~sla_02.mask]
    time_03 = time_02[~sla_02.mask]
    lat_03 = lat_02[~sla_02.mask]
    lon_03 = lon_02[~sla_02.mask]
    ssh_03 = ssh_02[~sla_02.mask]
    mss_03 = mss_02[~sla_02.mask]
    sic_03 = sic_02[~sla_02.mask]
    sit_03 = sit_02[~sla_02.mask]
    csit_03 = csit_02[~sla_02.mask]

    sla_03 = ssh_03 - mss_03

    print("4 & 5. Label along tracks and indicate the direction")
    # 4. label each AT with a number    
    # compute sequential geodetic distance between points
    dd = gsw.distance(lon_03, lat_03)
    # distances larger than say 1e3 m indicate a change to a new track
    ddm = ma.masked_less(dd, 1e6)
    idx = np.arange(len(dd))
    idxm = idx[~ddm.mask]+1 # index marks the beginning of the track 

    # add the start and end indices 
    indices = np.hstack((0, idxm, len(lon_03)-1))

    # 5. further check if the AT is ascending (S-N)/descending (N-S)
    # if the difference in latitude > 0 then the track is ascending
    atnum_03, atdir_03 = [ma.ones(len(lon_03)) for _ in range(2)]

    for i in range(len(indices)-1):
        start = indices[i]
        end = indices[i+1]

        atnum_03[start:end] = i
        latdif = lat_03[end-1] - lat_03[start]
        if latdif < 0:
            atdir_03[start:end] = -1
        # changed from std to mad 10 July 2019
        #track_median = ma.median(sla_03[start:end])
        #atmedian[start:end] = track_median
        #atmad[start:end] = ma.median(abs(sla_03[start:end]-track_median))

    # add dir/track to the last point
    atnum_03[-1] = atnum_03[-2]
    atdir_03[-1] = atdir_03[-2]

    # Save as an .nc file
    # ----------------------------------------------------------
    print("Preparing the Dataset for the .nc file.")

    # creating a Dataset
    newfile = ncdir + filenames[k] + '.nc'
    dataset = Dataset(newfile, 'w')

    # dimensions
    l = len(time_03)
    dataset.createDimension('nrows', l)

    # variables
    surfacetype = dataset.createVariable('SurfaceType', np.int32, ('nrows'))
    tim = dataset.createVariable('Time', np.float64, ('nrows'))
    lati = dataset.createVariable('Latitude', np.float64, ('nrows'))
    longi = dataset.createVariable('Longitude', np.float64, ('nrows'))
    elev = dataset.createVariable('Elevation', np.float64, ('nrows'))
    mSSH = dataset.createVariable('MeanSSH', np.float64, ('nrows'))
    ATNUM = dataset.createVariable('track_num', np.int32, ('nrows'))
    ATDIR = dataset.createVariable('track_dir', np.int32, ('nrows'))
    SIC = dataset.createVariable('ice_conc', np.float64, ('nrows'))
    SIT = dataset.createVariable('sea_ice_type', np.int32, ('nrows'))
    CSIT = dataset.createVariable('conf_sit', np.float64, ('nrows'))

    dataset.description = ('ENVISAT Monthly <YearMonth> altimetry record containing: \
                            *Surface Type: 1=Ocean, 2=Lead\
                            *Time \
                            *Latitude \
                            *Longitude \
                            *Elevation. MSS has not been removed \
                           *Mean Sea Surface Height [SLA = Elevation - Mean SSH];\
                           computed from 1 year of CS2 (Anna Hogg)\
                            *track_num - every track has been assigned a number\
                            *track_dir: (1:ascending, -1:descending)\
                            *percentage ice concentration\
                            *sea ice type (1:open water, 2:FirstYearIce, 3:MultiYearIce, 4:Ambiguous)\
                            *confidence in sea ice type - 4:good, 5:excellent\
                            *correction: keep only |SLA|<=3m')
    
    dataset.history = "Created " + today.strftime("%d/%m/%Y, %H:%M%S" )
    
    tim.units = time_units
    elev.units='meters'
    mSSH.units='meters'
    lati.units='degrees_north'
    longi.units='degrees_east'
    SIC.long_name='percentage_ice_concentration'
    SIT.long_name='sea_ice_type'
    CSIT.long_name='confidence_in_sea_ice_type'

    surfacetype[:] = surf_03.astype(int)
    tim[:] = time_03
    lati[:] = lat_03
    longi[:] = lon_03
    elev[:] = ssh_03
    mSSH[:] = mss_03
    ATNUM[:] = atnum_03.astype(int)
    ATDIR[:] = atdir_03.astype(int)
    SIC[:] = sic_03
    SIT[:] = sit_03
    CSIT[:] = csit_03

    dataset.close()

    print("File saved in " + ncdir)

    t_stop = runtime.process_time()
    print("execution time: %.1f min " %((t_stop-t_start)/60))

print("Script done!")
