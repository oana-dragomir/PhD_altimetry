'''
CRYOSAT-2 

This script applies some corrections to the raw data and converts the
text files (.elev) into netCDF format.

New .nc files only have the fields: 
    surface type (1=ocean, 2=lead) 
    time 
    lat, lon, 
    elevation, 
    mean SSH,
    percentage ice concentration (sic, removed outliers (-999))
    sea ice type (sit, keep all types)
    confidence in sea ice type (csit, keep only good (4) and excellent (5))
    atnum (along-track number),
    atdir (direction: 1=ascending [S-N], -1=descending)
    retracker (1=LRM, 2=SAR, 3=SARIN)

Post-processing steps:
    1. keep valid data (validity: 0=no, 1=yes)
    2. keep ocean and lead type only (surface: 0=unknown, 1=Ocean, 2=Lead, 3=Flow)
    3. discard |SSHA| > 3 m, SSHA = SSH - MSS
    4. label tracks with an integer starting from 1 based on the fact that
    consecutive tracks are separated by roughly 180 deg
    5. determine direction (i.e. ascending/descending)
    6. label in which retracker is every point (1=LRM, 2=SAR, 3=SARIN)

Last modified: 24 Mar 2023
'''
# Import modules
import numpy as np
from numpy import ma

from datetime import datetime
today = datetime.today()

import time as runtime
t_start = runtime.process_time()

from mpl_toolkits.basemap import Basemap

from netCDF4 import Dataset, num2date

# import a list with the names of the files to be analysed
from aux_filenames import cs2_id_list as filenames

import gsw

import shapefile
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import os
import sys 

# Define directories
# raw data are stored in a mounted dir
# >> cd ~/storage_soton/
# >> sudo ./mount_ODM ocd1n16

workdir = '/Volumes/SamT5/PhD_data/altimetry_cpom/'
datadir = workdir + '0_raw_elev/'
ncdir = workdir + '1_raw_nc/'
maskdir = workdir +  'CS2_mode_mask/'

#- - - - - - - - - - - - - - - - - - - - - - - - - 
time_units = 'days since 1950-01-01 00:00:00.0'

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

### ----------------------------------------------------------
print("Extracting polygons from CS2 mode masking file .. \n")
# Shapefile with CS2 mode masks
sf = shapefile.Reader(maskdir+"mask3_8")

# LRM vs. SAR
# shapes 0-23 are are boundaries between open ocean vs. sea-ice
# every two pairs are the same shape, so pick either to use
# assume 0/1 = Jan, ..22/23 = Dec

# indices of SAR/LRM shapes
SARidx = np.arange(0, 23, 2)
SARlon, SARlat = [], []
for k in SARidx:
    SARshape = sf.shapeRecord(k)
    lon = [i[0] for i in SARshape.shape.points[:]]
    lat = [i[1] for i in SARshape.shape.points[:]]
    SARlat.append(lat)
    SARlon.append(lon)

# SEA-ICE vs COAST/ICE Sheet (SAR-SARin) boundary has index 24
sarin_shape = sf.shapeRecord(24)
sarin_lon = [i[0] for i in sarin_shape.shape.points[:]]
sarin_lat = [i[1] for i in sarin_shape.shape.points[:]]

# construct SARin polygon
sarin_x, sarin_y = m(sarin_lon, sarin_lat)
sarin_xy = np.column_stack((sarin_x, sarin_y))
sarin_poly = Polygon(sarin_xy)

# West Antarctic Peninsula box
sarin_wap_shape = sf.shapeRecord(67)
sarin_wap_lon = [i[0] for i in sarin_wap_shape.shape.points[:]]
sarin_wap_lat = [i[1] for i in sarin_wap_shape.shape.points[:]]

sarin_wap_x, sarin_wap_y = m(sarin_wap_lon, sarin_wap_lat)
sarin_wap_xy = np.column_stack((sarin_wap_x, sarin_wap_y))
sarin_wap_poly = Polygon(sarin_wap_xy)

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
    # 3. remove |SLA| > 1
    sla_02 = ssh_02 - mss_02
    sla_02 = ma.masked_outside(sla_02, -3, 3) #does not mask +/-3

    surf_03 = surf_02[~sla_02.mask]
    time_03 = time_02[~sla_02.mask]
    lat_03 = lat_02[~sla_02.mask]
    lon_03 = lon_02[~sla_02.mask]
    ssh_03 = ssh_02[~sla_02.mask]
    mss_03 = mss_02[~sla_02.mask]
    sic_03 = sic_02[~sla_02.mask]
    sit_03 = sit_02[~sla_02.mask]
    csit_03 = csit_02[~sla_02.mask]

    print("4 & 5. Label along tracks and indicate the direction")
    # 4. label each AT with a number    
    # compute sequential geodetic distance between points
    dd = gsw.distance(lon_03, lat_03)
    # distances larger than say 1e3 m indicate a chage to a new track
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

    # add dir/track to the last point
    atnum_03[-1] = atnum_03[-2]
    atdir_03[-1] = atdir_03[-2]

    # determine the retracker type for every point 
    #------------------------------------------------------------------
    # SPLIT FURTHER INTO CS2 GEOGRAPHICAL MODES 
    #------------------------------------------------------------------
    date = num2date(time_03[0], units=time_units, calendar='gregorian')
    month = date.month
    # construct SAR polygon; it depends on the month of the data
    sar_lon, sar_lat = SARlon[month-1], SARlat[month-1]
    sar_x, sar_y = m(sar_lon, sar_lat)
    sar_xy = np.column_stack((sar_x, sar_y))
    sar_poly = Polygon(sar_xy)

    # data lat/lon in map projection coord
    xl, yl = m(lon_03, lat_03)

    # indices
    retracker = np.ones(len(xl))

    ### ----------------------------------------------------------
    print("checking if points are inside different retracker modes ... \n")
    for i in range(len(xl)):
        point = Point(xl[i], yl[i])
        # SAR polygon does not includes SARin polygon at all times
        if (sarin_poly.contains(point) == True
            or sarin_wap_poly.contains(point) == True):
            retracker[i] = 3
        elif sar_poly.contains(point) == True:
            retracker[i] = 2

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
    RETRACKER = dataset.createVariable('Retracker', np.float64, ('nrows'))

    dataset.description = ('Monthly <YearMonth> altimetry record containing: \
                            *Surface Type: 1=Ocean, 2=Lead\
                            *Time \
                            *Latitude \
                            *Longitude \
                            *Elevation. MSS has not been removed \
                            *Mean Sea Surface Height [SLA = Elevation - Mean SSH]\
                            *track_num - every track has been assigned a number\
                            *track_dir: (1:ascending, -1:descending)\
                            *percentage ice concentration\
                            *sea ice type (1:open water, 2:FirstYearIce, 3:MultiYearIce, 4:Ambiguous)\
                            *confidence in sea ice type - 4:good, 5:excellent\
                            *retracker index (1=LRM, 2=SAR, 3=SARIN)\
                            *along-track correction: keep only |SLA|<=3m')
    
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
    RETRACKER[:] = retracker
    dataset.close()

    print("File saved in " + ncdir)

    t_stop = runtime.process_time()
    print("execution time: %.1f min " %((t_stop-t_start)/60))

print("Script done!")
