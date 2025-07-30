"""
list with filenames of .nc files
(only contains data from 07.2002-03.2012 for ENV
and from 11.2010-10.2018 for CS2 - discarded some of the early months 
from CS2 due to low spatial converage)

Last modified: 28 Jan 2025
"""

# ENVISAT
# -----------------------------------------------------------------------------
yr_list = ['03', '04', '05', '06', '07', '08', '09', '10', '11']
month_list = ['01', '02', '03', '04', '05', '06',
              '07', '08', '09', '10', '11', '12']
id_list_mid = ['month'+yr_list[i] + month_list[j] for i in range(len(yr_list))
              for j in range(len(month_list))]
id_0 = ['0207', '0208', '0209', '0210', '0211', '0212']
id_1 = ['1201', '1202', '1203']
id_list_start = ['month' + id_0[i] for i in range(len(id_0))]
id_list_end = ['month' + id_1[i] for i in range(len(id_1))]
env_id_list = id_list_start + id_list_mid + id_list_end

# CS2
# -----------------------------------------------------------------------------
yr_list = ['2011', '2012', '2013', '2014', '2015', '2016', '2017']
id_list_mid = [yr_list[i] + month_list[j] for i in range(len(yr_list)) for j in
              range(len(month_list))]
id_list_start = ['201011', '201012']
id_list_end = ['201801', '201802', '201803', '201804', '201805', 
               '201806', '201807', '201808', '201809', '201810']
cs2_id_list = id_list_start + id_list_mid + id_list_end
