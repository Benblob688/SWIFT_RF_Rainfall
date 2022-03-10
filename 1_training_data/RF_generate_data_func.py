'''
Code to read SEVIRI data in HRIT format.
'''

import satpy # Requires virtual environmen for reading native (.nat)  and hrit files.
import numpy as np
import datetime
import glob
from pyresample import geometry, bilinear
import os
import h5py
import sys
from tempfile import gettempdir

# The following are necessary to read compressed HRIT files
os.environ['XRIT_DECOMPRESS_PATH']='/home/users/bpickering/SOFTWARE/PublicDecompWT/2.06/xRITDecompress/xRITDecompress' # Necessary to read compressed files.
my_tmpdir = gettempdir() + '/'#/gws/nopw/j04/swift/bpickering/SEVIRI_visualisation/temp' # should match the unix enviornment variable TMPDIR, setting this to homespace facillitates deleting decompressed files after they have been read, which can otherwise fill up tmp directory.
print ('Temporary Directory for Decompress: '+my_tmpdir)

sev_data_dir1='/gws/nopw/j04/swift/earajr/HRIT_archive/'
sev_data_dir2='/gws/nopw/j04/swift/SEVIRI/' # Second directory to check. Necessary as first directory is incomplete

unavailable_times = (
                     [datetime.datetime(2014,3,2,12,00,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2014,3,3,12,00,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2014,3,4,12,00,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2014,3,26,2,45,0)]
                    +[datetime.datetime(2014,5,21,1,45,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,7,15,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,8,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,10,30,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,10,45,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,11,15,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,21,12,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(6)] # Problems reading
                    +[datetime.datetime(2014,5,21,14,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,21,15,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,18,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,19,15,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,20,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,20,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,21,22,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,23,30,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,4,15,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,5,45,0)+datetime.timedelta(seconds=60*15*n) for n in range(8)] # Problems reading
                    +[datetime.datetime(2014,5,22,8,15,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,22,9,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,9,45,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,10,15,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,22,11,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,11,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,22,12,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,12,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,22,13,15,0)+datetime.timedelta(seconds=60*15*n) for n in range(4)] # Problems reading
                    +[datetime.datetime(2014,5,22,14,30,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,16,30,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,17,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,18,30,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,19,30,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,21,45,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,23,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,23,0,15,0)] # Problems reading
                    +[datetime.datetime(2014,5,23,0,45,0)] # Problems reading
                    +[datetime.datetime(2014,5,23,4,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,23,4,45,0)] # Problems reading
                    +[datetime.datetime(2014,5,23,10,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(9)] # Problems reading
                    +[datetime.datetime(2014,5,23,13,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,23,14,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,23,15,45,0)] # Problems reading
                    +[datetime.datetime(2014,5,23,16,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,23,19,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,23,20,15,0)] # Problems reading
                    +[datetime.datetime(2014,5,23,20,45,0)] # Problems reading
                    +[datetime.datetime(2014,5,23,22,15,0)] # Problems reading
                    +[datetime.datetime(2014,6,20,19,45,0)] # Problems reading
                    +[datetime.datetime(2014,8,15,14,45,0)] # Problems reading
                    +[datetime.datetime(2014,10,9,11,45,0)] # Problems reading
                    +[datetime.datetime(2014,10,10,11,45,0)] # Problems reading
                    +[datetime.datetime(2014,10,11,11,45,0)] # Problems reading
                    +[datetime.datetime(2014,10,12,11,45,0)] # Problems reading
                    +[datetime.datetime(2014,12,4,14,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2014,12,8,6,45,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2015,3,1,12,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2015,3,2,12,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2015,3,3,9,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(4)]
                    +[datetime.datetime(2015,3,3,12,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(3)]
                    +[datetime.datetime(2015,3,4,12,15,0)]
                    +[datetime.datetime(2015,3,16,11,30,0)]
                    +[datetime.datetime(2015,4,28,7,45,0)+datetime.timedelta(seconds=60*15*n) for n in range(4)]
                    +[datetime.datetime(2015,4,28,12,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(3)]
                    +[datetime.datetime(2015,5,14,9,45,0)]
                    +[datetime.datetime(2015,5,14,10,45,0)]
                    +[datetime.datetime(2015,5,14,11,15,0)]
                    +[datetime.datetime(2015,5,29,2,15,0)]
                    +[datetime.datetime(2015,6,1,5,45,0)]
#                    +[datetime.datetime(2015,6,1,13,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,2,2,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,2,13,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,3,2,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,3,5,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,3,7,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,4,3,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,4,19,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,4,20,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,5,12,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,5,16,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,8,2,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,9,6,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,9,10,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,9,21,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,10,18,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,10,19,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,11,4,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,11,5,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,12,15,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,12,18,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,12,22,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,13,19,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,14,1,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,14,6,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,14,7,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,14,20,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,14,21,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,14,23,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,15,2,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,15,4,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,15,7,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,15,7,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,15,14,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,15,19,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,15,20,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,15,22,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,15,22,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,16,3,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,16,7,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,16,12,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,16,15,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,16,16,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,17,12,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,17,12,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,17,14,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,18,10,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,18,15,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,18,15,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,18,16,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,18,17,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,18,19,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,18,21,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,19,7,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,19,14,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,19,16,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,20,4,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,20,13,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,21,4,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,21,7,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,21,9,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,21,15,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,21,18,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,22,1,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,22,9,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,22,16,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,22,16,30,0)] # Problems reading
                    +[datetime.datetime(2015,7,1,0,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(4)]
                    +[datetime.datetime(2015,10,10,11,45,0)]
                    +[datetime.datetime(2015,10,11,11,45,0)]
                    +[datetime.datetime(2015,10,12,11,45,0)]
                    +[datetime.datetime(2015,10,21,10,0,0)]
                    +[datetime.datetime(2015,11,15,3,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(19)] 
                    +[datetime.datetime(2015,11,16,8,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2015,11,25,12,15,0)]
                    +[datetime.datetime(2016,2,29,12,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2016,1,1,8,45,0)] #Problems reading
                    +[datetime.datetime(2016,1,1,23,15,0)] #Problems reading
                    +[datetime.datetime(2016,1,2,2,15,0)] #Problems reading
                    +[datetime.datetime(2016,1,3,3,15,0)] #Problems reading
                    +[datetime.datetime(2016,1,4,3,45,0)] #Problems reading
                    +[datetime.datetime(2016,1,4,7,30,0)] #Problems reading
                    +[datetime.datetime(2016,1,4,8,45,0)] #Problems reading
                    +[datetime.datetime(2016,1,4,16,30,0)] #Problems reading
                    +[datetime.datetime(2016,1,5,11,15,0)] #Problems reading
                    +[datetime.datetime(2016,1,5,12,15,0)] #Problems reading
                    +[datetime.datetime(2016,1,5,20,15,0)] #Problems reading
                    +[datetime.datetime(2016,1,6,9,45,0)] #Problems reading
                    +[datetime.datetime(2016,1,6,15,30,0)] #Problems reading
                    +[datetime.datetime(2016,1,6,20,45,0)] #Problems reading
                    +[datetime.datetime(2016,3,8,10,30,0)]
                    +[datetime.datetime(2016,6,8,14,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2016,7,20,15,15,0)] # Problems reading
                    +[datetime.datetime(2016,7,29,2,45,0)]
                    +[datetime.datetime(2016,8,17,9,45,0)]
                    +[datetime.datetime(2016,10,8,9,30,0)] # Unavailable from Eumetsat
                    +[datetime.datetime(2016,10,10,11,45,0)] # Problems reading
                    +[datetime.datetime(2016,10,11,11,0,0)] # Problems reading
                    +[datetime.datetime(2016,10,11,11,45,0)] # Problems reading
                    +[datetime.datetime(2016,10,12,11,45,0)] # Problems reading
                    +[datetime.datetime(2016,10,12,16,45,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Supposedly available from EumetSat website, but no files in folder on more than one attempt to download
                    +[datetime.datetime(2016,10,12,22,15,0)] # Supposedly available from EumetSat website, but no files in folder on more than one attempt to download
                    +[datetime.datetime(2016,10,14,9,30,0)] # Problems reading
                    +[datetime.datetime(2016,10,15,12,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(8)]
                    +[datetime.datetime(2016,10,16,13,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(66)] # Weird horizontal lines on these files...
                    +[datetime.datetime(2017,2,26,12,15,0)] # Unavailable for download
                    +[datetime.datetime(2017,2,27,12,15,0)] # Unavailable for download
                    +[datetime.datetime(2017,2,28,12,15,0)]
                    +[datetime.datetime(2017,3,17,21,45,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2017,4,22,22,15,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2017,6,22,11,45,0)]
                    +[datetime.datetime(2017,9,22,11,30,0) + datetime.timedelta(seconds=60*15*n) for n in range(3)]
                    +[datetime.datetime(2017,10,10,11,45,0)]
                    +[datetime.datetime(2017,10,11,11,45,0)]
                    +[datetime.datetime(2017,10,12,11,45,0)]
                    +[datetime.datetime(2017,10,13,11,45,0)]
                    +[datetime.datetime(2017,11,7,7,0,0) + datetime.timedelta(seconds=60*15*n) for n in range(5)]
                    +[datetime.datetime(2018,3,6,12,15,0,0)]
                    +[datetime.datetime(2018,3,21,12,15,0,0)]
                    +[datetime.datetime(2018,5,6,20,0,0,0)]
                    +[datetime.datetime(2018,5,7,4,45,0,0)]
                    +[datetime.datetime(2018,5,7,12,0,0,0)]
                    +[datetime.datetime(2018,6,20,9,15,0) + datetime.timedelta(seconds=60*15*n) for n in range(3)]
                    +[datetime.datetime(2018,7,3,13,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2018,7,4,4,0,0)]
                    +[datetime.datetime(2018,7,4,4,45,0)]
                    +[datetime.datetime(2018,7,10,23,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(16)]
                    +[datetime.datetime(2018, 9, 24, 12, 30)]
                    +[datetime.datetime(2018, 9, 27, 7, 30)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2018, 10, 10, 11, 45)]
                    +[datetime.datetime(2018, 10, 12, 11, 45)]
                    +[datetime.datetime(2018, 10, 13, 11, 45)]
                    +[datetime.datetime(2018, 10, 15, 11, 45)]
                    +[datetime.datetime(2018, 11, 19, 9, 45)]
                    +[datetime.datetime(2018, 12, 2, 15, 15)]
                    +[datetime.datetime(2018, 12, 6, 15, 0)]
                    +[datetime.datetime(2019, 1, 16, 10, 30)]
                    +[datetime.datetime(2019, 1, 22, 15, 30)+datetime.timedelta(seconds=60*15*n) for n in range(3)]
                    +[datetime.datetime(2019, 1, 29, 12, 0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2019, 2, 14, 9, 0)]
                    +[datetime.datetime(2019, 3, 5, 12, 15)] # NaNs in file
                    +[datetime.datetime(2019, 3, 6, 12, 15)] # NaNs in file
                    +[datetime.datetime(2019, 3, 7, 12, 15)] # NaNs in file
                    +[datetime.datetime(2019, 3, 12, 13, 0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # NaNs in file
                    +[datetime.datetime(2019, 4, 12, 10, 15)] # Unavailable from EumetSat
                    +[datetime.datetime(2019, 5, 6, 15, 45)] # Unavailable from EumetSat
                    +[datetime.datetime(2019, 5, 14, 10, 15)] # Unavailable from EumetSat
                    +[datetime.datetime(2019, 5, 24, 14, 0)] # Unavailable from EumetSat
                    +[datetime.datetime(2019, 5, 28, 13, 15)] # Unavailable from EumetSat
                    +[datetime.datetime(2019, 5, 29, 2, 15)] # Unavailable from EumetSat
                    +[datetime.datetime(2019, 5, 29, 15, 15)] # Unavailable from EumetSat
                    +[datetime.datetime(2019, 8, 17, 7, 15)]
                    +[datetime.datetime(2019, 10, 8, 10, 30)]
                    +[datetime.datetime(2019, 10, 8, 10, 45)]
                    +[datetime.datetime(2019, 10, 16, 12, 15)]
                    +[datetime.datetime(2019, 11, 11, 14, 0)]
                    +[datetime.datetime(2019, 11, 11, 14, 15)]
                    +[datetime.datetime(2019, 11, 11, 14, 30)]
                    +[datetime.datetime(2019, 12, 1, 0, 0)]
                    +[datetime.datetime(2019, 12, 2, 15, 0)]
                    +[datetime.datetime(2019, 12, 17, 9, 45)]
                    +[datetime.datetime(2020, 5, 3, 8, 15)]
                    +[datetime.datetime(2020, 5, 3, 8, 45)]
                    +[datetime.datetime(2020, 5, 3, 9, 45)]
                    +[datetime.datetime(2020, 5, 3, 10, 45)]
                    +[datetime.datetime(2020, 5, 3, 11, 45)]
                    +[datetime.datetime(2020, 5, 3, 12, 45)]
                    +[datetime.datetime(2020, 5, 3, 13, 45)]
                    +[datetime.datetime(2020, 5, 3, 14, 45)]
                    +[datetime.datetime(2020, 5, 3, 15, 45)]
                    +[datetime.datetime(2020, 5, 3, 16, 45)]
                    +[datetime.datetime(2020, 5, 3, 17, 45)]
                    )# List of missing times - these are not available from the EumetSat website, or have multiple lines of missing data.

file_dict = {
             0.6 : 'VIS006',
             0.8 : 'VIS008',
             1.6 : 'IR_016',
             3.9 : 'IR_039',
             6.2 : 'WV_062',
             7.3 : 'WV_073',
             8.7 : 'IR_087',
             9.7 : 'IR_097',
             10.8 : 'IR_108',
             12.0 : 'IR_120',
             13.4 : 'IR_134'
            }

def read_seviri_channel(channel_list, time, subdomain=(), regrid=False, my_area=geometry.AreaDefinition('pan_africa', 'Pan-Africa on Equirectangular 0.1 degree grid used by GPM', 'pan_africa', {'proj' : 'eqc'}, 720, 730, (-2226389.816, -3896182.178, 5788613.521, 4230140.650)), interp_coeffs=()):
    '''Read SEVIRI data for given channels and time (start of scan)

    Includes functionality to subsample or regrid. Requires satpy.
    Assumes SEVIRI files are located in sev_data_dir1 set above, with 
    directory structure sev_data_dir1/Year/YearMonthDay/Hour/

    Args:
        channel_list (list): list of channels to read, see file_dict for 
                             possible values
        time (datetime): SEVIRI file date and time, every 00, 15, 30 or 
                         45 minutes exactly, denoting the start of the scan.
        subdomain (tuple, optional): If not empty and regrid is False, then 
                                     tuple values are (West boundary, 
                                     South boundary, East boundary, 
                                     North boundary) Defaults to empty tuple.
        regrid (bool, optional): If True, then data is regriddedonto grid 
                                 defined by my_area. Defaults to False.
        my_area (AreaDefinition, optional): pyresmaple.geometry.AreaDefinition
                                            Only used if regrid=True
                                            Defaults to a Hatano equal area 
                                            projection ~4.5 km resolution
                                            extending from ~33W to ~63E and
                                            ~29S to ~29N.
        interp_coeffs (tuple, optional): Interpolation coefficients that may be
                                         used for bilinear interpolation onto 
                                         new grid. Faccilitates use of same 
                                         coeffcients when regridding operation 
                                         is repeated in multiple calls to 
                                         read_seviri_channel. 
                                         Defaults to empty tuple.

    Returns:
        data (dict): Dictionary containing following entries:
                     lons (ndarray, shape(nlat,nlon)): Array of longitude values
                     lats (ndarray, shape(nlat,nlon)): Array of latitude values
                     interp_coeffs (tuple): If regrid is True, then the 
                                            interpolation coefficients are 
                                            returned in this variable to 
                                            speed up future regridding
                     channel (ndarray, shape(nlat,nlon)): Dictionary contains 
                                                          separate entry for 
                                                          each channel in 
                                                          channel_list

    '''
    ### 0 ### Initialise
    filenames = []
    sat_names = ['MSG4', 'MSG3', 'MSG2', 'MSG1']
    sat_ind = -1
    
    
    ### 1 ### Check if files avaialable
    if time in unavailable_times:
        raise UnavailableFileError("SEVIRI observations for "+time.strftime("%Y/%m/%d_%H%M")+" are not available")
    print('time=', time)
    
    
    ### 2 ### Sometimes have data from multiple instruments (e.g. 20160504_1045 has MSG3 and MSG1), this ensures most recent is prioritised.
    while ((len(filenames) == 0) & (sat_ind < len(sat_names)-1)): 
        sat_ind += 1
        filenames=glob.glob(sev_data_dir1+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]+time.strftime("*EPI*%Y%m%d%H%M-*"))+ glob.glob(sev_data_dir1+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]+time.strftime("*PRO*%Y%m%d%H%M-*"))# PRO and EPI files necessary in all scenarios
        sev_dir = sev_data_dir1+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]
    
    
    ### 3 ### Try alternative directory for SEVIRI data if less than 2 files found before. (sev_data_dir2)
    if ((len(filenames) < 2)):
        sat_ind = -1
        while ((len(filenames) == 0) & (sat_ind < len(sat_names)-1)): # Sometimes have data from multiple instruments (e.g. 20160504_1045 has MSG3 and MSG1), this ensures most recent is prioritised.
            sat_ind += 1
            #print("A, filenames=", filenames)
            filenames=glob.glob(sev_data_dir2+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]+time.strftime("*EPI*%Y%m%d%H%M-*"))+ glob.glob(sev_data_dir2+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]+time.strftime("*PRO*%Y%m%d%H%M-*"))# PRO and EPI files necessary in all scenarios
            sev_dir = sev_data_dir2+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]
    
    
    if  ((time == datetime.datetime(2016,4,11,19,0,0))| (time == datetime.datetime(2018,4,23,7,15,0))): # These files are present in sev_dir1, but corrupt
        filenames=glob.glob(sev_data_dir2+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]+time.strftime("*EPI*%Y%m%d%H%M-*"))+ glob.glob(sev_data_dir2+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]+time.strftime("*PRO*%Y%m%d%H%M-*"))# PRO and EPI files necessary in all scenarios
        sev_dir = sev_data_dir2+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]    
    if (len(filenames) < 2):
        #print("B, filenames=", filenames)
        #print('sev_data_dir2+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]+time.strftime("*EPI*%Y%m%d%H%M-*")=', sev_data_dir2+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]+time.strftime("*EPI*%Y%m%d%H%M-*"))
        raise MissingFileError("SEVIRI observations for "+time.strftime("%Y/%m/%d_%H%M")+" are missing. Please check if they can be downloaded and if not, add to the list of unavailable times.")
        
    
    else:
        for channel in channel_list:
            filenames=filenames + glob.glob(sev_dir+'*'+file_dict[channel]+time.strftime("*%Y%m%d%H%M-*")) # add channels required
        #print("C, filenames=", filenames)
        scene = satpy.Scene(reader="seviri_l1b_hrit", filenames=filenames)
        data = {}
        scene.load(channel_list)
        
        if regrid != False:
            lons, lats = my_area.get_lonlats()
            if len(interp_coeffs) == 0:
                interp_coeffs = bilinear.get_bil_info(scene[channel_list[0]].area, my_area, radius=50e3, nprocs=1)
                data.update({'interp_coeffs': interp_coeffs})
            for channel in channel_list:
                data.update({str(channel): bilinear.get_sample_from_bil_info(scene[channel].values.ravel(), interp_coeffs[0], interp_coeffs[1], interp_coeffs[2], interp_coeffs[3], output_shape=my_area.shape)})
        else:
            if len(subdomain) > 0:
                scene = scene.crop(ll_bbox=subdomain)
            lons, lats = scene[channel_list[0]].area.get_lonlats()
            lons = lons[:,::-1] # Need to invert y-axis to get longitudes increasing.
            lats = lats[:,::-1]
            for channel in channel_list:
                data.update({str(channel) : scene[channel].values[:,::-1]})
        data.update({'lons' : lons, 'lats' : lats, 'interp_coeffs' : interp_coeffs})
        # Compressed files are decompressed to TMPDIR. Now tidy up
        # This doesn't seem to be applicable to me...
        delete_list = glob.glob(my_tmpdir+'/'+time.strftime("*%Y%m%d%H%M-*"))
        for d in delete_list: os.remove(d)
        return data


class FileError(Exception):
    """Base class for other exceptions

    """
    pass


class UnavailableFileError(FileError):
    """Raised when the file is not available from EumetSat

    """
    pass


class MissingFileError(FileError):
    """Raised when the file is missing, but we still need to
    check whether it is definitely not available from EumetSat

    """
    pass




gpm_dir = '/badc/gpm/data/GPM-IMERG-v6/' # Change as appropriate


def read_gpm(timelist, lon_min=-20., lon_max=52., lat_min=-35., lat_max=38., varname='HQprecipitation'):
    '''Reads GPM IMERG data for specified times and lat-lon limits

    Args:
        timelist (list): times to be read
        lon_min (float, optonal): Longitude of Western boundary of region of interest. Defaults to -23.
        lon_max (float, optional): Longitude of Eastern boundary of region of interest. Defaults to 58.
        lat_min (float, optional): Latitude of Southern boundary of region of interest. Defaults to -15.
        lat_max (float, optional): Latitude of Northern boundary of region of interest. Defaults to26.
        varname (string, optional): Name of IMERG variable to read. Defaults to precipitationCal

    Returns:
        lon (ndarray, shape(nlon)): Array of longitude values
        lat (ndarray, shape(nlat)): Array of latitude values
        rain (ndarray, shape(ntimes, nlon, nlat)): Array of values for varname

    '''
    if ((varname == 'precipitationNoIRCal') or (varname == 'precipitationNoIRUncal')):
        lon, lat, rain = read_gpm_no_ir(timelist, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max, varname=varname)
        return lon, lat, rain      
    rain = []
    for i, time in enumerate(timelist):
        f = get_gpm_filename(time)
        dataset = h5py.File(f, 'r')
        lon = dataset['Grid']['lon'][:]
        lat = dataset['Grid']['lat'][:]
        ind_lon = np.where((lon >= lon_min) & (lon <= lon_max))[0]
        ind_lat = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        if dataset['Grid'][varname].ndim == 3:
            rain += [dataset['Grid'][varname][0,ind_lon[0]:ind_lon[-1]+1, ind_lat[0]:ind_lat[-1]+1]]
        else:
            print(("dataset['Grid'][varname].ndim=", dataset['Grid'][varname].ndim))
            sys.exit()
    rain = np.ma.masked_array(np.array(rain), mask=(np.array(rain) < 0.0))
    return lon[ind_lon], lat[ind_lat], rain



def get_gpm_filename(time):
    '''Identify GPM IMERG HDF file corresponding to given time

    '''
    f = glob.glob(gpm_dir + time.strftime('%Y/%j/*%Y%m%d-S%H%M*.HDF5'))
    if len(f) != 1:
        print(("gpm_dir + time.strftime('%Y/%j/*%Y%m%d-S%H%M*.HDF5')=", gpm_dir + time.strftime('%Y/%j/*%Y%m%d-S%H%M*.HDF5')))
        print(("f=", f))
        sys.exit()
    return f[0]