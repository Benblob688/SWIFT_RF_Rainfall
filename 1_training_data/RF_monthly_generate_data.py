'''
Code to generate an array for training a random forest for a single requested time.

Usage:

RF_generate_data.py <YYYY> <MM>

inputs: date only (<hh> <mm> are both done inside this script)
accesses: GPM on CEDA, SEVIRI on SWIFT GWS, 
outputs: array of pixel-pairs for GPM, MSG, NWCSAF, GFS, terrain etc. with a filename YYYYMMDDhhmm.pkl
'''
###############
###### 0 ###### import everything and the separate module of functions needed
###############
import satpy # Requires virtual environment for reading native (.nat) and hrit files.
import numpy as np
import numpy.ma as ma
import datetime
import glob
from pyresample import geometry, bilinear
import os
import sys
import pickle
import h5py
import pandas as pd

print (sys.argv[:])

# You may see warnings due to cated pyproj features and when numpy/dask process NaNs values. NaNs are used to mark invalid pixels and space pixels (like you'll see in a full disk image).
import warnings
warnings.filterwarnings('ignore')

import RF_generate_data_functions as RF # all the messy functions to import all the different datasets are within here

# bring in the interp_coeffs tuple
with open('/home/users/bpickering/bpickering_swift/random_forest_precip/1_training_data/interp_coeffs_SEVIRI_to_GPM.pickle', 'rb') as f:
     interp_coeffs = pickle.load(f)

outdir = "/work/scratch-pw/bpickering/RF_generate_features/"   #"/home/users/bpickering/bpickering_swift/features_datasets/"


# Different days for each month as I work through error-producing times and add them to the missing datelist in RF_generate_data_func.py
if sys.argv[2] == '01':
    days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
elif sys.argv[2] == '02':
    days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']
elif sys.argv[2] == '03':
    days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
elif sys.argv[2] == '04':
    days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
elif sys.argv[2] == '05':
    days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
elif sys.argv[2] == '06':
    days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
elif sys.argv[2] == '07':
    days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
elif sys.argv[2] == '08':
    days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
elif sys.argv[2] == '09':
    days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
elif sys.argv[2] == '10':
    days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
elif sys.argv[2] == '11':
    days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
elif sys.argv[2] == '12':
    days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
else:
    print ("something is wrong with the month day selection")

for DD in days:
    for hh in ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']:
        for mm in ['00', '15', '30', '45']:
            print (DD + hh + mm)
            ###############
            ###### 1 ###### Setup time and blank array
            ###############
            # The start time of the window, i.e. 12:45 is actually 12:45–13:00 and GPM is sliced accordingly
            time = datetime.datetime(int(sys.argv[1]), int(sys.argv[2]), int(DD), int(hh), int(mm))   # int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6] for when used as a script
            
            # Check that SEVIRI exists otherwise move on to next time step (consider adding other)
            try:
                if time in RF.unavailable_times:
                    raise RF.UnavailableFileError("SEVIRI observations for "+time.strftime("%Y/%m/%d_%H%M")+" are not available")
            except:
                print ("SEVIRI observations for "+time.strftime("%Y/%m/%d_%H%M")+" are not available")
                continue

            # create the final array or pandas dataframe
            # it should be number of hyper parameters by 332,100 (410x810) which is the number of pixels in the region. Most will be removed in the last step.
            data = np.ma.zeros((525600, 19))

            # Columns are:
            # 0: YYYYMMDDhhmm
            # 1: annual_cos
            # 2: annual_sin
            # 3: dirunal_cos
            # 4: diurnal_sin
            # 5: lat [10.0, 10.0,....][9.9, 9.9, ....]
            # 6: lon [-10, -9.9, -9.8, ...][-10, -9.9, -9.8, ...]
            # 7: IMERG GPM (use np.array.flatten)
            # 8: MSG 0.6 (use np.array.flatten)
            # 9: MSG 0.8 (use np.array.flatten)
            # 10: MSG 1.6 (use np.array.flatten)
            # 11: MSG 3.9 (use np.array.flatten)
            # 12: MSG 6.2 (use np.array.flatten)
            # 13: MSG 7.3 (use np.array.flatten)
            # 14: MSG 8.7 (use np.array.flatten)
            # 15: MSG 9.7 (use np.array.flatten)
            # 16: MSG 10.8 (use np.array.flatten)
            # 17: MSG 12.0 (use np.array.flatten)
            # 18: MSG 13.4 (use np.array.flatten)
            # 


            ###############
            ###### 2 ###### Fill it with easy data like datetime (from the sys.argv user input), lat lon etc
            ###############
            data[:, 0] = np.repeat(int(sys.argv[1]+sys.argv[2]+DD+hh+mm), 525600)  # time in format YYYYMMDDhhmm repeated to fill array
            # For ML, the time needs to be cyclical, so have a sin and a cos
            data[:, 1] = np.repeat(np.cos((time.timetuple().tm_yday / datetime.datetime(time.year, 12, 31).timetuple().tm_yday)*2*np.pi), 525600) # cos of DoYear / days in that year
            data[:, 2] = np.repeat(np.sin((time.timetuple().tm_yday / datetime.datetime(time.year, 12, 31).timetuple().tm_yday)*2*np.pi), 525600) # sin of DoYear / days in that year
            data[:, 3] = np.repeat(np.cos((int(hh)*60 +int(mm))/1440*2*np.pi), 525600) # diurnal time in cyclical format (cos of minutes / 1440 in radians so *2pi)
            data[:, 4] = np.repeat(np.sin((int(hh)*60 +int(mm))/1440*2*np.pi), 525600) # diurnal time in cyclical format (sin of minutes / 1440 in radians so *2pi)

            data[:, 5] = np.repeat(np.linspace(37.95, -34.95, 730), 720)   # latitude repeated because order is top left to bottom right
            data[:, 6] = np.tile(np.linspace(-19.95, 51.95, 720), 730)     # longitude tiled because order is top left to bottom right

            ###############
            ###### 3 ###### Fill it with static data like topography, land type etc. (not yet implemented)
            ###############

            ###############
            ###### 4 ###### Open the GPM data with read_gpm_hdf5 function and fill it, including the NaNs (labels / truth)
            ###############
            if time.minute == 15 or time.minute == 45:
                latter_15min = True
                GPMstart = time - datetime.timedelta(minutes=15)
                #print (GPMstart)
            elif time.minute == 0 or time.minute == 30:
                latter_15min = False
                GPMstart = time
                #print (GPMstart)
            else:
                raise KeyError("The time you have selected is not on Meteosat observation frequency. Please use a 15-min interval (00, 15, 30, or 45)")

            #if time is a list, then it will return 2D spacial with 3rd dimension being time
            try:
                GPM_lon, GPM_lat, GPM_rain = RF.read_gpm([GPMstart], varname='HQprecipitation')   #HQprecipitation   precipitationCal    HQobservationTime
            except:
                print ("IMERG observations for "+time.strftime("%Y/%m/%d_%H%M")+" cannot be opened")
                continue
            
            # HQobservationtime
            GPM_lon, GPM_lat, GPM_time = RF.read_gpm([GPMstart], varname='HQobservationTime')   #HQprecipitation   precipitationCal    HQobservationTime


            GPM = ((np.flip(GPM_rain[0].T, axis=0)).flatten())
            GPM_time_check = ((np.flip(GPM_time[0].T, axis=0)).flatten())

            if latter_15min:
                GPM = np.ma.masked_where(GPM_time_check < 15, GPM)
            else:
                GPM = np.ma.masked_where(GPM_time_check >= 15, GPM)

            # slice sorted GPM pixels into the RF data array   
            data[:, 7] = GPM

            ###############
            ###### 5 ###### open the NWCSAF data with _________ function and fill it (baseline1). (not yet implemented)
            ###############

            ###############
            ###### 6 ###### open the HKV data with _________ function and fill it (baseline2). (not yet implemented)
            ###############

            ###############
            ###### 7 ###### open the SEVIRI channel data with read_seviri function, regridding all channels with interp_coeffs
            ###############
            try:
                SEVIRI = RF.read_seviri_channel(channel_list  = [0.6, 0.8, 1.6, 3.9, 6.2, 7.3, 8.7, 9.7, 10.8, 12.0, 13.4],
                                                 time=time, subdomain=(), regrid=True, 
                                                 my_area=geometry.AreaDefinition('pan_africa', 
                                                                                 'Pan-Africa on Equirectangular 0.1 degree grid used by GPM', 
                                                                                 'pan_africa', {'proj' : 'eqc'}, 720, 730, 
                                                                                 (-2226389.816, -3896182.178, 5788613.521, 4230140.650)), 
                                                                                 interp_coeffs=(interp_coeffs))
            except:
                print ("SEVIRI observations for "+time.strftime("%Y/%m/%d_%H%M")+" are not available")
                continue
            

            col = 8
            for channel in [0.6, 0.8, 1.6, 3.9, 6.2, 7.3, 8.7, 9.7, 10.8, 12.0, 13.4]:
                data[:, col] = (SEVIRI[str(channel)].flatten())
                col +=1

            ###############
            ###### 8 ###### open the GFS or other NWP data and extract the values desired for this region...
            ###############

            ###############
            ###### 9 ###### now sort through the whole array /pandas dataframe and remove rows where any NaNs occur in (any? column)
            ###############
            # Transfer the data array into a Pandas DataFrame
            features = pd.DataFrame(data=data, columns=["YYYYMMDDhhmm", "Annual_cos", "Annual_sin", "Diurnal_cos", "Diurnal_sin", "Latitude", "Longitude", "GPM_PR", "MSG_0.6", "MSG_0.8", "MSG_1.6", 
                                                       "MSG_3.9", "MSG_6.2", "MSG_7.3", "MSG_8.7", "MSG_9.7", "MSG_10.8", "MSG_12.0", "MSG_13.4"])
            # Drop any row which contains a NaN
            features = features.dropna()
            # Drop any row where 10.8 µm BT > 273.15 K (because this will likely not be a cloud)
            features = features[features['MSG_10.8'] < 273.15]

            # Save the filtered Pandas DataFrame
            features.to_pickle(outdir+"/"+sys.argv[1]+"/"+sys.argv[2]+"/"+sys.argv[1]+sys.argv[2]+DD+hh+mm+".pkl")
