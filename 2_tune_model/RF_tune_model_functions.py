import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import sys
import csv
import glob
import math
import pickle
import joblib
import numpy as np
from datetime import timedelta, date
import datetime
import joblib
import scipy.stats as sci
from sklearn.ensemble import RandomForestClassifier
import RF_tune_model_functions as RF

# Set the seed for random actions (for repeatability)
np.random.seed(42)


# Advanced data import
def import_feature_data(startdate, enddate, exclude_MM, exclude_hh, perc_keep, traindir, random, solar, seviri_diff, topography, wavelets):
    print("adding base SEVIRI channel features...")
    features = None # Just in case memory has not been cleared
    files = [] # Initialise empty list for the file paths to be appended to

    for n in range(0, int((enddate - startdate).total_seconds()/(15*60))):
        time = (startdate + n*datetime.timedelta(minutes = 15))
        #print (time.strftime("%Y-%m-%d %H:%M"))

        YYYY = time.strftime("%Y")
        MM = time.strftime("%m")
        DD = time.strftime("%d")
        hh = time.strftime("%H")
        mm = time.strftime("%M")

        if MM in exclude_MM:
            continue
        if hh in exclude_hh:
            continue

        try:
            files.append(glob.glob(traindir+YYYY+'/'+MM+'/'+YYYY+MM+DD+hh+mm+'.pkl')[0])
        except:
            #print ('no file exists for '+time.strftime("%Y-%m-%d %H:%M"))
            continue

    print ('Number of time files: ' + str(len(files)))
    if len(files)*perc_keep < 1.5: print ("perc_keep too low, zero file output is likely")

    # Use the file list to import a random subset of the training features data, using perc_keep to determine the final number included
    dframe_list = []
    for file in files:
        if np.random.rand() <= perc_keep:
            # print (file)
            frame = pd.read_pickle(file)
            dframe_list.append(frame)
    
    features = pd.concat(dframe_list)
    
    # Add supplementary metadat or datasets to the feature list for training
    if random:
        # Add random feature as a benchmark for feature importance
        print("adding random number feature...")
        features['random'] = np.random.rand(len(features['GPM_PR']))
    
    if solar:
        # Add solar elevation and cos/sin of solar azimuth
        print("adding solar metadata...")
        solar_el, cos_solar_az, sin_solar_az = np.zeros((len(features))), np.zeros((len(features))), np.zeros((len(features)))
        t = np.array(features['YYYYMMDDhhmm']).astype(str)
        lat = np.array(features['Latitude'])
        lon = np.array(features['Longitude'])
        # Loop through features and create solar metadata
        for i in range(0, len(features)):
            t_ = t[i]
            solar_el[i], cos_solar_az[i], sin_solar_az[i] = RF.sunpos(int(t_[0:4]), int(t_[4:6]), int(t_[6:8]), int(t_[8:10]), int(t_[10:12]), lat[i], lon[i], refraction=False)
        # Enter the solar metadata into the features array
        features['Solar_elevation'], features['Solar_azimuth_cos'], features['Solar_azimuth_sin'] = solar_el, cos_solar_az, sin_solar_az
        print("done!")
    
    # Dictionary to retain the key features after inclusion
    key_features = []
    
    if seviri_diff:
        print("adding SEVIRI channel difference features...")
        # Red channel on Covection RGB
        features['MSG_6.2-7.3'] = features['MSG_6.2'] - features['MSG_7.3']
        # Green channel on Covection RGB
        features['MSG_3.9-10.8'] = features['MSG_3.9'] - features['MSG_10.8']
        # Blue channel on Covection RGB
        features['MSG_1.6-0.6'] = features['MSG_1.6'] - features['MSG_0.6']
        
        # Every channel minus 10.8 channel        
        features['MSG_0.6-10.8'] = features['MSG_0.6'] - features['MSG_10.8']
        features['MSG_0.8-10.8'] = features['MSG_0.8'] - features['MSG_10.8']
        features['MSG_1.6-10.8'] = features['MSG_1.6'] - features['MSG_10.8']
        features['MSG_6.2-10.8'] = features['MSG_6.2'] - features['MSG_10.8']
        features['MSG_7.3-10.8'] = features['MSG_7.3'] - features['MSG_10.8']
        features['MSG_8.7-10.8'] = features['MSG_8.7'] - features['MSG_10.8']
        features['MSG_9.7-10.8'] = features['MSG_9.7'] - features['MSG_10.8']
        features['MSG_12.0-10.8'] = features['MSG_12.0'] - features['MSG_10.8']
        features['MSG_13.4-10.8'] = features['MSG_13.4'] - features['MSG_10.8']
        
        # Must list all key features to be force-added to chosen_features
        key_features.append(['MSG_6.2-7.3',
                            'MSG_3.9-10.8',
                            'MSG_1.6-0.6',
                            'MSG_0.6-10.8',
                            'MSG_0.8-10.8',
                            'MSG_1.6-10.8',
                            'MSG_6.2-10.8',
                            'MSG_7.3-10.8',
                            'MSG_8.7-10.8',
                            'MSG_9.7-10.8',
                            'MSG_12.0-10.8',
                            'MSG_13.4-10.8'])
        
        
    if topography:
        print("adding pixel topography features...")
        # Load topography file
        elevation_file = pickle.load( open('/gws/nopw/j04/swift/bpickering/random_forest_precip/1_training_data/topography/RF_elevation.pkl', "rb") )
        prominence_file = pickle.load( open('/gws/nopw/j04/swift/bpickering/random_forest_precip/1_training_data/topography/RF_prominence.pkl', "rb") )
        
        # Pull out the latitude and longitude arrays from pandas.df
        lat = np.array(features['Latitude'])
        lon = np.array(features['Longitude'])
        
        # Lats and lons of the domain
        lats = np.linspace(37.95, -34.95, 730)
        lons = np.linspace(-19.95, 51.95, 720)
        
        # Make blank 1D array the length of features, to capture the topography data
        elevation = np.ma.masked_all(len(features))
        prominence = np.ma.masked_all(len(features))
        for i in range(0, len(features)):
            # Use lat/lon to extract
            y = np.where(np.round(lats*100)/100 == np.round(lat[i]*100)/100)[0]
            x = np.where(np.round(lons*100)/100 == np.round(lon[i]*100)/100)[0]
            
            elevation[i] = elevation_file['elevation'][ y , x ]
            prominence[i] = prominence_file['prominence'][ y , x ]

        # add to features array
        features['elevation'] = elevation
        features['prominence'] = prominence
        
        # Must list all key features to be force-added to chosen_features
        key_features.append(['elevation',
                            'prominence'])
        
        
    if wavelets:
        print("adding wavelet features...")
        wavdir = '/home/users/bpickering/bpickering_swift/random_forest_precip/1_training_data/wavelets/RF_training_data/'
        
        # Pull out the latitude and longitude arrays from pandas.df
        lat = np.around(np.array(features['Latitude']), 2)
        lon = np.around(np.array(features['Longitude']), 2)
        
        # Read in one wavelet file just to get dimensions
        t = np.array(features['YYYYMMDDhhmm']).astype(str)
        t_ = t[0]
        wav = pd.read_pickle(
            wavdir+t_[0:4]+'/'+t_[4:6]+'/wavelets___Tcut_-40_Twav_-50___'+t_[0:4]+t_[4:6]+t_[6:8]+t_[8:10]+t_[10:12]+".pkl",
            compression='gzip'
        )
        # Make lengths label list, and make an empty array for all the wavelet data.
        lengths = list(wav.columns[2:])
        print('Wavelet length scales:', lengths)
        wav_data = np.zeros((len(features), len(lengths)))
        
        # Turn wavelets into np.array for faster slicing
        wav_np = np.around(np.array(wav), 2)
        
        # Loop through each element in the features array
        for i in range(0,len(features)):
            
            # Select the current time, and check if this is new from the last loop.
            # Works on first loop because t_ set as t[0] above, and wav for t[0] is also defined above.
            if not t_ == t[i]:
                # if new time, set it (t_) and import new wavelets file
                t_ = t[i]
                wav = pd.read_pickle(
                    wavdir+t_[0:4]+'/'+t_[4:6]+'/wavelets___Tcut_-40_Twav_-50___'+t_[0:4]+t_[4:6]+t_[6:8]+t_[8:10]+t_[10:12]+".pkl",
                    compression='gzip'
                )
                wav_np = np.around(np.array(wav), 2) # Turn wavelets into np.array for faster slicing
                
            # use features lat and lon in here
            # extracts wavelets for that lat and lon
            wav_data[i, :] = wav_np[(wav_np[:,0] == lat[i]) & (wav_np[:,1] == lon[i]), 2:][0]
            
        # add each scale of wavelet powers to the features array
        c = 0
        for length in ['30.0','60.0', '120.0', '240.0', '302.4']: #lengths: #!!!!!
            features['w_'+length] = wav_data[ : , c ]
            
            # Must list all key features to be force-added to chosen_features
            key_features.append(['w_'+length])
            
            c += 1
        
        print("done!")


    # Save some memory
    dframe_list = None
    frame = None

    print('The shape of the features array is:', features.shape)
    print('The size of the features array is: ' + str(sys.getsizeof(features)/1000000000)[:-6] + ' GB.')
    
    return features, key_features



# Sort the feature data by binning, clipping and reformating the DataType
# Works if new features added previously with func: import_feature_data
def sort_feature_data(features, bin_edges, force_features, desired_feature_list):
    # Labels are the values we want to predict
    labels = np.array(features['GPM_PR'])

    # Apply binning of data
    labels = np.digitize(labels, bins=bin_edges, right=False)

    # Remove the labels from the features
    # axis 1 refers to the columns
    features_no_labels= features.drop('GPM_PR', axis = 1)
    features_pd= features_no_labels.drop('YYYYMMDDhhmm', axis = 1)
    print (features_pd.head(1))
    
    
    if force_features:
        # Pass features just imported, and a list of desired features.
        # Will fail if the desired_feature_list has any typo or doesn't exist.
        features_pd = features_pd[desired_feature_list]


    # Saving feature names for later use
    feature_list = list(features_pd.columns)
    

    # Convert to numpy array
    features = np.array(features_pd)
    return features, labels, feature_list




def create_bin_values_and_labels(boundaries):
    '''
    Bin values and labels for converting digitized data and for graphing
    '''
    bins_ = boundaries.split(sep=",")
    bin_edges = np.array(bins_).astype(np.float64)
    bin_values = {}
    bin_labels = {}
    for i in range(0,len(bins_)+1):
        if i == 0:
            bin_values.update({i: "Error_<_"+bins_[i]})
            bin_labels.update({i: "< "+bins_[i]})
        elif i == 1:
            bin_values.update({1: np.float64(0.0)})
            bin_labels.update({1: '0.0'})
        elif i == len(bins_):
            bin_values.update({len(bins_): "Error_>_"+bins_[i-1]})
            bin_labels.update({i: "> "+bins_[i-1]})
        else:
            bin_values.update({i: (bin_edges[i-1]+bin_edges[i])/2})
            bin_labels.update({i: bins_[i-1]+"–"+bins_[i]})
            
    return bin_edges, bin_values, bin_labels


def precip_bin_values(data, bin_values):
    '''
    Convert digitised rain rate class numbers into real rain rate values.
    Rain rate values were calculated as a dictionary in the create_bin_values_and_labels function.
    '''
    precip_values = np.ma.copy(data)
    
    # Check for values outside the boundaries (nominally < 0 mm/h and > 200 mm/h)
    if len(precip_values[data==0]) or len(precip_values[data==len(bin_values.items())-1]) > 0:
        raise ValueError("Values less than zero mm/h or greater than the maximum rain boundary exist within the verification table data.")
    
    # Convert integer class-values to rain rate values
    for k, v in bin_values.items():
        if type(v) == np.float64:
            precip_values[data==k] = v
            
    return precip_values


def sunpos(year, month, day, hour, minute, lat, lon, refraction):
    '''
    Written by John Clark Craig https://levelup.gitconnected.com/python-sun-position-for-solar-energy-and-research-7a4ead801777
    
    Parameters:
    year, month, day, hour, minute   Type: float.
                                     Time must be in UTC.
    lat, lon                         Type float.
    refraction                       Type Bool.
                                     Whether to use refraction correction.
    
    Returns:
    Azimuth     Type: float. Solar position accurate to 0.01 degrees with 2 d.p. precision
    Elevation   Type: float. Solar position accurate to 0.01 degrees with 2 d.p. precision
    '''
    # Math shortcuts and radians conversion of lat/lon
    rad, deg, sin, cos, tan, asin, atan2 = math.radians, math.degrees, math.sin, math.cos, math.tan, math.asin, math.atan2 # Math typing shortcuts
    rlat, rlon = rad(lat), rad(lon) # Convert latitude and longitude to radians
    
    # Days from J2000, accurate from 1901 to 2099
    daynum = (367 * year - 7 * (year + (month + 9) // 12) // 4 + 275 * month // 9
              + day - 730531.5 + (hour + (minute / 60)) / 24)
    
    # Solar settings
    mean_long = daynum * 0.01720279239 + 4.894967873 # Mean longitude of the sun
    mean_anom = daynum * 0.01720197034 + 6.240040768 # Mean anomaly of the sun
    eclip_long = (mean_long + 0.03342305518 * sin(mean_anom) + 0.0003490658504 * sin(2 * mean_anom)) # Ecliptic longitude of the sun
    obliquity = 0.4090877234 - 0.000000006981317008 * daynum # Obliquity of the ecliptic
    rasc = atan2(cos(obliquity) * sin(eclip_long), cos(eclip_long)) # Right ascension of the sun
    decl = asin(sin(obliquity) * sin(eclip_long)) # Declination of the sun
    sidereal = 4.894961213 + 6.300388099 * daynum + rlon # Local sidereal time
    hour_ang = sidereal - rasc # Hour angle of the sun
    
    # Sun position in radians
    elevation = asin(sin(decl) * sin(rlat) + cos(decl) * cos(rlat) * cos(hour_ang)) # Local elevation of the sun
    azimuth = atan2(-cos(decl) * cos(rlat) * sin(hour_ang), sin(decl) - sin(rlat) * sin(elevation)) # Local azimuth of the sun
    
    # Convert azimuth and elevation to degrees
    azimuth, elevation = into_range(deg(azimuth), 0, 360), into_range(deg(elevation), -180, 180)
    
    # Refraction correction (optional)
    if refraction:
        targ = rad((elevation + (10.3 / (elevation + 5.11))))
        elevation += (1.02 / tan(targ)) / 60

    return round(elevation, 2), round(np.cos((azimuth)/360 *2*np.pi),3), round(np.sin((azimuth)/360 *2*np.pi),3)


def into_range(x, range_min, range_max):
    shiftedx = x - range_min
    delta = range_max - range_min
    return (((shiftedx % delta) + delta) % delta) + range_min


# def add_solar(ver_arr):
#     '''
#     Pass the verification array, and it will use the datetime and location to fill in
#     solar azimuth (cos and sin) and solar elevation into the verification array.
#     '''
#     # Loop through whole verification array
#     for i in range(0, len(ver_arr)):
#         # Retrieve solar azimuth and elevation
#         azimuth, elevation = sunpos(year=int(str(ver_arr[i, 0])[:4]),
#                                     month=int(str(ver_arr[i, 0])[4:6]),
#                                     day=int(str(ver_arr[i, 0])[6:8]),
#                                     hour=int(str(ver_arr[i, 0])[8:10]),
#                                     minute=int(str(ver_arr[i, 0])[10:12]), 
#                                     lat=ver_arr[i, 5],
#                                     lon=ver_arr[i, 6],
#                                     refraction=False)
#         # Put solar azimuth and elevation data into verification array
#         ver_arr[i, 7] = np.cos((azimuth)/360 *2*np.pi)
#         ver_arr[i, 8] = np.sin((azimuth)/360 *2*np.pi)
#         ver_arr[i, 9] = elevation
        
#     return ver_arr


def plot_importances(RF_dict, outdir):
    '''
    RF_dict   is a dictionary of random forest models which will be looped through.
              The importances are embedded within each model object.
    flabels   is a list of the feature labels, not contained in the model. 
              Must match original model order.
    
    writes a csv with the features labelled and their importance.
    plots a vertical bar chart to show their importance against random, if incldued. So works for both randomisedCVsearch and final verification.
    '''
    for model in RF_dict:
        # Select the model and the feature labels
        clf = RF_dict[model]['model']
        flabels = RF_dict[model]['labels'].copy()
        
        # Get numerical feature importances
        importances = list(clf.feature_importances_)
        print ("importances within function loop step 1:", importances)
        
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(flabels, importances)]
        print ("importances within function loop step 2:", feature_importances)
        
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        print ("importances within function loop step 3:", feature_importances)
        
        # Export the features and importances as     
        with open(outdir+'feature_importances_'+model+'.csv',"w+") as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(feature_importances)


        # Set the style
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(15,10))

        # list of x locations for plotting
        x_values = list(range(len(importances)))

        # Make a bar chart (differs if random is included) 
        if 'random' in flabels:
            # Work out where it is
            rand_index = flabels.index('random')

            # Plot red line instead
            plt.plot([-0.5, -1.5+len(importances)],[importances[rand_index], importances[rand_index]], color='red', linestyle='--', linewidth=3., label='Random')

            # Remove from importances and plot
            del importances[rand_index]
            flabels.remove('random')

            plt.bar(x_values[:-1], importances, orientation = 'vertical')
            plt.legend()

        else:
            plt.bar(x_values, importances, orientation = 'vertical')

        # Tick labels for x axis
        plt.xticks(x_values, flabels, rotation='vertical')

        # Axis labels and title
        plt.ylabel('Importance'); plt.xlabel('Feature'); plt.title(model+' Feature Importances');
        plt.axis([None, None, -0.01, None])

        # Save
        plt.savefig(outdir+'feature_importances_'+model+'.png', bbox_inches="tight", dpi=250)
        
    return feature_importances