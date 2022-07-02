import pandas as pd
import matplotlib
# matplotlib.use("agg")
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import sys
import glob
import math
import pickle
import joblib
import numpy as np
from datetime import timedelta, date
import datetime
import scipy.stats as sci
np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.filterwarnings('ignore') # comment out if the software is not working, may help diagnose issues. Otherwise unnecessary 
np.random.seed(42)
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import csv
from sklearn import tree
import string
import RF_verification_functions as RF # Very important must import so that these functions work when called inside these other functions.




def import_feature_data_verification(startdate, enddate, exclude_MM, exclude_hh, perc_exclude,
                                    traindir, random, solar, seviri_diff, topography, wavelets):
    '''           Usage:
    startdate     Type: datetime YYYY MM DD hh mm. 
    enddate       Type: datetime YYYY MM DD hh mm. 
    exclude_MM    Type: list of strings with leading zero, e.g. [‘01’, ‘02’].
    exclude_hh    Type: list of strings with leading zero, e.g. [‘01’, ‘02’].
    perc_keep     Type: float. Set what approximate percentage of the file list should be imported for training.
    traindir      Type: str. Path to the training data directory.
    random        Type: Bool. Whether to include random number generator in the feature list.
    solar         Type: Bool. Whether to include solar eleveation and azimuth in the feature list.
    ____ !!!! add more feature y/n bools here, such as specific SEVIRI channels (after RandomizedSearchCV) and new data like CEH wavelet powers and topography !!!!!!!!

                  Returns:
    features      Type: pandas.dataframe. All features and label data valid for the user-input settings.
    '''
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
            files.append(glob.glob('/gws/nopw/j04/swift/bpickering/random_forest_precip/1_training_data/'+YYYY+'/'+MM+'/'+YYYY+MM+DD+hh+mm+'.pkl')[0])
        except:
            #print ('no file exists for '+time.strftime("%Y-%m-%d %H:%M"))
            continue
            

    print ('Number of time files: ' + str(len(files)) + ' * ' + str(1-perc_exclude))
    if len(files)*(1-perc_exclude) < 1.5: print ("perc_exclude too high, zero file output is likely")

    # Use the file list to import a random subset of the training features data, using perc_keep to determine the final number included
    dframe_list = []
    for file in files:
        if np.random.rand() > perc_exclude:
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
        for length in ['30.0','60.0', '120.0', '240.0', '302.4']: #lengths:
            features['w_'+length] = wav_data[ : , c ]
            
            c += 1
        
        print("done!")

    # Save some memory
    dframe_list = None
    frame = None

    print('The shape of the features array is:', features.shape)
    print('The size of the features array is: ' + str(sys.getsizeof(features)/1000000000)[:-6] + ' GB.\n')
    
    return features






def sort_feature_data(features, bin_edges):
    '''           Usage:
    classifier    Type: bool. Values must be binned using the precision below.
    precision     Type: float. Set the precipitation rate resolution of the model in mm h-1. Example: 5 = model will be trained to classify to the nearest 5 mm h-1.
    clip          Type: bool. Limit the minimum and maximum value and nudge all values outside to be equal to these.
    min_clip      Type: float. All values below this value will be increased, to equal this value.
    max_clip      Type: float. All values below this value will be increased, to equal this value.

                  Returns:
    features      Type: np.array. Feature values for each pixel, where the indexing matches the variable "labels" for a given pixel.
    labels        Type: np.array. True values for each pixel, where the indexing matches the variable "features" for a given pixel.
    feature_list  Type: list. Names for all the columns in the features set.

    Must be run immediately after import_feature_data
    '''
    # Labels are the values we want to predict
    labels = np.array(features['GPM_PR'])

    # Apply binning of data (no mask exists nor is needed, unlike CRR/Ph/NIPE. Feature data files by design always have SEVIRI and IMERG inputs.)
    labels = np.digitize(labels, bins=bin_edges, right=False)

    # Remove the labels from the features
    # axis 1 refers to the columns
    features_no_labels= features.drop('GPM_PR', axis = 1)
    features_pd= features_no_labels.drop('YYYYMMDDhhmm', axis = 1)
    #print (features_pd.head(5))

    # Saving feature names for later use
    feature_list = list(features_pd.columns)

    # Convert to numpy array
    features = np.array(features_pd)
    return features, labels, feature_list




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





def add_CRR(ver_arr, bin_edges, CRR_dir = '/gws/nopw/j04/swift/bpickering/random_forest_precip/4_verify_model/CRR_regridded/data/'):
    '''
    Add CRR to the ver_arr array. Must provide path to 10 km regridded CRR data in sub-directiories with YYYY/YYYYMMDD/, and filename must be in format NWCSAF_regridded_PanAfrica_YYYYMMDDThhmm.npy
    '''
    # Create lookup dict for lat/lon to index of array
    lats, lat_indexes = np.linspace(-34.95, 37.95, 730), np.linspace(0,730,731)
    lons, lon_indexes = np.linspace(-19.95, 51.95, 720), np.linspace(0,720,721)
    lati = dict(zip(np.round(lats*100)/100, lat_indexes))
    loni = dict(zip(np.round(lons*100)/100, lon_indexes))

    # Setup pre-variables
    last_strdate = 'none'

    # Loop over the length of the verification array
    for i in range(0, len(ver_arr)):
        # Extract just the date from the current row of the verification array
        strdate = (str(ver_arr[i,0])[:-2])

        # Only load new file if date has changed
        if strdate != last_strdate:
            # Attempt to open the CRR file for that date
            try:
                file = (glob.glob(CRR_dir+strdate[:4]+'/'+strdate[:8]
                                        +'/NWCSAF_regridded_PanAfrica_'
                                        +strdate[:8]+'T'+strdate[8:]+'.npy')[0])
                #print(file)
                CRR = np.load(file, allow_pickle=True)

            except:
                #print ('no file exists for '+strdate)
                continue

        # Store the CRR value for the current pixel in the verification array
        ver_arr[i, 11] = CRR.T[int( lati[np.round(ver_arr[i,5]*100)/100] ) , int( loni[np.round(ver_arr[i,6]*100)/100] ) ]

        # Save the current file time to avoid re-ingesting the file on the next iteration of the loop
        last_strdate = strdate

    
    # Apply binning of data
    digitized = np.digitize(ver_arr[:, 11], bins=bin_edges, right=False) # does not copy over mask
    ver_arr[:, 11] = np.ma.array(data=digitized, mask=ver_arr[:, 11].mask) # so we must add in the mask again
    
    return ver_arr








def add_CRR_Ph(ver_arr, bin_edges, CRR_Ph_dir = '/gws/nopw/j04/swift/bpickering/random_forest_precip/4_verify_model/CRR-Ph_regridded/data/'):
    '''
    Add CRR_Ph to the ver_arr array. Must provide path to 10 km regridded CRR-Ph data in sub-directiories with YYYY/YYYYMMDD/, and filename must be in format CRR-Ph_regridded_PanAfrica_YYYYMMDDThhmm.npy
    '''
    # Create lookup dict for lat/lon to index of array
    lats, lat_indexes = np.linspace(-34.95, 37.95, 730), np.linspace(0,730,731)
    lons, lon_indexes = np.linspace(-19.95, 51.95, 720), np.linspace(0,720,721)
    lati = dict(zip(np.round(lats*100)/100, lat_indexes))
    loni = dict(zip(np.round(lons*100)/100, lon_indexes))

    # Setup pre-variables
    last_strdate = 'none'

    # Loop over the length of the verification array
    for i in range(0, len(ver_arr)):
        # Extract just the date from the current row of the verification array
        strdate = (str(ver_arr[i,0])[:-2])

        # Only load new file if date has changed
        if strdate != last_strdate:
            # Attempt to open the CRR_Ph file for that date
            try:                
                file = (glob.glob(CRR_Ph_dir+strdate[:4]+'/'+strdate[:8]
                                        +'/CRR-Ph_regridded_PanAfrica_'
                                        +strdate[:8]+'T'+strdate[8:]+'.npy')[0])
                CRR_Ph = np.load(file, allow_pickle=True)

            except:
                #print ('no file exists for '+strdate)
                continue

        # Store the CRR_Ph value for the current pixel in the verification array
        ver_arr[i, 12] = CRR_Ph.T[int( lati[np.round(ver_arr[i,5]*100)/100] ) , int( loni[np.round(ver_arr[i,6]*100)/100] ) ]

        # Save the current file time to avoid re-ingesting the file on the next iteration of the loop
        last_strdate = strdate

    # Apply binning of data
    digitized = np.digitize(ver_arr[:, 12], bins=bin_edges, right=False) # does not copy over mask
    ver_arr[:, 12] = np.ma.array(data=digitized, mask=ver_arr[:, 12].mask) # so we must add in the mask again
        
    return ver_arr



def create_bin_values_and_labels(boundaries):
    '''
    Bin values and labels for converting digitized data and for graphing
    '''
    bin_edges = np.array(boundaries).astype(np.float64)
    bin_values = {}
    bin_labels = {}
    for i in range(0,len(boundaries)+1):
        if i == 0:
            bin_values.update({i: "Error_<_"+boundaries[i]})
            bin_labels.update({i: "< "+boundaries[i]})
        elif i == 1:
            bin_values.update({1: np.float64(0.0)})
            bin_labels.update({1: '0.0'})
        elif i == len(boundaries):
            bin_values.update({len(boundaries): "Error_>_"+boundaries[i-1]})
            bin_labels.update({i: "> "+boundaries[i-1]})
        else:
            bin_values.update({i: (bin_edges[i-1]+bin_edges[i])/2})
            bin_labels.update({i: boundaries[i-1]+"–"+boundaries[i]})
            
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



def mask_binary(precip_data):
    # Base array for stacking all the arrays into
    precip_mask = None
    precip_mask = np.zeros((730, 720))
    
    for i in range(730):
        for j in range(720):
            if precip_data.mask[i, j]:
                precip_mask[i, j] = 1
        
    return precip_mask



def plot_rainfall(YYYY, MM, DD, hh, mm, precip_data, precip_mask, origin='unknown_origin', settings='', show=False):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-20, 52, -35, 38], crs=ccrs.PlateCarree())
    
    # Map Features
    ax.add_feature(cfeature.OCEAN, linewidth=0., edgecolor=(0,0,0,0), facecolor=(0,0,0,0.25))
    ax.add_feature(cfeature.LAND, linewidth=0., edgecolor=(0,0,0,0), facecolor=(0,0,0,0.4))
    ax.add_feature(cfeature.COASTLINE, edgecolor=(0,0,0,1), linewidth=1., zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1., zorder=3)
    ax.add_feature(cfeature.LAKES, edgecolor=(0,0,0,0), linewidth=0.25, facecolor=(1,1,1,0.4), zorder=1)
    ax.add_feature(cfeature.LAKES, edgecolor=(0,0,0,1), linewidth=0.25, facecolor=(0,0,0,0), zorder=3)
    #ax.add_feature(cfeature.RIVERS, edgecolor=(0,0,0,0.1))
    
    # Precipitation Data
    cmap = matplotlib.cm.viridis
    cmap.set_under((0,0,0,0))
    precip = ax.imshow(precip_data, extent=[-20, 52, -35, 38], 
                       vmin=0.1, vmax=150, interpolation='nearest',
                       cmap=cmap, alpha=1., zorder=2)
    
    cmap_mask = matplotlib.cm.binary
    cmap_mask.set_under((0,0,0,0))
    ax.imshow(precip_mask, extent=[-20, 52, -35, 38], 
                   vmin=0.0001, interpolation='nearest', 
                   cmap=cmap_mask, alpha=0.3, zorder=5)
    
    plt.colorbar(precip, shrink=0.5, pad = 0.01, cmap=cmap, extend='max', 
                 label='Precipitation Rate (mm h$^{-1}$)')

    # Save Figure
    filename = (origin + '_rainfall_' + str(YYYY) + str(MM) + str(DD) + str(hh) + str(mm) + settings)
    plt.title(filename)
    plt.savefig(outdir+filename+'.pdf', interpolation='nearest', format='pdf', bbox_inches="tight", dpi=300)
    plt.savefig(outdir+filename+'.png', bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def multi_HSS(matrix):
    '''
    Calculate multi-dimensional HSS on any square array passed to it.
    '''
    n = sum(sum(matrix))
    PC = 0
    PCR = 0
    
    for i in range (0, len(matrix)):
        PC += matrix[i,i] / n
        PCR += (sum(matrix[i,:]) / n)*(sum(matrix[:,i]) / n)

    HSS = (PC - PCR) / (1 - PCR)

    return HSS





def general_stats(truth, test, bins, PRINT=False):
    '''
    Calculate several common statistical metrics for comparing two 1D datasets. The bin widths of the products must be specified.
    '''
    # Errors
    errors = (test - truth)
    abs_errors = abs(errors)
    
    if not len(abs_errors) > 1:
        mae = np.ma.masked_all(1)
        max_err = np.ma.masked_all(1)
        cov = np.ma.masked_all(1)
        mean_err = np.ma.masked_all(1)
        bias = np.ma.masked_all(1)
        p_corr = np.ma.masked_all(1)
        r2 = np.ma.masked_all(1)
        s_corr = np.ma.masked_all(1)
        mse = np.ma.masked_all(1) 
        rmse = np.ma.masked_all(1)
        hss = np.ma.masked_all(1)
        
    else:
        # Mean Absolute Error
        mae = np.round(np.ma.mean(abs_errors), 2)
        max_err = np.round(np.ma.max(abs_errors), 2) # This one makes little sense when the products are all so poor skilled, the max will always be 60 mm/h
        
        # Covariance
        cov = np.round(np.cov(test, truth)[0,1], 2)

        # Bias
        mean_err = np.round(np.ma.mean(errors), 2) # also referred to as bias
        bias = np.round(np.ma.mean(test)/np.ma.mean(truth), 2)# Specifically, frequency bias

        # r^2
        # calculate Pearson's correlation (must be Gaussian distributed data, which precipitation is not)
        p_corr, _ = sci.pearsonr(test, truth)
        r2 = p_corr**2
        p_corr = np.round(p_corr, 2)
        r2 = np.round(r2, 2)

        # calculate spearman's correlation (non-Gaussian distributed data)
        s_corr, _ = sci.spearmanr(test, truth)
        s_corr = np.round(s_corr, 2)

        # MSE and RMSE
        mse = np.ma.mean((test-truth)**2)
        rmse = np.round(mse**0.5, 2)
        mse = np.round(mse, 2)

        # HSS 
        # Create the 2D histogram matrix using bins provided
        matrix, xedges, yedges = np.histogram2d(truth, test, bins=bins)
        hss = np.round(multi_HSS(matrix), 2)
        

    if PRINT:
        # Print out the stats
        print('Mean Absolute Error:', mae, 'mm/h.')
        print('Maximum Error:', max_err, 'mm/h.')
        print('Covariance:', cov)
        print('Mean Error:', mean_err, 'mm/h.')
        print('Freuqency Bias:', bias)
        print('Pearsons correlation:', p_corr)
        print('Coeffiecient of Determination:', r2)
        print('Spearmans correlation:', s_corr)
        print('Mean Squared Error:', mse)
        print('Root Mean Squared Error:', rmse)
        print('Multi-dimensional Heidke Skill Score:', hss)
    
    # !!! If any addition or edit here, also change hourly, diurnal and rate functions
    return mae, max_err, cov, mean_err, bias, p_corr, r2, s_corr, mse, rmse, hss # !!!
    # !!! If any addition or edit here, also change hourly, diurnal and rate functions




def generate_stats_file(outdir, ver_labels, ver, bin_edges):
    
    # Create list with headers for stats to go into:
    stats_table = [['product', 'mae', 'max_err', 'cov', 'mean_err', 'bias', 'p_corr', 'r2', 's_corr', 'mse', 'rmse', 'hss']]
    
    for i in range(11, len(ver[0,:])):
        # Make stats for one product vs. IMERG
        mae, max_err, cov, mean_err, bias, p_corr, r2, s_corr, mse, rmse, hss = RF.general_stats(
            truth = ver[:, 10], 
            test = ver[:, i], 
            bins=np.array([-2.5, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5]),
            PRINT=False)
        
        print(ver_labels[i])
        stats_table.append([ver_labels[i], mae, max_err, cov, mean_err, bias, p_corr, r2, s_corr, mse, rmse, hss])
        

    # Export the stats   
    with open(outdir+"stats.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(stats_table)
        
        
    return
    
    
    
    
    
    
def reverse_ciruclar(cos, sin):
    '''
    Returns fraction of circular angle. Can then multiply by hours of day (24) or days of year (365.25) to get real time.
    '''
    circ_frac = np.zeros(len(cos))
    if not len(sin) == len(cos):
        raise ValueError("The sin and cos arrays differ in length.")

    for i in range(0, len(circ_frac)):
        if sin[i] >= 0. and cos[i] > 0.:
            # first quadrant
            circ_frac[i] = np.arctan( sin[i] / cos[i] )*180/np.pi

        elif sin[i] > 0. and cos[i] <= 0.:
            # second quadrant
            circ_frac[i] = np.arctan( -cos[i] / sin[i] )*180/np.pi + 90

        elif sin[i] <= 0. and cos[i] < 0.:
            # third quadrant
            circ_frac[i] = np.arctan( sin[i] / cos[i] )*180/np.pi + 180

        elif sin[i] < 0. and cos[i] >= 0.:
            # fourth quadrant
            circ_frac[i] = np.arctan( cos[i] / -sin[i] )*180/np.pi + 270 
        
    return circ_frac / 360






def diurnal_stats(hour_of_day, truth, test, timestep=3): # timestep in unit hours
    '''
    Calculate statistics arrays for x number of hours over a day, controlled by `timestep` which must be a factor of 24.
    '''
    # Number of timesteps in the day based on user input (must be factor of 24 so timesteps is an int)
    timesteps = 24. / timestep
    if not timesteps-int(timesteps) == 0.:
        raise ValueError('timestep must be a factor of 24. ' + str(timestep) + ' is not a factor of 24.')
    timesteps = int(timesteps)
    
    # Initialise the empty stats arrays of length equal to timesteps
    mae_, max_err_, cov_, mean_err_, bias_, p_corr_, r2_, s_corr_, mse_, rmse_, hss_ = np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps)
    
    # Loop through each time period in the day
    for h in range(0, timesteps):
#         # Check that the hours are being pulled correctly
#         h_h = hour_of_day[ (hour_of_day >= h*timestep) & (hour_of_day < (h+1)*timestep) ]
#         plt.plot(h_h)
        
        # Pull out just the data valid 
        test_h = test[ (hour_of_day >= h*timestep) & (hour_of_day < (h+1)*timestep) ]
        truth_h = truth[ (hour_of_day >= h*timestep) & (hour_of_day < (h+1)*timestep) ]
        
        # Apply the stats to the current timestep
        mae_[h], max_err_[h], cov_[h], mean_err_[h], bias_[h], p_corr_[h], r2_[h], s_corr_[h], mse_[h], rmse_[h], hss_[h] = general_stats(
            truth = truth_h, test = test_h, 
            bins=np.array([-2.5, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5]),
            PRINT=False) # !!! check the names are even needed
    
#     plt.show() # Show the plot of all of the time checks in different colours
    
    return mae_, max_err_, cov_, mean_err_, bias_, p_corr_, r2_, s_corr_, mse_, rmse_, hss_






def annual_stats(month_of_year, truth, test, timestep=1): # Currently using 12 "months" of equal length, so timestep is in unit months
    '''
    Calculate statistics arrays for x number of months over a year, controlled by `timestep` which must be a factor of 12.
    '''
    # Number of timesteps in the day based on user input (must be factor of 12 so timesteps is an int)
    timesteps = 12. / timestep
    if not timesteps-int(timesteps) == 0.:
        raise ValueError('timestep must be a factor of 12. ' + str(timestep) + ' is not a factor of 12.')
    timesteps = int(timesteps)
    
    # Initialise the empty stats arrays of length equal to timesteps
    mae_, max_err_, cov_, mean_err_, bias_, p_corr_, r2_, s_corr_, mse_, rmse_, hss_ = np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps)
    
    # Loop through each time period in the day
    for m in range(0, timesteps):
#         # Check that the hours are being pulled correctly
#         m_m = month_of_year[ (month_of_year >= m*timestep) & (month_of_year < (m+1)*timestep) ]
#         plt.plot(m_m)
        
        # Pull out just the data valid 
        test_m = test[ (month_of_year >= m*timestep) & (month_of_year < (m+1)*timestep) ]
        truth_m = truth[ (month_of_year >= m*timestep) & (month_of_year < (m+1)*timestep) ]
        
        # Apply the stats to the current timestep
        mae_[m], max_err_[m], cov_[m], mean_err_[m], bias_[m], p_corr_[m], r2_[m], s_corr_[m], mse_[m], rmse_[m], hss_[m] = general_stats(
            truth = truth_m, test = test_m, 
            bins=np.array([-2.5, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5]),
            PRINT=False) # !!! check the names are even needed
    
#     plt.show() # Show the plot of all of the time checks in different colours
    
    return mae_, max_err_, cov_, mean_err_, bias_, p_corr_, r2_, s_corr_, mse_, rmse_, hss_





def time_stats(ver, prod_i, ver_markers, outdir, bin_edges):
    '''
    Create all the hourly stats for each product.
    Simply add another product to the dict below and its ver_array column number, and it will be included.
    Then plot the data as multi-plot figures.
    '''
    # Turn circular cos and sin dirunal and annual values into fractional hours of the day, and fractional months of the year.
    hour_of_day = 24 * RF.reverse_ciruclar(cos=ver[:, 3], sin=ver[:, 4])
    
    # Assumes a month is 366/12 = 30.5 days long. This is acceptable as the other errors in the whole process are larger... 
    # climate models use this protocol (assume 30 day months, 360 day years for coding simplicity).
    month_of_year = 12 * RF.reverse_ciruclar(cos=ver[:, 1], sin=ver[:, 2])
    

    for p in prod_i: # Diurnal stats
        exec (p+"_mae_h,"+p+"_max_err_h,"+p+"_cov_h,"+p+"_mean_err_h,"+p+"_bias_h,"+p+"_p_corr_h,"+p+"_r2_h,"+p+"_s_corr_h,"+p+"_mse_h,"+p+"_rmse_h,"+p+"_hss_h = RF.diurnal_stats(hour_of_day=hour_of_day, truth=ver[:, 7], test=ver[:, "+str(prod_i[p])+"], timestep=3)")

    for p in prod_i: # Annual stats
        exec (p+"_mae_m,"+p+"_max_err_m,"+p+"_cov_m,"+p+"_mean_err_m,"+p+"_bias_m,"+p+"_p_corr_m,"+p+"_r2_m,"+p+"_s_corr_m,"+p+"_mse_m,"+p+"_rmse_m,"+p+"_hss_m = RF.annual_stats(month_of_year=month_of_year, truth=ver[:, 7], test=ver[:, "+str(prod_i[p])+"], timestep=1)")
        
    
    # Loop through diurnal hour (h), annual month (m)
    for kind in ['h','m']:
        fig = plt.figure(figsize=(10,10))
        
        gs1 = GridSpec(4, 3, wspace=0.3, hspace=0.7)
        ax1 = fig.add_subplot(gs1[0:1, 0:1]) # mae
        ax2 = fig.add_subplot(gs1[0:1, 1:2]) # max_err
        ax3 = fig.add_subplot(gs1[0:1, 2:3]) # cov
        ax4 = fig.add_subplot(gs1[1:2, 0:1]) # mean_err
        ax5 = fig.add_subplot(gs1[1:2, 1:2]) # bias
        ax6 = fig.add_subplot(gs1[1:2, 2:3]) # p_corr
        ax7 = fig.add_subplot(gs1[2:3, 0:1]) # r2
        ax8 = fig.add_subplot(gs1[2:3, 1:2]) # s_corr
        ax9 = fig.add_subplot(gs1[2:3, 2:3]) # mse
        ax10 = fig.add_subplot(gs1[3:4, 0:1]) # rmse
        ax11 = fig.add_subplot(gs1[3:4, 1:2]) # hss

        ax_stat = {1: 'mae_',
                   2: 'max_err_',
                   3: 'cov_',
                   4: 'mean_err_',
                   5: 'bias_',
                   6: 'p_corr_',
                   7: 'r2_',
                   8: 's_corr_',
                   9: 'mse_',
                  10: 'rmse_',
                  11: 'hss_'}

        # Plot the data
        for i in range(1,12):
            if i == 1:
                for prod in prod_i:
                    exec("ax"+str(i)+".plot("+prod+"_"+ax_stat[i]+kind+", marker='"+ver_markers[prod_i[prod]]+"', label='"+prod+"')")
            else:
                for prod in prod_i:
                    exec("ax"+str(i)+".plot("+prod+"_"+ax_stat[i]+kind+", marker='"+ver_markers[prod_i[prod]]+"')")

        # Set titles
        ax1.set_title("Mean Absolute Error")
        ax2.set_title("Maximum Error")
        ax3.set_title("Covariance")
        ax4.set_title("Mean Error")
        ax5.set_title("Frequency Bias")
        ax6.set_title("Pearson's Correlation")
        ax7.set_title("Coefficient of Determination")
        ax8.set_title("Spearman's Correlation")
        ax9.set_title("Mean Squared Error")
        ax10.set_title("Root Mean Squared Error")
        ax11.set_title("Heidke Skill Score")

        # Other plot settings
        if kind == 'h':
            ticklabels = ['00-03Z','03–06Z','06–09Z','09–12Z','12-15Z','15–18Z','18–21Z','21–00Z']
            ticks = np.array([0,1,2,3,4,5,6,7])
        elif kind == 'm':
            ticklabels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            ticks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])
        elif kind == 'r':
            ticklabels = ['0–5','5-10','10-15','finish the rest later...']
            ticks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])
        else:
            print("kind is not diurnal, annual, or rate")

        for i in range(1,12):
            exec("ax"+str(i)+".set_xticks(ticks)")
            exec("ax"+str(i)+".set_xticklabels(ticklabels, rotation=55.)")

            exec("ax"+str(i)+".grid()")

        # Legend
        ax1.legend(loc=2, bbox_to_anchor=(2.57, -4.05), fontsize=12, fancybox=True, shadow=True)

        # Save and display
        filename = {'h': '3hourly', 'm': 'monthly'}
        fig.savefig(outdir+filename[kind]+'_stats.pdf', bbox_inches="tight", dpi=250)
        fig.savefig(outdir+filename[kind]+'_stats.png', bbox_inches="tight", dpi=250)
        
        #plt.show()
        plt.close()
        
    return
        
    
        

def map_stats(truth, test,
              bin_edges,
              lats, lons,
              d_lats, d_lons,
              stat_px,
              ):
    '''
    Loop through each 10x10 pixel array in the domain, find all datapoints within that, then apply all stats to those data.
    Return 2D arrays (73 x 72) of statistics ready for plotting.
    ''' 
    # Set up empty map array dimensions (just makes code more readable)
    y=int((len(d_lats))/stat_px)
    x=int((len(d_lons))/stat_px)
    if not y-int(y)==0. and x-int(x)==0.:
        raise ValueError('stat_px must be a factor of domain dimensions. Either ' + str(stat_px) + ' is not a factor of ' + str(len(d_lats)) + ' or ' + str(stat_px) + ' is not a factor of ' + str(len(d_lons)) + '.')
    y=int(y)
    x=int(x)
    # Initialise the empty stats arrays of length equal to timesteps
    mae_, max_err_, cov_, mean_err_, bias_, p_corr_, r2_, s_corr_, mse_, rmse_, hss_ = np.ma.zeros((y,x)), np.ma.zeros((y,x)), np.ma.zeros((y,x)), np.ma.zeros((y,x)), np.ma.zeros((y,x)), np.ma.zeros((y,x)), np.ma.zeros((y,x)), np.ma.zeros((y,x)), np.ma.zeros((y,x)), np.ma.zeros((y,x)), np.ma.zeros((y,x))
    # track number of samples used for each stats calculation
    num_samples = np.ma.zeros((y,x))
    
    # Loop through each region in the domain
    for lat in range(0, y):
        for lon in range(0, x):
            # each stat pixel
            # find all the datapoints in ver within this region (pull out just the data valid)
            test_r = test[ (lats > d_lats[stat_px*(lat+1)]) & (lats < d_lats[stat_px*lat]) & (lons > d_lons[stat_px*lon]) & (lons < d_lons[stat_px*(lon+1)])]
            truth_r = truth[ (lats > d_lats[stat_px*(lat+1)]) & (lats < d_lats[stat_px*lat]) & (lons > d_lons[stat_px*lon]) & (lons < d_lons[stat_px*(lon+1)])]
            
            # Track how many samples used for each pixel
            num_samples[lat,lon] = test_r.count()
            
            if lon == 0:
                print (lat+1, 'out of 73. Num px in region:', len(test_r), 'out of', len(truth), 'or', round(len(test_r)/len(truth)*720*730, 1), '% of one whole map.')
            
            # Apply the stats to the current region, save to right matrix element
            mae_[lat,lon], max_err_[lat,lon], cov_[lat,lon], mean_err_[lat,lon], bias_[lat,lon], p_corr_[lat,lon], r2_[lat,lon], s_corr_[lat,lon], mse_[lat,lon], rmse_[lat,lon], hss_[lat,lon] = general_stats(
                truth = truth_r, test = test_r, 
                bins=bin_edges,
                PRINT=False) # !!! check the names are even needed
            
        
    return num_samples, mae_, max_err_, cov_, mean_err_, bias_, p_corr_, r2_, s_corr_, mse_, rmse_, hss_
# Might be faster to go from the full 1D array into a 3D (lat, lon, event) stack and then group lat/lon into regions 10x10 px and then do stats on each?? Vectorised.






def plot_map(ax, data):
    '''
    Plot single map inside a larger gridspec.
    ax: which subplot to draw in
    data: which data to display (will be forced to fit domain)
    '''
    # Map Features
    ax.add_feature(cfeature.COASTLINE, edgecolor=(0,0,0,1), linewidth=1., zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1., zorder=3)
    ax.add_feature(cfeature.LAKES, edgecolor=(0,0,0,0), linewidth=0.25, facecolor=(1,1,1,0.4), zorder=1)
    ax.add_feature(cfeature.LAKES, edgecolor=(0,0,0,1), linewidth=0.25, facecolor=(0,0,0,0), zorder=3)

    # Precipitation Data
    cmap = matplotlib.cm.viridis
    cmap.set_under((0,0,0,0))
    img = ax.imshow(data, extent=[-20, 52, -35, 38], 
                       vmin=0.01, interpolation='nearest',
                       cmap=cmap, alpha=1., zorder=2)
    # Colorbar
    plt.colorbar(img, ax=ax, shrink=0.6, pad = 0.05, cmap=cmap, extend='max')
    
    # Lat/Lon ticks
    ax.set_xticks(np.array([-20, -10, 0, 10, 20, 30, 40, 50]))
    ax.set_yticks(np.array([-30, -20, -10, 0, 10, 20, 30]))
    ax.grid()
    
    return
    
    
    


def plot_map_stats(prod_i, statlabels, outdir):
    '''
    Plot a large selection of maps with statistics of products across the Pan-Africa domain.
    '''
    # Loop through each product
    for p in prod_i:
        fig = plt.figure(figsize=(16,20))
        
        gs1 = GridSpec(4, 3, wspace=0.15, hspace=0.)
        ax1 = fig.add_subplot(gs1[0:1, 0:1], projection=ccrs.PlateCarree()) # num_samples
        ax2 = fig.add_subplot(gs1[0:1, 1:2], projection=ccrs.PlateCarree()) # mae
        ax3 = fig.add_subplot(gs1[0:1, 2:3], projection=ccrs.PlateCarree()) # max_err
        ax4 = fig.add_subplot(gs1[1:2, 0:1], projection=ccrs.PlateCarree()) # bias
        ax5 = fig.add_subplot(gs1[1:2, 1:2], projection=ccrs.PlateCarree()) # r2
        ax6 = fig.add_subplot(gs1[1:2, 2:3], projection=ccrs.PlateCarree()) # rmse
        ax7 = fig.add_subplot(gs1[2:3, 0:1], projection=ccrs.PlateCarree()) # hss
        ax8 = fig.add_subplot(gs1[2:3, 1:2], projection=ccrs.PlateCarree()) # precip_data
        ax9 = fig.add_subplot(gs1[2:3, 2:3], projection=ccrs.PlateCarree()) # num_samples (other method)
#         ax10 = fig.add_subplot(gs1[3:4, 0:1], projection=ccrs.PlateCarree()) # mse
#         ax11 = fig.add_subplot(gs1[3:4, 1:2], projection=ccrs.PlateCarree()) # rmse
#         ax12 = fig.add_subplot(gs1[3:4, 2:3], projection=ccrs.PlateCarree()) # hss
        
        
        # Plot the data
        for i in range(0, len(statlabels)):
            exec(p + statlabels[i] + "=np.load('"+outdir+'/map_stats/'+p+statlabels[i]+"', allow_pickle=True)")
            exec("RF.plot_map(ax=ax"+str(i+1)+", data="+ p + statlabels[i] +")")
        
        
        # Set titles
        ax1.set_title("Number of Samples")
        ax2.set_title("Mean Absolute Error")
        ax3.set_title("Maximum Error")
        #ax4.set_title("Covariance")
        #ax5.set_title("Mean Error")
        ax4.set_title("Frequency Bias")
        #ax7.set_title("Pearson's Correlation")
        ax5.set_title("Coefficient of Determination")
        #ax9.set_title("Spearman's Correlation")
        #ax6.set_title("Mean Squared Error")
        ax6.set_title("Root Mean Squared Error")
        ax7.set_title("Heidke Skill Score")
        ax8.set_title("Total Rainfall")
        ax9.set_title("No. of Samples (should be same as ax1)")
        

        # Save and display
        fig.savefig(outdir+'map_stats_'+p+'.pdf', bbox_inches="tight", dpi=250)
        fig.savefig(outdir+'map_stats_'+p+'.png', bbox_inches="tight", dpi=250)
        #plt.show()
        plt.close()
        
    return



        

def generate_map(rain_col, lat_data, lon_data):
    '''
    Takes a rain column at a time, returns a map array where each pixel is the sum of rain there divided by the number of datapoints there. 
    '''
    # Make array of x720 * y730
    precip_data = np.zeros((730 ,720))
    precip_num = np.zeros((730 ,720))

    # Set up dicts which convert lat into row and lon into column of array
    lats = np.linspace(37.95, -34.95, 730)
    rows = np.linspace(0, 729, 730)
    lat_2_row = dict(zip(lats.tolist(), rows.tolist()))

    lons = np.linspace(-19.95, 51.95, 720)
    cols = np.linspace(0, 719, 720)
    lon_2_col = dict(zip(lons.tolist(), cols.tolist()))
    
    counter = 0
    # Use lat and lon with dicts to add each rain value to map array
    for i in range(0, len(rain_col)):
        if not np.ma.is_masked(rain_col[i]):
            precip_data[ int(lat_2_row[ lat_data[i] ] ) , int(lon_2_col[ lon_data[i] ]) ] += rain_col[i]
            precip_num[ int(lat_2_row[ lat_data[i] ] ) , int(lon_2_col[ lon_data[i] ]) ] += 1
            counter += 1
    
    return precip_data, precip_num



# RF.grid(
#     outdir=RF_parameters['verdir'], 
#     ver=ver, 
#     ver_labels=ver_labels,
#     bin_edges=np.array(RF_parameters["bin_edges"]).astype(np.float64), 
#     bin_labels=bin_labels,
#     truthname='IMERG'

def grid(outdir, ver, ver_labels, bin_edges, bin_labels, truthname): ## REMOVED truth and test, need to define inside here in loop starting at [:, 11] to len([0,:])
    '''
    Function for plotting a grid of two inputs, 
    plus a histogram below of two inputs, 
    plus a diagonal normalised hit rate bar chart above.
    '''
    # Loop through every product under examination
    for p in range(11, len(ver[0,:])):
        
        fig = plt.figure(figsize=(10,10))

        gs1 = GridSpec(21, 21, wspace=0.5, hspace=0.1)
        ax1 = fig.add_subplot(gs1[0:3, 6:18]) # top hit rate bar chart
        # ax2 = fig.add_subplot(gs1[1:5, 0:2]) # y-axis histogram
        ax3 = fig.add_subplot(gs1[3:15, 6:18]) # grid
        ax4 = fig.add_subplot(gs1[5:13, 18]) # colorbar
        ax5 = fig.add_subplot(gs1[17:21, 6:18]) # x-axis histogram

        # Create the 2D histogram using bins provided
        heatmap, xedges, yedges = np.histogram2d(ver[:, 10], ver[:, p], bins=bin_edges)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # Plot the hit rate for each column
        hit_rate = np.diagonal(heatmap) / (np.sum(heatmap, axis=1)) * 100
        steps = np.linspace(0, 200, len(bin_edges))
        width = 0.9*(steps[1]-steps[0])
        midbins = (steps[1:] + steps[:-1]) / 2
        ax1.bar(midbins, hit_rate, width=width, color='grey', zorder=1)
        ax1.set_yscale('log')
        ax1.set_ylim([0.05, 500])
        ax1.set_ylabel("Hit Rate (%)")  
        ax1.set_yticks([0.1, 1, 10, 100])
        ax1.set_yticklabels([0.1, 1, 10, 100])
        ax1.set_xticks(midbins)
        ax1.set_xlim([xedges[0], xedges[-1]])
        ax1.set_title(ver_labels[p]+' vs. '+truthname)
        for i in range(0, len(midbins)):
            ax1.text(x=midbins[i], y=200, s=str(hit_rate[i])[:3], fontsize='small', fontweight='bold', ha='center', va='center', color='grey')
        ax1.grid(which='both', axis='y', zorder=10)


        # Plot the histogram
        cs = ax3.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect="auto")

        # Plot diagonal box borders (since these are correct diagnoses)
        for i in range(0, len(steps)-2):
            ax3.plot( steps[i:i+3], np.repeat(steps[i+1], 3), 'k-', linewidth=1., zorder=2.)
            ax3.plot( np.repeat(steps[i+1], 3), steps[i:i+3], 'k-', linewidth=1., zorder=2.)

        # Other plot settings
        ax3.set_xticks(midbins)
        ax3.set_xticklabels(list(bin_labels.values())[1:-1], rotation=35.)
        ax3.set_xlabel(truthname+" Rainfall (mm h$^{-1}$)", labelpad=-5)
        ax3.set_yticks(midbins)
        ax3.set_yticklabels(list(bin_labels.values())[1:-1])
        ax3.set_ylabel(ver_labels[p]+" Rainfall (mm h$^{-1}$)", labelpad=-5)


        # Plot the colorbar in the right middle box
        ax4.plot(0.5,0.5, 'r-', marker='o')
        cbar = fig.colorbar(cs, shrink=0.8, panchor=(-50., 0.5), anchor=(-50., 0.5), label="Log Frequency", cax=ax4)


        # Plot the distribution of the truth array
        truthdist = (np.sum(heatmap, axis=1))
        testdist = (np.sum(heatmap, axis=0))
        ax5.bar(midbins - width/4, truthdist, width=width/2, zorder=1, label=truthname)
        ax5.bar(midbins + width/4, testdist, width=width/2, zorder=1, label=ver_labels[p])
        ax5.set_yscale('log')
        ax5.set_ylabel("Frequency")
        ax5.set_xlim([xedges[0], xedges[-1]])
        ax5.set_xticks(midbins)
        ax5.set_xticklabels(list(bin_labels.values())[1:-1], rotation=35.)
        ax5.set_xlabel("Rainfall (mm h$^{-1}$)", labelpad=-5)
        ax5.grid(which='major', axis='y', zorder=10)
        ax5.legend()


        # Save and display
        fig.savefig(outdir+'grid_'+ver_labels[p]+'_vs_'+truthname+'.pdf', bbox_inches="tight", dpi=250)
        fig.savefig(outdir+'grid_'+ver_labels[p]+'_vs_'+truthname+'.png', bbox_inches="tight", dpi=250)
        #plt.show()
        plt.close()
    
    return

    

    
def labelAtEdge(levels, cs, ax, fmt, side='all', pad=0.005, **kwargs):
    '''Label contour lines at the edge of plot
    Args:
        levels (1d array): contour levels.
        cs (QuadContourSet obj): the return value of contour() function.
        ax (Axes obj): matplotlib axis.
        fmt (str): formating string to format the label texts. E.g. '%.2f' for
            floating point values with 2 demical places.
    Keyword Args:
        side (str): on which side of the plot intersections of contour lines
            and plot boundary are checked. Could be: 'left', 'right', 'top',
            'bottom' or 'all'. E.g. 'left' means only intersections of contour
            lines and left plot boundary will be labeled. 'all' means all 4
            edges.
        pad (float): padding to add between plot edge and label text.
        **kwargs: additional keyword arguments to control texts. E.g. fontsize,
            color.
    '''
    from matplotlib.transforms import Bbox
    collections = cs.collections
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bbox = Bbox.from_bounds(xlim[0], ylim[0], xlim[1]-xlim[0], ylim[1]-ylim[0])
    eps = 1e-5  # error for checking boundary intersection
    # -----------Loop through contour levels-----------
    for ii, lii in enumerate(levels):
        cii = collections[ii]  # contours for level lii
        pathsii = cii.get_paths()  # the Paths for these contours
        if len(pathsii) == 0:
            continue
        for pjj in pathsii:
            # check first whether the contour intersects the axis boundary
            if not pjj.intersects_bbox(bbox, False):  # False significant here
                continue
            xjj = pjj.vertices[:, 0]
            yjj = pjj.vertices[:, 1]
            # intersection with the left edge
            if side in ['left', 'all']:
                inter_idx = np.where(abs(xjj-xlim[0]) <= eps)[0]
                for kk in inter_idx:
                    inter_x = xjj[kk]
                    inter_y = yjj[kk]
                    ax.text(inter_x-pad, inter_y, fmt % lii,
                            ha='right',
                            va='center',
                            **kwargs)
            # intersection with the right edge
            if side in ['right', 'all']:
                inter_idx = np.where(abs(xjj-xlim[1]) <= eps)[0]
                for kk in inter_idx:
                    inter_x = xjj[kk]
                    inter_y = yjj[kk]
                    ax.text(inter_x+pad, inter_y, fmt % lii,
                            ha='left',
                            va='center',
                            **kwargs)
            # intersection with the bottom edge
            if side in ['bottom', 'all']:
                inter_idx = np.where(abs(yjj-ylim[0]) <= eps)[0]
                for kk in inter_idx:
                    inter_x = xjj[kk]
                    inter_y = yjj[kk]
                    ax.text(inter_x-pad, inter_y, fmt % lii,
                            ha='center',
                            va='top',
                            **kwargs)
            # intersection with the top edge
            if side in ['top', 'all']:
                inter_idx = np.where(abs(yjj-ylim[-1]) <= eps)[0]
                for kk in inter_idx:
                    inter_x = xjj[kk]
                    inter_y = yjj[kk]
                    ax.text(inter_x+pad, inter_y, fmt % lii,
                            ha='center',
                            va='bottom',
                            **kwargs)
    return




def performance_stats(ver, truth_i, ver_labels, ver_markers, ver_colors, bin_edges):
    '''
    Produce the performance dictionary, looping through all columns 11 onwards
    '''
    performance={}
    for i in range(truth_i+1, len(ver[0])):
        # Create the 2D histogram using bins provided
        heatmap, xedges, yedges = np.histogram2d(ver[:, truth_i], ver[:, i], bins=bin_edges)
        
        # Calculate the hit rate for each column
        POD = np.diagonal(heatmap) / (np.sum(heatmap, axis=1))
        
        # Calculate the FAR and then SR for each column
        FAR = ((np.sum(heatmap, axis=0)) - np.diagonal(heatmap)) / (np.sum(heatmap, axis=0))
        SR = 1 - FAR
        
        # print(SR, FAR)
        
        # Add the scores of the product to the overall dictionary
        performance.update({ver_labels[i]: 
                       {'pod': POD,
                        'sr': SR,
                        'marker': ver_markers[i],
                        'color': ver_colors[i]
                       }})
    
    return performance



                                         
def performance_diagram(outdir, p, bin_labels, description=''):  
    '''
    Function to create a performance diagram and plot as many products as are provided
    From: https://gist.github.com/djgagne/64516e3ea268ec31fb34#file-distributedroc-py 
    '''
    # Create the figure and ax
    figure = plt.figure(figsize=(10, 10), dpi=100)
    ax1 = figure.add_subplot(1, 1, 1)

    # Set up some basic parameters
    ticks=np.arange(0, 1.01, 0.1)
    csi_levels=np.arange(0.1, 1.1, 0.1)  
    b_levels=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.66, 2.5, 5])#([0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3.3, 10])

    # Set up the basic performance diagram lines
    grid_ticks = np.arange(0.0, 1.01, 0.01)
    sr_g, pod_g = np.meshgrid(grid_ticks, grid_ticks)
    bias = pod_g / sr_g
    csi = 1.0 / (1.0 / sr_g + 1.0 / pod_g - 1.0)

    # CSI lines
    #csi_contour = ax1.contourf(sr_g, pod_g, csi, levels=csi_levels, extend="max", cmap='Blues', alpha=0.5) # filled contour
    csi_contour = ax1.contour(sr_g, pod_g, csi, levels=csi_levels, extend="max", colors=[(0,0,0,0.2)])
    csi_contour.collections[0].set_label('Critical Success Index') # to make CSI show on plot legend
    RF.labelAtEdge(levels=csi_levels, cs=csi_contour, ax=ax1, fmt='%.1f', side='top', pad=0.005, rotation=0., color=(0,0,0,0.5), fontstyle='oblique')
    RF.labelAtEdge(levels=csi_levels, cs=csi_contour, ax=ax1, fmt='%.1f', side='right', pad=0.005, rotation=0., color=(0,0,0,0.5), fontstyle='oblique')

    # Bias lines
    b_contour = ax1.contour(sr_g, pod_g, bias, levels=b_levels, colors=[(0,0,0,0.2)], linestyles="dashed")
    b_contour.collections[0].set_label('Frequency Bias') # to make CSI show on plot legend
    # ax1.clabel(b_contour, fmt="%.1f", manual=[(0.11,0.89), (0.2, 0.8), (0.33, 0.67), (0.4, 0.6), (0.5, 0.5),
    #                                           (0.56, 0.44), (0.62, 0.38), (0.71, 0.29), (0.83,0.17)]) # diagonal bias labels
    ax1.clabel(b_contour, fmt="%.2f", manual=[(0.08,0.69), (0.20,0.67), (0.32,0.63), (0.45,0.54), (0.5,0.5),
                                              (0.55,0.44), (0.59,0.38), (0.64,0.28), (0.68,0.13)]) # mid-curve bias labels

    # Plot precipitation product statistics
    ABC = list(string.ascii_uppercase)
    for i in p:
        ax1.plot(p[i]['sr'], p[i]['pod'], marker=p[i]['marker'], color=p[i]['color'], label=i, linewidth=0., markersize=15., alpha=0.8)
        for j in range(0,len(p[i]['pod'])):
            ax1.text(p[i]['sr'][j], p[i]['pod'][j], ABC[j], ha='center', va='center', fontsize=8.)
    
    # Key for precipitation rates
    legend_left = 'Key:\n'
    legend_right = '\n'
    for j in range(0,len(p[i]['pod'])):
        legend_left = legend_left + ABC[j] + ':\n' 
        legend_right= legend_right + bin_labels[j+1] + ' mm h$^{-1}$\n'
    ax1.text(1.07, 0.00, legend_left, ha='left', va='bottom', fontsize=12)
    ax1.text(1.12, 0.00, legend_right, ha='left', va='bottom', fontsize=12)

    # Other plot parameters
    ax1.set_xlabel(xlabel="Success Ratio (1-FAR)", fontsize=14)
    ax1.set_ylabel(ylabel="Probability of Detection", fontsize=14)
    ax1.set_title("Performance Diagram\n"+description+"\n", fontsize=14, fontweight="bold")
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.grid(alpha=0.1, which='both')
    ax1.legend(loc=2, bbox_to_anchor=(1.04, 1.0), fontsize=12, fancybox=True, shadow=True)

    # Save and display the figure
    figure.savefig(outdir+'performance_diagram'+description+'.pdf', dpi=250, bbox_inches="tight")
    figure.savefig(outdir+'performance_diagram'+description+'.png', dpi=250, bbox_inches="tight")
    #plt.show()
    plt.close()
    
    return
    


def performance_diagram_zoomed(outdir, p, bin_labels, description=''):
    '''
    Function to create a performance diagram and plot as many products as are provided
    Zoomed in version for very low skill products
    From: https://gist.github.com/djgagne/64516e3ea268ec31fb34#file-distributedroc-py 
    '''
    # Create the figure and ax
    figure = plt.figure(figsize=(10, 10), dpi=100)
    ax1 = figure.add_subplot(1, 1, 1)

    # Set up some basic parameters
    ticks=np.arange(0, 0.11, 0.01)
    csi_levels=np.arange(0.01, 0.1, 0.01)
    b_levels=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.66, 2.5, 5])#([0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3.3, 10])

    # Set up the basic performance diagram lines
    grid_ticks = np.arange(0, 0.101, 0.001)
    sr_g, pod_g = np.meshgrid(grid_ticks, grid_ticks)
    bias = pod_g / sr_g
    csi = 1.0 / (1.0 / sr_g + 1.0 / pod_g - 1.0)

    # CSI lines
    #csi_contour = ax1.contourf(sr_g, pod_g, csi, levels=csi_levels, extend="max", cmap='Blues', alpha=0.5) # filled contour
    csi_contour = ax1.contour(sr_g, pod_g, csi, levels=csi_levels, extend="max", colors=[(0,0,0,0.2)])
    csi_contour.collections[0].set_label('Critical Success Index') # to make CSI show on plot legend
    labelAtEdge(levels=csi_levels, cs=csi_contour, ax=ax1, fmt='%.2f', side='top', pad=0.0005, rotation=0., color=(0,0,0,0.5), fontstyle='oblique')
    labelAtEdge(levels=csi_levels, cs=csi_contour, ax=ax1, fmt='%.2f', side='right', pad=0.0005, rotation=0., color=(0,0,0,0.5), fontstyle='oblique')

    # Bias lines
    b_contour = ax1.contour(sr_g, pod_g, bias, levels=b_levels, colors=[(0,0,0,0.2)], linestyles="dashed")
    b_contour.collections[0].set_label('Frequency Bias') # to make CSI show on plot legend
    # ax1.clabel(b_contour, fmt="%.1f", manual=[(0.11,0.89), (0.2, 0.8), (0.33, 0.67), (0.4, 0.6), (0.5, 0.5),
    #                                           (0.56, 0.44), (0.62, 0.38), (0.71, 0.29), (0.83,0.17)]) # diagonal bias labels
    ax1.clabel(b_contour, fmt="%.2f", manual=[(0.008,0.069), (0.020,0.067), (0.032,0.063), (0.045,0.054), (0.05,0.05),
                                              (0.055,0.044), (0.059,0.038), (0.064,0.028), (0.068,0.013)]) # mid-curve bias labels

    # Plot precipitation product statistics
    ABC = list(string.ascii_uppercase)
    for i in p:
        counter = 0
        for j in range(0,len(p[i]['pod'])):
            if p[i]['sr'][j] <= 0.1 and p[i]['pod'][j] <= 0.1:
                if counter == 0: # only label the product once
                    ax1.plot(p[i]['sr'][j], p[i]['pod'][j], marker=p[i]['marker'], color=p[i]['color'], label=i, linewidth=0., markersize=15., alpha=0.8)
                    counter = 1
                else:
                    ax1.plot(p[i]['sr'][j], p[i]['pod'][j], marker=p[i]['marker'], color=p[i]['color'], linewidth=0., markersize=15., alpha=0.8)
                # Plot the text describing the precipitation rate    
                ax1.text(p[i]['sr'][j], p[i]['pod'][j], ABC[j], ha='center', va='center', fontsize=8.)
    
    # Key for precipitation rates
    legend_left = 'Key:\n'
    legend_right = '\n'
    for j in range(0,len(p[i]['pod'])):
        legend_left = legend_left + ABC[j] + ':\n' 
        legend_right= legend_right + bin_labels[j+1] + ' mm h$^{-1}$\n'
    ax1.text(0.107, 0.00, legend_left, ha='left', va='bottom', fontsize=12)
    ax1.text(0.112, 0.00, legend_right, ha='left', va='bottom', fontsize=12)

    # Other plot parameters
    ax1.set_xlabel(xlabel="Success Ratio (1-FAR)", fontsize=14)
    ax1.set_ylabel(ylabel="Probability of Detection", fontsize=14)
    ax1.set_title("Performance Diagram Zoomed\n"+description+"\n", fontsize=14, fontweight="bold")
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.grid(alpha=0.1, which='both')
    ax1.legend(loc=2, bbox_to_anchor=(1.04, 1.0), fontsize=12, fancybox=True, shadow=True)

    # Zooms in to lower right corner
    ax1.set_ylim(0., 0.1)
    ax1.set_xlim(0., 0.1)

    # Save and display the figure
    figure.savefig(outdir+'performance_diagram_zoomed'+description+'.pdf', dpi=250, bbox_inches="tight")
    figure.savefig(outdir+'performance_diagram_zoomed'+description+'.png', dpi=250, bbox_inches="tight")
    #plt.show()
    plt.close()
    
    return











'''
RF_dict   is the dictionary containing all the models (importances within) and feature labels
clf       is the model itself, the importances are embedded within.
flabels   is a list of the feature labels, not contained in the model. Must match original model order

writes a csv with the features labelled and their importance, ranked by importance.
plots a vertical bar chart to show their importance against random, if incldued. So works for both randomisedCVsearch and final verification.
'''
def plot_importances(RF_dict, outdir):
    for RF in RF_dict:

        # Extract model and flabels from RF_dict
        clf, flabels = RF_dict[RF]['model'], RF_dict[RF]['labels']
        
        # Get numerical feature importances
        importances = list(clf.feature_importances_)

        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(flabels, importances)]

        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

        # Export the features and importances as csv
        with open(outdir+"feature_importances_"+str(RF)+".csv","w+") as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(feature_importances)


        # Set the style
        plt.style.use('fivethirtyeight')

        # list of x locations for plotting
        x_values = list(range(len(importances)))

        # Make a bar chart (differs if random is included) 
        if 'random' in flabels:
            # Work out where it is
            rand_index = flabels.index('random')
            print(rand_index)
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
        plt.axis([None, None, -0.01, None])

        # Axis labels and title
        plt.ylabel('Importance'); plt.xlabel('Feature'); plt.title('Feature Importances');
        plt.savefig(outdir+'feature_importances_'+str(RF)+'.png', bbox_inches="tight", dpi=250)
        
        plt.close()
    


    
    
def make_thresholds(RF_dict):
    '''
    Make the threshold dictionary
    Each feature has a value for the threshold used for the decision at that node.
    The weight of that threshold is determined by the fraction of samples at that node vs. the start of the decision tree.
    '''
    thresholds_dict = {}
    for RF in RF_dict:
        feature_list, clf = RF_dict[RF]['labels'], RF_dict[RF]['model']
        
        thresholds = {}
        for i in range(0, len(feature_list)):
            thresholds.update({i: {'value': [], 'weight': []} })


        t_nodes = 0
        for i in range(0, len(clf.estimators_)):
            # for each decision tree produced (num_estimators)
            for n in range(0, clf.estimators_[i].tree_.node_count):
                t_nodes += 1

                #print (clf.estimators_[i].tree_.feature[n])

                if not clf.estimators_[i].tree_.feature[n] == -2:

                    thresholds[ clf.estimators_[i].tree_.feature[n] ]['value'].append( clf.estimators_[i].tree_.threshold[n] )

                    thresholds[ clf.estimators_[i].tree_.feature[n] ]['weight'].append( clf.estimators_[i].tree_.n_node_samples[n] / clf.estimators_[i].tree_.n_node_samples[0])
                    
        thresholds_dict.update({RF: {'thresholds': thresholds,
                                   't_nodes': t_nodes,
                                   'feature_list': feature_list}})
            
    return thresholds_dict
    







def threshold_hist(ax, data, minn, maxx, rangee, xlabel):
    ax.hist(data, bins=20, zorder=2)
    ax.yaxis.tick_right()
    ax.set_xlim([minn - rangee*0.0, maxx + rangee*0.0])
    ax.set_xlabel(xlabel)
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(ax.get_yticks()[1]/5))
    ax.grid(which='both', zorder=-1)
    

def threshold_heatmap(ax, thresholds, i, minn, maxx, rangee, cm):
    h, xedges, yedges, image = ax.hist2d(np.array(thresholds[i]['value']), np.array(thresholds[i]['weight']), bins=[40,20],  range=[[minn, maxx],[0,1]],  cmin=1.0, cmap=cm, zorder=2)
    ax.set_ylim([0.0,1.0])
    ax.set_xlim([minn - rangee*0.0, maxx + rangee*0.0])
    ax.set_xticklabels([''])
    ax.set_ylabel('')
    ax.set_facecolor((0,0,0,0.2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.grid(which='both', zorder=0)
    
    return np.nanmax(h)


def threshold_colorbar(ax, max_h, cm):
    a = np.array([[1.0, max_h]])
    img = ax.imshow(a, cmap=cm)
    ax.set_visible(False)
    cb = plt.colorbar(mappable=img, orientation="vertical", ax=ax, shrink=0.7, aspect=10)
    cb.ax.minorticks_on()
    


def plot_feature_thresholds(RF_dict, thresholds_dict, feature_units, outdir): # was thresholds, feature_list, outdir !!!!!
    '''
    Take the thresholds for each feature and plot heatmaps plus histogram below.
    Make adaptable to each scenario where a model has few or many features, or the order of importance differs.
    Order it by importance.
    '''
    for RF in thresholds_dict:
        thresholds = thresholds_dict[RF]['thresholds']
        feature_list = thresholds_dict[RF]['feature_list']
        
        # Create custom colormap (logarithmic magma)
        magma, bob, counter, cmap_name = matplotlib.cm.get_cmap('magma'), list(), 0.0, 'log_magma'       
        for step in ['aa','bb','cc','dd','ee','ff','gg','hh','ii']:
            exec(step+"=("+ str((counter)**3) +", magma(counter))")
            exec("bob.append("+step+")")
            counter += 0.125
        cm = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, bob)

        # Define the additive widths of each plot
        c = {'1s':0, '1e':40, '2s':0, '2e':40, '3s':37, '3e':42}
        r = {'1s': 0, '1e': 20, '2s': 20, '2e':30, '3s': 0, '3e': 18}

        # Define the number of columns in the plot. height is unlimited
        cols = 3

        # Deine the total height and width of a single figure
        # If this changes, the total GridSpec size needs to also change
        h = 45
        w = 50

        # Define matplotlib style
        plt.style.use('default')

        # Start figure and GridSpec
        fig = plt.figure(figsize=(20,20))
        gs1 = GridSpec(320, 165, wspace=0.0, hspace=0.0)

        # Loop through each feature (feature list and thresholds could be ranked by importance)
        for i in range(1, len(feature_list)+1):
            #print ("\n\n\n", feature_list[i])

            # Calculate the column and row that the figure should be in
            col = (i-1)%cols+1
            row = int(np.floor((i-1)/3)+1)

            # Calculate the GridSpec index to add to each row and column
            c_ = (col-1)*w
            r_ = (row-1)*h

            # Create the three subplots for the figure
            exec("ax"+str(i)+" = fig.add_subplot(gs1["+str(r_ + r['1s'])+":"+str(r_ + r['1e'])+", "+str(c_ + c['1s'])+":"+str(c_ + c['1e'])+"])")
            exec("ax"+str(i)+"_ = fig.add_subplot(gs1["+str(r_ + r['2s'])+":"+str(r_ + r['2e'])+", "+str(c_ + c['2s'])+":"+str(c_ + c['2e'])+"])")
            exec("ax"+str(i)+"__ = fig.add_subplot(gs1["+str(r_ + r['3s'])+":"+str(r_ + r['3e'])+", "+str(c_ + c['3s'])+":"+str(c_ + c['3e'])+"])")

            # Calculate the min, max and range of thresholds
            minn = np.min(thresholds[i-1]['value'])
            maxx = np.max(thresholds[i-1]['value'])
            rangee = maxx - minn

            # Plot heatmap                 
            exec("heat_ax = ax"+str(i))
            exec("max_h = threshold_heatmap(ax=heat_ax, thresholds=thresholds, i=i-1, minn=minn, maxx=maxx, rangee=rangee, cm=cm)")

            # Plot histogram
            exec("hist_ax = ax"+str(i)+"_")
            exec("threshold_hist(ax=hist_ax, data=thresholds[i-1]['value'], minn=minn, maxx=maxx, rangee=rangee, xlabel=(feature_list[i-1] + feature_units[feature_list[i-1]]))")

            # Plot heatmap colorbar
            exec("hist_ax = ax"+str(i)+"__")
            exec("threshold_colorbar(ax=hist_ax, max_h=max_h, cm=cm)")

        # Save and display
        fig.savefig(outdir+'feature_thresholds_'+str(RF)+'.pdf', bbox_inches="tight", dpi=250)
        fig.savefig(outdir+'feature_thresholds_'+str(RF)+'.png', bbox_inches="tight", dpi=250)

        #plt.show()
        plt.close()




def plot_rf_tree(RF_dict, outdir, number=0):
    for RF in RF_dict:
        clf, flabels = RF_dict[RF]['model'], RF_dict[RF]['labels']
        tree.plot_tree(clf.estimators_[number], feature_names=flabels, node_ids=True, filled=True, proportion=True, rounded=True, precision=2) 
        plt.savefig(outdir+'tree.png')
        plt.savefig(outdir+'tree.pdf')
        plt.close()