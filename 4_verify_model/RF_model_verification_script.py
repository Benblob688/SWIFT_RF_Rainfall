'''
Perform a full set of verification plots for as many RF models and other algorithms as specified.

Columns of ver (verification) array are:
0:  YYYYMMDDhhmm
1:  Annual_cos
2:  Annual_sin
3:  Diurnal_cos
4:  Diurnal_sin
5:  Latitude
6:  Longitude
7:  Solar_elevation
8:  Solar_azimuth_cos
9:  Solar_azimuth_sin
10: IMERG
11: CRR
12: CRR-Ph      # Need to add NIPE below here !!!!
13: RF v1
14: RF v2
X:  RF vX

Many RF models can be added without too much compute time, importing the original data is the difficulty.
The ver array may get very large and thus be an issue for memory, however.
May need to do verification of each year separately, and then combine.
'''


import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import os
import sys
import glob
import math
import pickle
import joblib
import numpy as np
from datetime import timedelta, date
import datetime
import joblib
import RF_verification_functions as RF
import scipy.stats as sci
np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.filterwarnings('ignore') # comment out if the software is not working, may help diagnose issues. Otherwise unnecessary 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import csv
from sklearn import tree
import string 
from pprint import pprint
#!!!!!!!!!!!!!!!! consider offloading all the imports to the functions file, then not needed here. Ugly for end user. In documentation, group into "common python modules" and "special ones to download for this to work" and will have python environment (not OS-specific) to help ease of install too.

np.random.seed(42)

start_time = datetime.datetime.now() # For tracking the performance of this verification script.

#########################
####### SETTINGS ########
#########################

# Import parameters for this particular model (e.g. base_SEVIRI_no_diff). Full path to file needed.
param_file = sys.argv[1] #/gws/nopw/j04/swift/bpickering/random_forest_precip/0_model_parameters/parameters___base_seviri_no_diff.pkl

RF_parameters = pickle.load( open(param_file, "rb") )

pprint(RF_parameters)

np.random.seed(RF_parameters["random_state"])

  
# Bin midpoints/boundaries and end caps for all the products.
bin_edges, bin_values, bin_labels = RF.create_bin_values_and_labels(boundaries=RF_parameters["bin_edges"])
print("\nRain rate boundaries:", list(bin_labels.values()))

# Models to be verified (supports multiple)
# !!! be aware that timestamp cannot be automated and will have to be entered manually.
modelpaths = {
        1: RF_parameters['modeldir']
    + 'RF_model___base_seviri_plus_diff_plus_topog_plus_wavelets___201501-201912_all_0.5perc___bins_0-0.5-2-5-10-20-35-60-100-200__'
    + 'est100_balanced_subsample_gini_maxfeat4_minsplit5_maxdepth10_bootstrap_0.0001_time-20220306-0647.pkl'}
    
#     + 'RF_model___base_seviri_plus_diff_plus_topog_plus_wavelets___201501-201912_all_5.0perc___bins_0-0.5-2-5-10-20-35-60-100-200__'
#     + 'est100_balanced_subsample_gini_maxfeat4_minsplit5_maxdepth10_bootstrap_0.0001_time-20220304-1912.pkl'}
    
#         + 'RF_model___short_seviri_diff_all___201501-201912_all_0.1perc___bins_0-0.5-2-5-10-20-35-60-100-200__est100_balanced_subsample_gini_maxfeatNone_minsplit0.001_maxdepthNone_bootstrap_0.01_time-20220118-2153.pkl'}
#         + 'RF_model___base_seviri_plus_diff_plus_topography___201501-201912_all_10.0perc___bins_0-0.5-2-5-10-20-35-60-100-200__est250_balanced_gini_maxfeatlog2_minsplit2_maxdepth40_bootstrap_0.0001_time-20220207-1703.pkl'}
    # short_test
#     1: RF_parameters['modeldir']
#     + 'RF_model___base_seviri_no_diff___201501-201912_all_10.0perc___bins_0-0.5-2-5-10-20-35-60-100-200__est150_balanced_subsample_gini_maxfeatlog2_minsplit0.005_maxdepth40_bootstrap_0.001_time-20220119-0103.pkl',
#     2: RF_parameters['modeldir']
#     + 'RF_model___short_seviri_diff_all___201501-201912_all_0.1perc___bins_0-0.5-2-5-10-20-35-60-100-200__est100_balanced_subsample_gini_maxfeatNone_minsplit0.001_maxdepthNone_bootstrap_0.01_time-20220118-2153.pkl',
#     3: RF_parameters['modeldir']
#     + 'RF_model___short_seviri_diff_day_only___201501-201912_all_0.1perc___bins_0-0.5-2-5-10-20-35-60-100-200__est100_balanced_subsample_gini_maxfeatNone_minsplit0.001_maxdepthNone_bootstrap_0.01_time-20220118-2151.pkl',
#     4: RF_parameters['modeldir']
#     + 'RF_model___short_seviri_diff_night_only___201501-201912_all_0.1perc___bins_0-0.5-2-5-10-20-35-60-100-200__est100_balanced_subsample_gini_maxfeatNone_minsplit0.001_maxdepthNone_bootstrap_0.01_time-20220118-2151.pkl'
# }



# Product name: ver_array column index
prod_i = {'CRR': 11,
          'CRR_Ph': 12,
          'Random_Forest': 13} #,
#           'RF_short_seviri_diff_all': 14,
#           'RF_short_seviri_diff_day_only': 15,
#           'RF_short_seviri_diff_night_only': 16
#          }#,
#           'RF_v5': 14,
#           'RF_v6': 15,
#           'RF_v7': 16,
#           'RF_v8': 17}

# Columns of ver (verification) array are:
ver_labels={0: 'YYYYMMDDhhmm',
            1: 'Annual_cos',
            2: 'Annual_sin',
            3: 'Diurnal_cos',
            4: 'Diurnal_sin',
            5: 'Latitude',
            6: 'Longitude',
            7: 'Solar_elevation',
            8: 'Solar_azimuth_cos',
            9: 'Solar_azimuth_sin',
            10: 'IMERG',
            11: 'CRR',
            12: 'CRR_Ph',
            13:'Random_Forest'}#,
#             14:'RF_short_seviri_diff_all',
#             15:'RF_short_seviri_diff_day_only',
#             16:'RF_short_seviri_diff_night_only'}
#             17:'RF_v5',
#             18:'RF_v6',
#             19:'RF_v7',
#             20:'RF_v8'}
            #X: 'RF_vX'

# Units of each feature – update as more are added.
feature_units = {
    'Annual_cos': ' ',
    'Annual_sin': ' ',
    'Diurnal_cos': ' ',
    'Diurnal_sin': ' ',
    'Latitude': ' (º)',
    'Longitude': ' (º)',
    'Solar_elevation': ' (º)',
    'Solar_azimuth_cos': ' ',
    'Solar_azimuth_sin': ' ',
    'MSG_0.6': ' Reflectance (%)',
    'MSG_0.8': ' Reflectance (%)',
    'MSG_1.6': ' Reflectance (%)',
    'MSG_3.9': ' Brightness Temperature (K)',
    'MSG_6.2': ' Brightness Temperature (K)',
    'MSG_7.3': ' Brightness Temperature (K)',
    'MSG_8.7': ' Brightness Temperature (K)',
    'MSG_9.7': ' Brightness Temperature (K)',
    'MSG_10.8': ' Brightness Temperature (K)',
    'MSG_12.0': ' Brightness Temperature (K)',
    'MSG_13.4': ' Brightness Temperature (K)',
    'random': ' ',
    'MSG_6.2-7.3': ' BT Difference (K)',
    'MSG_3.9-10.8': ' BT Difference (K)',
    'MSG_1.6-0.6': ' Reflectance Difference (%)',
    'MSG_0.6-10.8': ' N/A',
    'MSG_0.8-10.8': ' N/A',
    'MSG_1.6-10.8': ' N/A',
    'MSG_6.2-10.8': ' BT Difference (K)',
    'MSG_7.3-10.8': ' BT Difference (K)',
    'MSG_8.7-10.8': ' BT Difference (K)',
    'MSG_9.7-10.8': ' BT Difference (K)',
    'MSG_12.0-10.8': ' BT Difference (K)',
    'MSG_13.4-10.8': ' BT Difference (K)',
    'elevation': ' a.m.s.l. (m)',
    'prominence': ' (m)',
    'w_30.0': ' (exponent)',
    'w_60.0': ' (exponent)', 
    'w_120.0': ' (exponent)', 
    'w_240.0': ' (exponent)', 
    'w_302.4': ' (exponent)'
}

# Markers for time stats plots and performance diagram
ver_markers={11: 'h', # hexagon (vertical point)
             12: '*', # 5-point star
             13:'P', # plus (filled)
             14:'s', # square
             15:'p', # pentagon
             16:'v', # tringle down
             17:'o', # circle
             18:'X', # X (filled)
             19:'d', # thin diamond
             20:'^'} # triangle_up

# Colours for performance diagram
ver_colors={11:'tab:blue',
            12:'tab:orange',
            13:'tab:green',
            14:'tab:red',
            15:'tab:purple',
            16:'tab:brown',
            17:'tab:pink',
            18:'tab:gray',
            19:'tab:olive',
            20:'tab:cyan'}



#########################
#### LOAD FEATURES ######
#########################
print("2-IMPORT VERIFICATION DATA")
'''            Usage:
startdate      Type: datetime YYYY MM DD hh mm. 
enddate        Type: datetime YYYY MM DD hh mm. 
exclude_MM     Type: list of strings with leading zero, e.g. [‘01’, ‘02’].
exclude_hh     Type: list of strings with leading zero, e.g. [‘01’, ‘02’].
perc_keep      Type: float. Set what approximate percentage of the file list should be imported for training.
traindir       Type: str. Path to the training data directory.
random         Type: Bool. Whether to include random number generator in the feature list.
solar          Type: Bool. Whether to include solar eleveation and azimuth in the feature list.
other features Type: Bool. Whether to include that feature in the feature list.

               Returns:
features       Type: pandas.dataframe. All features and label data valid for the user-input settings.
'''
features = RF.import_feature_data_verification(
    # general settings
    startdate=RF_parameters["verify_startdate"], 
    enddate=RF_parameters["verify_enddate"], 
    exclude_MM=RF_parameters["exclude_MM"], 
    exclude_hh=RF_parameters["exclude_hh"], 
    perc_exclude=RF_parameters["verify_perc_exclude"], 
    traindir=RF_parameters["traindir"], 
    # base_features
    random=RF_parameters["base_features"]["random"], 
    solar=(RF_parameters["base_features"]["Solar_elevation"] or RF_parameters["base_features"]["Solar_azimuth_cos"] or RF_parameters["base_features"]["Solar_azimuth_sin"]), # if any solar is true then yes
    # key_features:
    seviri_diff=RF_parameters["key_features"]["seviri_diff"], # Maybe they should just always be true, it doesn't take much longer (as %) to import all features...
    topography=RF_parameters["key_features"]["topography"],
    wavelets=RF_parameters["key_features"]["wavelets"],
#     GFS_TCVW=RF_parameters["key_features"]["GFS_TCVW"],
#     GFS_field=RF_parameters["key_features"]["GFS_field"]
    )



# Create empty masked array for verification database
ver = np.ma.masked_all(( (len(features)), 13+len(modelpaths) )) # 13 is the length of the array with just metadata, IMERG, CRR, CRR-Ph

# Fill the first 7 columns with basic data (time, location)
ver[:,:7] = np.array(features)[:,:7]
# Fill the columns [7:10] with solar azimuth (cos and sin) and solar elevation data.
ver[:, 7], ver[:, 8], ver[:, 9] = np.array(features['Solar_elevation']), np.array(features['Solar_azimuth_cos']), np.array(features['Solar_azimuth_sin'])



'''           Usage:
features      Type: pandas.dataframe. All features and label data valid for the user-input settings.
bin_edges     Type: list. Edges of precipitation rate bins set in the model parameters file.

              Returns:
features      Type: np.array. Feature values for each pixel, where the indexing matches the variable "labels" for a given pixel.
labels        Type: np.array. True values for each pixel, where the indexing matches the variable "features" for a given pixel.
feature_list  Type: list. Names for all the columns in the features set.

N.B.:         Must be run immediately after import_feature_data
'''
# !!!!! Delete
print (features.columns)
#
features, labels, all_features_list = RF.sort_feature_data(
    features=features, 
    bin_edges=np.array(RF_parameters["bin_edges"]).astype(np.float64)
)


# Put IMERG values into the verification array
ver[:, 10] = labels[:]

#########################
#### LOAD RF MODELS #####
#########################
# Load a previously trained Random Forest model.
# !!! be aware that timestamp cannot be automated and will have to be entered manually to modelpaths dictionary at top of script.
print("loading RF models...")

RF_dict = {}

for i in range(1, len(modelpaths)+1):
    start = 12 # starting index of models !!!! this "12+" will have to change if verification products (NIPE), or metadata like topography are added
    
    # Load model
    RF_model = joblib.load(modelpaths[i])
    
    # Load the model metadata (currently only feature_list)
    feature_list = RF_model.feature_names_in_
    print(feature_list)
    
    # Make features_ array containing only the features that this particular model is expecting.
    if len(feature_list) > len(all_features_list):
        raise ValueError("Model has more features in it's list than are avaialble from the main features array.")
    f_index = []
    for f in feature_list: # loop through all the feature labels for this particular model
        f_index.append(all_features_list.index(f)) # find the position of the feature label in the wider/main features array
    features_ = features[:,f_index] # slice out only the features needed for this array
    
    # Use the forest's predict method on the test data to generate RF estimates. Put RF estimates into the verification array.
    RF_predictions = RF_model.predict(features_) #provide only the features this particular model expects to see.
    ver[:, start+i]= RF_predictions[:] 
    
    # Add model and labels to RF_dict
    exec("RF_dict.update({'"+ver_labels[start+i]+"': {'model': RF_model, 'labels': feature_list} })")
    print(ver_labels[start+i], "loaded")
    exec("print('Last tree count:', RF_dict['"+ver_labels[start+i]+"']['model'].estimators_[-1].tree_.node_count)")
    
pprint(RF_dict)


#########################
### LOAD BASE PRECIP ####
#########################
print("adding CRR...")
ver = RF.add_CRR(ver_arr = ver, bin_edges=np.array(RF_parameters["bin_edges"]).astype(np.float64), CRR_dir = '/gws/nopw/j04/swift/bpickering/random_forest_precip/4_verify_model/CRR_regridded/data/')
print("adding CRR-Ph...")
ver = RF.add_CRR_Ph(ver_arr = ver, bin_edges=np.array(RF_parameters["bin_edges"]).astype(np.float64), CRR_Ph_dir = '/gws/nopw/j04/swift/bpickering/random_forest_precip/4_verify_model/CRR-Ph_regridded/data/')
# print("adding NIPE...")
# ver = RF.add_NIPE(ver_arr = ver, bin_edges=np.array(RF_parameters["bin_edges"]).astype(np.float64), NIPE_dir = '/gws/nopw/j04/swift/bpickering/random_forest_precip/4_verify_model/NIPE_regridded/data/')

# Convert digitised values into real rain rate values
ver[:, 10:] = RF.precip_bin_values(data=ver[:, 10:], bin_values=bin_values)

# Print the first full verification table entry with textual descriptions
for i in range(0, len(ver[0, :])):
    if len(ver_labels[i]) < 11:
           print (ver_labels[i] + ':\t\t' + str(ver[0, i]))
    else:
           print (ver_labels[i] + ':\t' + str(ver[0, i]))
            
            

########################
### OVERALL STATS ######
########################
print('Creating general stats file...')
RF.generate_stats_file(outdir=RF_parameters['verdir'], ver_labels=ver_labels, ver=ver, bin_edges=np.array(RF_parameters["bin_edges"]).astype(np.float64))
print('Done!')


#########################
###### TIME STATS #######
#########################   !!!!! not all plots appear. Lines too thick for multiple models.
print('Creating diurnal and annual plots...')
RF.time_stats(ver=ver, prod_i=prod_i, ver_markers=ver_markers, outdir=RF_parameters['verdir'], bin_edges=np.array(RF_parameters["bin_edges"]).astype(np.float64))
print('Done!')



#########################
##### SOLAR STATS #######
#########################
# Make plots that show the statistics vs. solar elevation, so max around +/- 50 º, seeing if sunset changes the skill level.
# Will not show solar elevation direction (sunset or sunrise) so still need 3-hly plot above.



#########################
###### MAP STATS ######## # Will add the total normalised precip here long-term (need to remove some stats plots to make room for it)
#########################
print('Creating statistics map plots...')

# Create the directory for map_stats arrays if it doesn't exist
if not os.path.exists(RF_parameters['verdir']+"/map_stats"):
    os.mkdir(RF_parameters['verdir']+"/map_stats")
    
# Make arrays of statistics for regions of Africa    
for p in prod_i:
    print ('\n\n\n', p)
    
    # Create the map_stats 2D arrays
    exec (p+"_num_samples_r,"+p+"_mae_r,"+p+"_max_err_r,"+p+"_cov_r,"+p+"_mean_err_r,"+p+"_bias_r,"+p+"_p_corr_r,"+p+"_r2_r,"+p+"_s_corr_r,"+p+"_mse_r,"+p+"_rmse_r,"+p+"_hss_r = RF.map_stats(truth=ver[:, 10], test=ver[:, "+str(prod_i[p])+"], bin_edges=np.array(RF_parameters['bin_edges']).astype(np.float64), lats = ver[:, 5], lons = ver[:, 6], d_lats = np.linspace(38., -35., 731), d_lons = np.linspace(-20., 52., 721), stat_px = 10)")
    
    # Save the map_stats 2D arrays for quick import later.
    for stat in ["_num_samples_r","_mae_r","_max_err_r","_cov_r","_mean_err_r","_bias_r","_p_corr_r","_r2_r","_s_corr_r","_mse_r","_rmse_r","_hss_r"]:
        exec(p+stat+".dump('"+RF_parameters['verdir']+"/map_stats/"+p+stat+"')")

# inputs: ver, outdir, ver_labels
# Make arrays of total observations, and total precipitation observed, for each pixel in the Africa domain
for i in range(10, len(ver[0,:])): # loop through all the products, including the truth label, IMERG
    print(ver_labels[i])
    
    # Create the precipitation data and number of observations
    exec(ver_labels[i]+"_precip_data, "+ver_labels[i]+"_precip_num = RF.generate_map(rain_col=ver[:, "+str(i)+"], lat_data=ver[:, 5], lon_data=ver[:, 6])")
    
    # Save the map_stats 2D arrays for quick import later
    exec(ver_labels[i]+"_precip_data.dump('"+RF_parameters['verdir']+"/map_stats/"+ver_labels[i]+"_precip_data')")
    exec(ver_labels[i]+"_precip_num.dump('"+RF_parameters['verdir']+"/map_stats/"+ver_labels[i]+"_precip_num')")

# Strings depicting the statistics to be plotted.
# statlabels = ["_num_samples_r", "_mae_r","_max_err_r","_cov_r","_mean_err_r","_bias_r","_p_corr_r","_r2_r","_s_corr_r","_mse_r","_rmse_r","_hss_r"] # old, full list
statlabels = ["_num_samples_r", "_mae_r","_max_err_r","_bias_r","_r2_r","_rmse_r","_hss_r","_precip_data","_precip_num"] # new simplified list for ease of interpretation

# Plot multiple subplots of statistics of products across the Pan-Africa domain.
RF.plot_map_stats(prod_i, statlabels, outdir=RF_parameters['verdir'])
print('Done!')



#########################
######### GRID ##########
#########################
print('Creating grid plots...')
# Grid card plots of product performance  
RF.grid(
    outdir=RF_parameters['verdir'], 
    ver=ver, 
    ver_labels=ver_labels,
    bin_edges=np.array(RF_parameters["bin_edges"]).astype(np.float64), 
    bin_labels=bin_labels,
    truthname='IMERG'
)
print('Done!')



#########################
## PERFORMANCE DIAGRAM ##
#########################
print('Creating performance diagrams...')
performance = RF.performance_stats(ver=ver, truth_i=10, ver_labels=ver_labels, ver_markers=ver_markers, ver_colors=ver_colors, bin_edges=np.array(RF_parameters["bin_edges"]).astype(np.float64))

RF.performance_diagram(outdir=RF_parameters['verdir'], p=performance, bin_labels=bin_labels, description='')
RF.performance_diagram_zoomed(outdir=RF_parameters['verdir'], p=performance, bin_labels=bin_labels, description='')
print('Done!')



    
#########################
###### IMPORTANCES ######
#########################
print('Creating importances plot...')
RF.plot_importances(RF_dict=RF_dict, outdir=RF_parameters['verdir'])
print('Done!')



#########################
## FEATURE THRESHOLDS ###
#########################
print('Creating feature thresholds plot...')

# Make thresholds dictionary
thresholds_dict = RF.make_thresholds(RF_dict=RF_dict)

# Make and save plot
RF.plot_feature_thresholds(RF_dict=RF_dict, thresholds_dict=thresholds_dict, feature_units=feature_units, outdir=RF_parameters['verdir'])

print('Done!')



#########################
##### DECISION TREE #####
#########################
print('Creating decision tree plot...')
RF.plot_rf_tree(RF_dict=RF_dict, outdir=RF_parameters['verdir'], number=0)
print('Done!')





# Report execution time
end_time = datetime.datetime.now()
time_taken = end_time-start_time
print("Full Script Time Taken = ", time_taken)
print("SCRIPT SUCCESSFUL")