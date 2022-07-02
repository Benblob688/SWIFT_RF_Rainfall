'''
Code to train a random forest model with scikit-learn and then save it.

Usage:

python RF_train_model.py <path to parameters.pkl file>

[on JASMIN LOTUS] sbatch -p high-mem -t 48:00:00 -mem 262144 -o $basedir/out/$YYYY$MM-%j.out -e $basedir/err/$YYYY$MM-%j.err --wrap="python RF_train_model.py parameters.pkl "

inputs:
sys.arvg[1]    str     path to parameters .pkl file

accesses: features_datasets sub-directories

outputs: one scikit-learn model with a filename containing the training metadata, in .pkl format
'''
###
### 0-Import statements
###
import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import sys
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
import RF_train_model_functions as RF
from pprint import pprint

start = datetime.datetime.now() # For tracking the performance of this training script.

# Import parameters for this particular model (e.g. base_SEVIRI_no_diff). Full path to file needed.
param_file = sys.argv[1] #/gws/nopw/j04/swift/bpickering/random_forest_precip/0_model_parameters/parameters___base_seviri_no_diff.pkl

RF_parameters = pickle.load( open(param_file, "rb") )
#!!!!! This run only, cycle through perc_keep
RF_parameters['train_perc_keep'] = float(sys.argv[2])
#!!!!!
pprint(RF_parameters)

np.random.seed(RF_parameters["random_state"])

  
# Bin midpoints/boundaries and end caps for all the products.
bin_edges, bin_values, bin_labels = RF.create_bin_values_and_labels(boundaries=RF_parameters["bin_edges"])
print("\nRain rate boundaries:", list(bin_labels.values())) 
for eh in bin_edges, bin_values, bin_labels:
    print (eh, '\n')


# Import training settings for this particular model (e.g. base_SEVIRI_no_diff)
file = open(RF_parameters['settingsdir']+"settings___"+RF_parameters['name']+".pkl", "rb")
RF_settings = pickle.load(file)
pprint(RF_settings)



###
### 3-Import and sort training data
###
print("2-IMPORT TRAINING DATA")
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
features = RF.import_feature_data(
    # general settings
    startdate=RF_parameters["train_startdate"], 
    enddate=RF_parameters["train_enddate"], 
    exclude_MM=RF_parameters["exclude_MM"], 
    exclude_hh=RF_parameters["exclude_hh"], 
    perc_keep=RF_parameters["train_perc_keep"], 
    traindir=RF_parameters["traindir"], 
    # base_features
    random=RF_parameters["base_features"]["random"], 
    solar=(RF_parameters["base_features"]["Solar_elevation"] or RF_parameters["base_features"]["Solar_azimuth_cos"] or RF_parameters["base_features"]["Solar_azimuth_sin"]), # if any solar is true then yes
    # key_features:
    seviri_diff=RF_parameters["key_features"]["seviri_diff"],
    topography=RF_parameters["key_features"]["topography"],
    wavelets=RF_parameters["key_features"]["wavelets"],
#     GFS_TCVW=RF_parameters["key_features"]["GFS_TCVW"],
#     GFS_field=RF_parameters["key_features"]["GFS_field"]
    )


'''           Usage:
features      Type: pandas.dataframe. All features and label data valid for the user-input settings.
bin_edges     Type: list. Edges of precipitation rate bins set in the model parameters file.

              Returns:
features      Type: np.array. Feature values for each pixel, where the indexing matches the variable "labels" for a given pixel.
labels        Type: np.array. True values for each pixel, where the indexing matches the variable "features" for a given pixel.
feature_list  Type: list. Names for all the columns in the features set.

N.B.:         Must be run immediately after import_feature_data
'''
all_features, labels, all_features_list = RF.sort_feature_data(
    features=features, 
    bin_edges=np.array(RF_parameters["bin_edges"]).astype(np.float64)
)
print(all_features)
print(all_features.shape)
print(all_features_list)


# Make features_ array containing only the features that have been chosen for this particular model.
feature_list = RF_settings['chosen_features']
print(feature_list)
if len(feature_list) > len(all_features_list):
    raise ValueError("Model has more features in it's list than are avaialble from the main features array.")
f_index = []
for f in feature_list: # loop through all the feature labels chosen for this particular model
    f_index.append(all_features_list.index(f)) # find the position of the feature label in the wider/main features array
features = all_features[:,f_index] # slice out only the features needed for this array
print(features)




###
### 4-Train model
###

# Instantiate model
rf = RandomForestClassifier(
    n_jobs       = RF_parameters['train_n_jobs'],
    criterion    = RF_parameters['train_criterion'],
    random_state = RF_parameters["random_state"], 
    
    n_estimators      = RF_settings['hyperparameters']['n_estimators'], 
    min_samples_split = RF_settings['hyperparameters']['min_samples_split'], 
    max_features      = RF_settings['hyperparameters']['max_features'], 
    max_depth         = RF_settings['hyperparameters']['max_depth'], 
    max_samples       = RF_settings['hyperparameters']['max_samples'],
    bootstrap         = RF_settings['hyperparameters']['bootstrap'],
    class_weight      = RF_settings['hyperparameters']['class_weight'] 
)

# Train the model on training data
start_rf = datetime.datetime.now()
rf.fit(features, labels);
end_rf = datetime.datetime.now()

# Report execution time
time_taken = end_rf-start_rf
print("RF Training Time Taken = ", time_taken)


###
### 5-Export model
###

# Basic Data Settings
modelname = 'RF_model___' + RF_parameters['name'] + '___' + RF_parameters["train_startdate"].strftime('%Y%m') + '-' + RF_parameters["train_enddate"].strftime('%Y%m') + '_' + RF_parameters["diurnal"] + '_' + str(RF_parameters["train_perc_keep"]*100) + 'perc___bins_'
# Bin spacings
for binn in RF_parameters["bin_edges"]:
    if binn == RF_parameters["bin_edges"][-1]:
        modelname = modelname + str(binn) # no dash on last bin edge
    else:
        modelname = modelname + str(binn) + "-"
# Basic Training Settings
modelname = modelname + '__est' + str(RF_settings['hyperparameters']["n_estimators"]) + '_' + RF_settings['hyperparameters']["class_weight"] + '_' + RF_parameters['train_criterion'] + '_maxfeat' + str(RF_settings['hyperparameters']["max_features"]) + '_minsplit' + str(RF_settings['hyperparameters']["min_samples_split"]) + '_maxdepth' + str(RF_settings['hyperparameters']["max_depth"])
# Optional Training Settings
if RF_settings['hyperparameters']['bootstrap']:
    modelname = modelname + '_bootstrap_' + str(RF_settings['hyperparameters']["max_samples"])

# Save the model (be aware that timestamp won't work and will have to be entered manually)

name = RF_parameters['modeldir'] + modelname + '_time-' + datetime.datetime.now().strftime("%Y%m%d-%H%M")
print (name)
rf.feature_names_in_ = feature_list
joblib.dump(rf, name + '.pkl')


###
### 6-Stats Reporting
###

# Report execution time
end = datetime.datetime.now()
time_taken = end-start
print("Full Script Time Taken = ", time_taken)
print("SCRIPT SUCCESSFUL")