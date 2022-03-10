'''
Description:
    Code to train a random forest model with scikit-learn and then save it.

Usage:
    [STANDARD]
    python RF_tune_model.py <PATH_TO_PARAMETERS_FILE>
    [SLURM]
    sbatch -p par-single -n 16 -t 2-00:00:00 -mem 65536 -o $basedir/out/$YYYY$MM-%j.out -e $basedir/err/$YYYY$MM-%j.err --wrap="python <PATH>/RF_tune_model.py <PATH_TO_PARAMETERS_FILE> "

Inputs:
    sys.arvg[1]    str     full path to the file containing all of the model parameters which will guide this script

Accesses: 
    current directory + /RF_settings
    parameters file

Outputs:
    a python dictionary containing the chosen hyperparameters and most important features for use in model training,
    in the following format: ./RF_settings/settings___<model_name>.pkl
'''
#########################
####### 0-IMPORTS #######
#########################
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
from sklearn.model_selection import RandomizedSearchCV
import RF_tune_model_functions as RF
from sklearn.feature_selection import SelectFromModel
from itertools import compress
from pprint import pprint

# Track the performance of this training script. Only useful when "run all cells" applied.
start = datetime.datetime.now()

#########################
###### 1-SETTINGS ####### !!!! (these should be baked-in or user selectable in some way)
#########################
print("1-SCRIPT SETTINGS")

# Import parameters for this particular model (e.g. base_SEVIRI_no_diff). Full path to file needed.
param_file = sys.argv[1] #/gws/nopw/j04/swift/bpickering/random_forest_precip/0_model_parameters/parameters___base_seviri_no_diff.pkl

RF_parameters = pickle.load( open(param_file, "rb") )


pprint(RF_parameters)

np.random.seed(RF_parameters["random_state"])



#########################
##### 2-IMPORT DATA #####
#########################
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
features, key_features = RF.import_feature_data(
    # general settings
    startdate=RF_parameters["tune_startdate"], 
    enddate=RF_parameters["tune_enddate"], 
    exclude_MM=RF_parameters["exclude_MM"], 
    exclude_hh=RF_parameters["exclude_hh"], 
    perc_keep=RF_parameters["tune_perc_keep"], 
    traindir=RF_parameters["traindir"],
    # base_features
    random=RF_parameters["base_features"]["random"], 
    solar=(RF_parameters["base_features"]["Solar_elevation"] or RF_parameters["base_features"]["Solar_azimuth_cos"] or RF_parameters["base_features"]["Solar_azimuth_sin"]), # if any solar is true then yes
    # key_features:
    seviri_diff=RF_parameters["key_features"]["seviri_diff"],
    topography=RF_parameters["key_features"]["topography"],
    wavelets=RF_parameters["key_features"]["wavelets"] #,
#     GFS_TCVW=RF_parameters["key_features"]["GFS_TCVW"],
#     GFS_field=RF_parameters["key_features"]["GFS_field"]
    )

force_features, desired_feature_list = False, None
# This step is optional, only here for final publication results
if RF_parameters["name"] == 'base_seviri_plus_diff_plus_topog_plus_wavelets':
    force_features=True
    desired_feature_list = [
            'MSG_13.4-10.8',
            'MSG_7.3-10.8',
            'MSG_6.2-7.3',
            'Longitude', 
            'MSG_3.9-10.8',
            'Solar_azimuth_sin',
            'Solar_elevation',
            'Latitude',
            'w_30.0',
            'w_60.0'
        ]

if RF_parameters["name"] == '8_features':
    force_features=True
    desired_feature_list = [
            'MSG_13.4-10.8',
            'MSG_7.3-10.8',
            'MSG_6.2-7.3',
            'Longitude', 
            'MSG_3.9-10.8',
            'Solar_azimuth_sin',
            'Latitude',
            'w_30.0'
        ]

    
if RF_parameters["name"] == '6_features':
    force_features=True
    desired_feature_list = [
            'MSG_13.4-10.8',
            'MSG_6.2-7.3',
            'Longitude', 
            'MSG_3.9-10.8',
            'Solar_azimuth_sin',
            'Latitude'
        ]

if RF_parameters["name"] == '4_features':
    force_features=True
    desired_feature_list = [
            'MSG_13.4-10.8',
            'MSG_6.2-7.3',
            'MSG_3.9-10.8',
            'Solar_azimuth_sin'
        ]
    
if RF_parameters["name"] == '2_features':
    force_features=True
    desired_feature_list = [
            'MSG_13.4-10.8',
            'MSG_6.2-7.3'
        ]

'''           Usage:
features      Type: pandas.dataframe. All features and label data valid for the user-input settings.
bin_edges     Type: list. Edges of precipitation rate bins set in the model parameters file.

              Returns:
features      Type: np.array. Feature values for each pixel, where the indexing matches the variable "labels" for a given pixel.
labels        Type: np.array. True values for each pixel, where the indexing matches the variable "features" for a given pixel.
feature_list  Type: list. Names for all the columns in the features set.

N.B.:         Must be run immediately after import_feature_data
'''
features, labels, feature_list = RF.sort_feature_data(
    features = features, 
    bin_edges = np.array(RF_parameters["bin_edges"]).astype(np.float64),
    force_features = force_features,
    desired_feature_list = desired_feature_list
)
print(features)
print(features.shape)
print(feature_list)




#########################
# 3-RANDOMIZEDSEARCHCV ##
#########################
print("3-RandomizedSearchCV")
random_grid = RF_parameters["random_grid"]
pprint(random_grid)

# Calculate total number of parameters to be tested.
total=1
for arg in random_grid:
    total = total * len(random_grid[arg]) 
total = total
print("Total number of RandomizedSearchCV ensembles:", total) 

'''                 Usage:
estimator           Type: RandomForestClassifier object. Can be changed to other type of statistical model if desired, but will require heavy edits to all other code.
param_distributions Type: dict. All possible parameter values for the RandomForestClassifier to use in training.
n_iter              Type: int. Number of interations (ensembles) that will be tested in RandomizedSearchCV. Here a fraction of the total number are used.
cv                  Type: int. Number of train/test splits to perform to average out the model score results. 5 is default, 3 used to be default. Lower is faster.
verbose             Type: int. How much information to print out during RandomizedSearchCV.
random_state        Type: int. Allows the randomness to be repeated later.
n_jobs              Type: int. Number of jobs (i.e. threads/cores) to use when performing RandomizedSearchCV.

                    Returns:
rf_random           Type: RandomizedSearchCV object. Contains all of the results of the RandomizedSearchCV process, including best_estimator_ and best_params_ which get exported later.
'''
rf_random = RandomizedSearchCV(
    estimator           = RandomForestClassifier(
        random_state    = RF_parameters["random_state"], # Keeps the results repeatable by locking the randomness
        criterion       = RF_parameters["train_criterion"],
        verbose         = 0
        ), 
    param_distributions = random_grid, 
    n_iter              = int(total * RF_parameters["tune_n_iter"] ), #!!!! change to total * 0.1 (10% or something)
    cv                  = RF_parameters["tune_cv"], # cross validation. Default=5. No. of repeat experiments to do through splitting dataset 5 times
    verbose             = 2, 
    random_state        = RF_parameters["random_state"], 
    n_jobs              = RF_parameters["tune_n_jobs"]
    )

# Fit the random search model
start_train = datetime.datetime.now()
rf_random.fit(features, labels)
end_train = datetime.datetime.now()

# Report execution time 
print("RF Training Time Taken = ", end_train-start_train)
### Takes 48 minutes to run for 3-fold, 10 iterations.


file = open(RF_parameters["settingsdir"]+"DEBUG_MODEL___"+RF_parameters["name"]+".pkl", "wb")
pickle.dump(rf_random, file)
file.close()
#########################
### 4-EXPORT RESULTS ####
#########################
print("4-EXPORTING RESULTS")
# RandomizedSearchCV attaches the best parameters to the object class
pprint(rf_random.best_params_)







#!!!!!!!!!!!!!
print('\n\n!!!!! IMPORTANCES !!!!!\n\n')
print ("importances before function 1:", rf_random.best_estimator_.feature_importances_)
print ("whole best_estimator object:")
pprint(rf_random.best_estimator_)
#!!!!!!!!!!!!!







# Plot a histogram of the best estimator's importances
feature_importances = RF.plot_importances(
    RF_dict={
        RF_parameters["name"]: {
            'model': rf_random.best_estimator_,
            'labels': feature_list
        }
      }, 
    outdir=RF_parameters["settingsdir"])



### !!!!! need a better way to pick features. !!!!!
### Suggestion is to use the hyperparameter "n_features" and
### pick that number of the top-importance ranking features.
### Downside is that similar features have low importance but 
### removing both would deteriorate the skill.
    
# with open(outdir+'feature_importances_'+model+'.csv',"w+") as my_csv:
#     csvWriter = csv.writer(my_csv,delimiter=',')
#     csvWriter.writerows(feature_importances)


# Choose the number of features to keep     # NEW !!!!
n_features = rf_random.best_params_['max_features']

chosen_features = []
for i in range(0,len(feature_importances)):
    if not feature_importances[i][0] == 'random':
        chosen_features.append(feature_importances[i][0])

print(chosen_features[:n_features])


# Pick the best features. Always remove random, and force the key features to be put back in.     # OLD !!!!
# sel = SelectFromModel(rf_random.best_estimator_, prefit=True, threshold="mean") # can do "0.75*mean" or "0.5*mean" or choose top x (e.g. top 10) features
# chosen_features = list(compress(feature_list, sel.get_support()))
# if 'random' in chosen_features:
#     print("The 'random' feature importance > mean. Suggest increasing the size of training data given to RandomizedSearchCV'")
#     chosen_features.remove("random")
# for key in key_features:
#     if not key in chosen_features:
#         chosen_features.append(key)
# print (chosen_features)





# Create a dictionary with the chosen features and hyperparameters
RF_settings = {'name': RF_parameters["name"],
               'chosen_features': chosen_features,
               'hyperparameters': rf_random.best_params_}
pprint(RF_settings)

# Dump the dictionary to disk
file = open(RF_parameters["settingsdir"]+"settings___"+RF_parameters["name"]+".pkl", "wb")
pickle.dump(RF_settings, file)
file.close()

# Report execution time
end = datetime.datetime.now()
time_taken = end-start
print("Full Script Time Taken = ", time_taken)
print("SCRIPT SUCCESSFUL")
