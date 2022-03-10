'''
Code to create a dictionary of parameters for a new single RF model 
which will be used by later scripts to tune, train, and test that new model.

Usage:

python RF_make_model_parameters.py

inputs:
all done in the script which must be opened and edited to make a new model parameter file

accesses: current directory only

outputs: one dictionary containing parameters for tuning, training and testing a new RF model, in .pkl format
'''
#########################
####### 0-IMPORTS #######
#########################


import pickle
import numpy as np
import datetime
from pprint import pprint

dirunal_dict = {'all': [],
                # Only three UTC hours have perfect 'day' and 'night' solar properties.
                'day':  ['13','14','15','16','17','18','19','20','21','22','23','00','01','02','03','04','05','06','07','08','09'], # 10:00-12:45 (3h)
                'night': ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21']} # 22:00-00:45 (3h)


#########################
###### 1-SETTINGS #######
######################### 

RF_parameters = {
    'name':         '2_features',
    'traindir':     '/gws/nopw/j04/swift/bpickering/random_forest_precip/1_training_data/',
    'settingsdir':  '/gws/nopw/j04/swift/bpickering/random_forest_precip/2_tune_model/RF_settings/',
    'modeldir':     '/gws/nopw/j04/swift/bpickering/random_forest_precip/3_train_model/RF_models/',
    'verdir':       '/gws/nopw/j04/swift/bpickering/random_forest_precip/4_verify_model/output/base_seviri_plus_diff_plus_topog_plus_wavelets/2_features/',
    
    'bin_edges': ['0', '0.5', '2', '5', '10', '20', '35', '60', '100', '200'], # Classes will be between these values in mm/h e.g. 0-0.5 mm/hr
    'random_state': 42, # Random repeatability
    'diurnal': 'all',
    'exclude_MM': [],
    'exclude_hh': dirunal_dict['all'], # must match 'diurnal' key above short_test = all
    
    # Features
    'base_features': {
        'Annual_cos': True,
        'Annual_sin': True,
        'Diurnal_cos': True,
        'Diurnal_sin': True,
        'Latitude': True,
        'Longitude': True,
        'Solar_elevation': True,
        'Solar_azimuth_cos': True,
        'Solar_azimuth_sin': True,
        'MSG_0.6': True,
        'MSG_0.8': True,
        'MSG_1.6': True,
        'MSG_3.9': True,
        'MSG_6.2': True,
        'MSG_7.3': True,
        'MSG_8.7': True,
        'MSG_9.7': True,
        'MSG_10.8': True,
        'MSG_12.0': True,
        'MSG_13.4': True,
        'random': True
    },
    'key_features': { 
        'seviri_diff': True,
        'topography': False,
        'wavelets': True,
        'GFS_TCVW': False, # Feature was never added
        'GFS_field': False # Feature was never added
        # adding new features requires editing all scripts
    },
    
    # Tune
    'tune_startdate': datetime.datetime(2015,1,1,0,0),
    'tune_enddate': datetime.datetime(2019,12,31,23,45),
    'tune_perc_keep': float(0.001),    # quick testing value is 0.0001, normal 0.001, higher results in memory issues (5-year 256GB mem on JASMIN)
    'random_grid': {
        # Number of trees in random forest
        'n_estimators': [50, 100, 150, 250, 500], 
        # Number of features to consider at every split
        'max_features': [4, 5, 6, 8, 10, 12], 
        # Maximum number of levels in tree
        'max_depth': [10, 20, 40, 80, None, None], 
        # Minimum number of samples required to split a node
        # This can't be higher than the number of samples given to a tree. Otherwise zero splits occur.
        'min_samples_split': [2, 5, 10, 50, 100], 
        # Method of selecting samples for training each tree
        'bootstrap': [True, False], 
        # Weights of each rain rate class.
        # Either balance the whole training dataset before bootstrapping for a single tree (balanced),
        # or balance the training set provided for each tree (balanced_subsample).
        'class_weight': ['balanced', 'balanced_subsample'],
        # Maximum fraction of samples used for bootstrapped trees
        'max_samples': [0.0001, 0.0005, 0.001, 0.005, 0.01],
        # more can be added if desired
    },
    'tune_n_jobs': -2, # Use 2 less than the max number of cores avaialble. Allows overhead processing.
    'tune_n_iter': 0.04, # % of total as decimal    quick testing is 0.001, normal 0.05, higher is too much for JASMIN HPC (16-core, 256GB mem, 48h-runtime)
    'tune_cv': 2, # cross validation (train/test split)
    
    # Train
    'train_startdate': datetime.datetime(2015,1,1,0,0),
    'train_enddate': datetime.datetime(2019,12,31,23,45),
    'train_perc_keep': float(0.1), # Quick testing is 0.001, normal is 0.1
    'train_criterion': 'gini',
    'train_n_jobs': -2,
    
    # Verify
    'verify_startdate': datetime.datetime(2020,9,1,0,0),
    'verify_enddate': datetime.datetime(2021,8,31,23,45),
    'verify_perc_exclude': float(0.9) # Quick testing is 0.98, normal is 0.9
    
    
}


#########################
####### 2-EXPORT ########
#########################
pprint(RF_parameters)

file = open("parameters___"+RF_parameters["name"]+".pkl", "wb")
pickle.dump(RF_parameters, file)

print("parameters___"+RF_parameters["name"]+".pkl")
print("DONE!")