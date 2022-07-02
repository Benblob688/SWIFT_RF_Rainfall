README.md

# SWIFT_RF_Rainfall
[![DOI](https://zenodo.org/badge/461589905.svg)](https://zenodo.org/badge/latestdoi/461589905)
[![release](https://img.shields.io/badge/release-v1.0.0-blue)](https://github.com/Benblob688/SWIFT_RF_Rainfall/releases/tag/v1.0.0)

Python scripts to tune, train, verify and apply random forest (RF) models for MSG-based rainfall retrieval over Africa.

[insert image of final published product output working here]
![An example of the SWIFT RF Rainfall algorithm output](https://github.com/Benblob688/SWIFT_RF_Rainfall/raw/main/RF_rainfall_20180628_1630.png)

# Prerequisites
This package has been developed and tested entirely on the [JASMIN compute cluster](https://jasmin.ac.uk/), within the SWIFT Group Workspace (GWS). If operating outside of the SWIFT GWS on JASMIN, there are a number of other dependancies that are required for these scripts to function. A Python environment file from `conda` is included which is not OS-specific. Setting up a new Python environment for this package is recommended. 
If you already recieve real-time SEVIRI images or NWCSAF CRR products through a satellite dish, you will have these dependencies already. Completely new users of NWCSAF products will have to set these up and gain a license from EUMETSAT.
- Spinning Enhanced Visible and InfraRed Imager (SEVIRI) satellite files in `HRIT` format. These can be recieved directly with a satellite dish (see [Roberts et al. 2021](https://doi.org/10.1002/wea.3936)) or by downloading them from [EUMETSEAT](https://navigator.eumetsat.int/product/EO:EUM:DAT:MSG:HRSEVIRI) under "By format > HRIT" where the format is explained (a file for each channel plus EPI and PRO files demarking the start and end of an observing window) as well as links to "Get Access". Either method requires a login/license with EUMETSAT.
- [xRITDecompress](https://gitlab.eumetsat.int/open-source/PublicDecompWT) should be installed to be able to decompress the `HRIT` files.

SEVIRI data in NetCDF and other formats could be supported, but the scripts given here must then be modified by the user. At some point the satellite images are all in `numpy` arrays which is a common overlap point in the scripts for anyone who wishes to modify the scripts to support other data formats.

# Summary
There are 6 steps to creating a custom version of the algorithm published in Pickering et al., 2022.
|    | Step             | _Description_ |
| -- | ---------------- | ------------- |
| 0. | Model Parameters | _high-level settings used commonly throughout the process_ |
| 1. | Training Data    | _create appropriate training data for the model_ |
| 2. | Tune Model       | _find the best-performing hyperparameters and features for the model on a small subset of training data_ |
| 3. | Train Model      | _use the best-performing hyperparameters and features from the tuning step and train on a large dataset_ |
| 4. | Verify Model     | _check the performance of the model against exisitng rainfall products_ |
| 5. | Apply Model      | _use the trained model to produce maps of rainfall rate_ |

# 0. Model Parameters
`RF_make_model_parameters.py` creates a dictionary of parameters such as paths to different directories, amount of tune, train and verify data, date ranges for tune, train, verify, and more exaplined within or in the `scikit-learn` documentation.

# 1. Training Data
This is where a good portion of time will be spent when developing a new RF model. All training data must be provided on the same resolution spatial grid, at the same time resolution. In Pickering et al. (2022), a 0.1º equirectangular grid @15-min intervals was used, to match GPM IMERG Final Run V06B. Any spatiotemporal resolution, domain size and input features can be implemented by a user, with the standard caveats about random forests:
- RFs have poor performance if too many features are provided, as it lowers the liklihood of the best feature being used.
- RFs have bias toward high cardinality features (continuous rather than discrete). Binary features are less likely to be use and will have lower importance.
- Other characterisations not covered in this list.

The `topography` directory contains the data supplied to the RF in Pickering et al. (2022), valid only on the 0.1º resolution domain with `min_lat=-35`, `max_lat=38`, `min_lon=-20`, and `max_lon=52`.

`interp_coeffs_SEVIRI_to_GPM.pickle` is a pickle file containing a list of interpolation coefficients that may be used for bilinear interpolation of SEVIRI data onto the desired domain grid. This saves time in generating training data as the coefficients would need to be recalculated each time a regridding occurs (every timestep of the desired training data).

`RF_generate_data_functions.py` is imported by the generate data scripts, and contains all the code used to decompress, regrid and manipulate SEVIRI channel data from the SWIFT group workspace on JASMIN (although the source location can be changed by a user to work on another system), as well as ingesting GPM IMERG final run V06B data from the Centre for Environmental Data Analysis (CEDA; this can also be changed by a user if another source for the 'truth' label data is being used in training). The script also contains a list of corrupt files in the source directory (default list is for the JASMIN SWIFT GWS). A "badlist" should also be implemented if the user decides to change the data source.

`RF_monthly_generate_data.py` takes a year and month, then cycles through each 15 minute period within. It searches for the necessary SEVIRI and IMERG files, regrids the SEVIRI data to match IMERG, and pairs IMERG pixels with all the SEVIRI channels valid at the same time in a Pandas dataframe. To reduce storage requirements, pixels with 10.8 µm brightness temperature greater than 273.15 K are discarded, since these make up > 95% of the pixels and add no value in training a precipitation algorithm. The filtered training data is then exported as a Pandas dataframe in a pickle file ready to be ingested by the later scripts.

`RF_single_generate_data.py` is identical to `RF_monthly_generate_data.py` but takes a specific year, month, day, hour and minute, creating only one output pickle file.

# 2. Tune Model
The tuning step uses brute force to test different hyperparameters of the RF model, and ranks the input features by importance. This step requires the most human input, as there is little automation in the choice of feature importances. If any step were to be improved in future versions, tuning would be top of the list of priorities. The fraction of total possible hyperparameter settings is very small due to computational requirements, however the distribution of OOB scores reaches a plataeu, signaling that the gains from increasing the number of hyperparameter combinations is likely to be small. This may differ for new domains, label datasets, and input features.

`RF_settings` contains some examples of the output from the tuning script. These are the outputs for the methodology in Pickering et al. (2022) that assisted with the subjective determination of final input features used. Inside is a `.csv` of input features and their impurity-decrease measured importance, ranked. A `.png` image showing the same data also exists. Finally a `.pkl` pickle file contains a dictionary of the best hyperparameter combination and the chosen input features to be used in training. Ideally the features should be chosen first, and then the hyperparameter tuning should take place. This was done manually in Pickering et al. 2022 (in preparation).

`RF_tune_model_functions.py` contains all the necessary functions for running the tuning script. This includes many duplicates in the training and verification function scripts, but is included here to keep each step separate.

`RF_tune_model.py` takes one argument when called in the terminal: the path to the `0_model_parameters` file. The parameters file contains a dictionary with all parameters inside. The script takes all the information it needs from that parameters dictionary.

# 3. Train Model
Training the random forest model uses the best hyperparameters found in the `2_tune_model` step, and by defualt uses a larger sample set of training data than in the tune step. The model is trained and then saved. The labels of the features are kept in the model object under `rf_object.feature_names_in_`.

`RF_models` is where trained RF models should go. One example is included from the publication Pickering et al. 2022 (in preparation).

`RF_train_model_functions.py` contains all the necessary functions for running the training script. This includes many duplicates in the tuning and verification function scripts, but is included here to keep each step separate.

`RF_train_model.py` takes one argument when called in the terminal: the path to the `0_model_parameters` file. The parameters file contains a dictionary with all parameters inside. The script takes all the information it needs from that parameters dictionary.


# 4. Verify Model
The verification step takes one or multiple models and verifies them. Only the parameters file need to be passes in the command line, but the paths to the models you wish to have verified must be editied in the python script itself at this stage. There is a dictionary containing paths to each model you wish to have verified. The reason for this is that the model training steps so far have been independent of each model. At the verification step, multiple models saves importing the verification sample data multiple times, and allows plotting of the verification results between models on a single plot.

`RF_verification_functions.py` contains all the necessary functions for running the verification script. This includes many duplicates in the tuning and training function scripts, but is included here to keep each step separate. This script contains all of the statistical codes and plotting codes for the verification script.

`RF_model_verification_script.py` takes one argument when called in the terminal: the path to the `0_model_parameters` file. The parameters file contains a dictionary with all parameters inside. The script takes all the information it needs from that parameters dictionary. Models to be verifid must be specified in the script itself. Multiple can be entered into the `modelpaths` dictionary at the top of the script (line 81).
In this script, comparison rainfall datasets are used. Currently, the two supported datasets are the NWCSAF CRR and CRR-Ph products. These have been regrided to 0.1º to match the radnom forests created in Pickering et al. 2022 (in preparation), but any rainfall dataset could be added in with some modification. So long as the data are on the same grid and temporal resolution as the other products being verified, against GPM, then the verification is valid. Some tinkering with the code will be required for this.
Many files are output, so it is recommended to make a unique directory within `/4_verify_model/output` for each run of the verification script. Outputs include:
- Statistics summary table. A `.csv` with standard statistics about each model or product supplied.
- Time-based statistics. Line plots of various statistical metrics plotted on line graphs over:
    - 3-hourly periods of the day (in Zulu time).
    - months of the year.
- Location-based statistics. Maps of Africa with veification done on 1º (10x10 pixels of each product) grid, with overall dimensions of 73 x 72 pixels. Various statistical metrics are shown.
- Precipitation rate-based statistics. Grids of each product versus GPM precipitation for each precipitation rate class. Hit score is included above, and a comparative climatology is shown below.
- Performance diagrams. These statistical plots combine several metrics into one plot, and both different products, and different precipitation rate classes, are shown simultaneously. A zoomed in version of this plot is also made, since most data points are in the lower left 10% of the standard plot.
- Importances. A bar chart of features given to each model, with their relative importance based on gini-impurity reduction. Better methods of calculating relative importance such as permutation are avaialble but not implemented here.
- Feature thresholds. These heatmaps show what values of each feature are used to split samples in the trees of the forest. For example, in Pickering et al. 2022 (in preparation), it was found that the latitude boundary between the Sahel and the Sahara was used more frequently than other latitudes for splitting (the hypothesis being that the trained model noticed that less rainfall fell north of this boundary).
- Decision tree. The first tree of the forest in the model is plotted. Note that this is usually plotted with lots of overlap if the forest has more than 20 nodes, making it unhelpful in most cases.

The verification can quickly overwhelm system resources during the plot_maps step. The time period (`start` and `end`) for verification should differ than that of `tune` and `train`. However, some overlap is possible because the variable `perc_keep` used in the tuning and training step is instead flipped here, and `perc_exclude` is used. This means that if `perc_keep` in earlier steps and `perc_exclude` in the verification step, are both less than `0.5`, the scripts will not use the same data. The use of `np.random.seed` fixed (defualt=42) means the results are repeatable in nature with random-like behaviour.

# 5. Apply Model
Only one python notebook exists here as a demonstration of how to use a trained model to plot rainfall. Current SEVIRI data can be used or older SEVIRI data, just put in a date and the function script (imported at the top of the notebook) will source the file from where you are storing your files. Bear in mind you need an archive of HRIT-format SEVIRI files, and `xRITDecompress` as described in the prerequisites above. Existing users of NWCSAF products do have these both running already. You will need to edit all the scripts in this module to source files from your own HRIT archive. The default is assuming that you are running this code on the JASMIN HPC platform.

This demo `apply_model` code can then be turned into operational scripts to plot and save images of derived rainfall as required for new incoming satellite data in near real time.
