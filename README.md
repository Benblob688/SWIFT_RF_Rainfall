# SWIFT_RF_Rainfall
Python scripts to tune, train, verify and apply random forest (RF) models for MSG-based rainfall retrieval over Africa.

[insert image of final published product output working here]
![An example of the SWIFT RF Rainfall algorithm output](https://github.com/Benblob688/SWIFT_RF_Rainfall/raw/main/RF_rainfall_20180628_1630.png)

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
The tuning step uses brute force to test different hyperparameters of the RF model, and ranks the input features by importance. This step requires the most human input, as there is little automation in the choise of feature importances. If any step were to be improved in future versions, tuning would be top of the list of priorities. The fraction of total possible hyperparameter settings is very small due to computational requirements, however the distribution of OOB scores reaches a plataeu, signaling that the gains from increasing the number of hyperparameter combinations is likely to be small. This may differ for new domains, label datasets, and input features.

`RF_settings` contains some examples of the output from the tuning script. These are the outputs for the methodology in Pickering et al. (2022) that assisted with the subjective determination of final input features used. Inside is a `.csv` of input features and their impurity-decrease measured importance, ranked. A `.png` image showing the same data also exists. Finally a `.pkl` pickle file contains a dictionary of the best hyperparameter combination and the chosen input features to be used in training. Ideally the features should be chosen first, and then the hyperparameter tuning should take place. This was done manually in Pickering et al. (2022).

`RF_tune_model_functions.py` contains all the necessary functions for running the tuning script. This includes many duplicates in the training and verification function scripts, but is included here to keep each step separate.

`RF_tune_model.py` takes one argument when called in the terminal: the name of the model. This is set in the `0_model_parameters` step, and is used for both ingesting the parameters file from the 0th step, and also for exporting the three files in `RF_settings`, where the model name is used in the filename.

# 3. Train Model


