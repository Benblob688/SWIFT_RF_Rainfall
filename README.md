# SWIFT_RF_Rainfall
Python scripts to tune, train, verify and apply random forest models for MSG-based rainfall retrieval over Africa.

(insert image of final published product output working here)

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
`RF_make_model_parameters.py` creates a dictionary of parameters such as paths to different directories, amount of tune, train and verify data, date ranges for tune, train, verify, and more exaplined within or in the 'scikit-learn' documentation.