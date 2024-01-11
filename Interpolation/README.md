**Note:** See primary *README* file directory tree for script prerequisite scripts.

# Single/Simple Regressors

## Multiple Linear Regression Testing

**File Path(s):**
- ./Single_Regressor/Multiple_Linear_Regressor.csv

**Description:** Tests performance of Multiple Linear Regression model at predicting VWC with different topographic input variables.

**Output(s):**
- *./Single_Regressor/Output_Data/Multiple_Linear_Regressor_Results.csv* - Table of Input Parameters, R2 Score, RMSE, and MAE.

## Universal Krigging Regression Testing

**File Path(s):**
- ./Single_Regressor/Universal_Kriging.csv

**Description:** Tests performance of Universal Krigging, with and without external drift, models at predicting VWC with different topographic input variables.

**Output(s):**
- *./Single_Regressor/Output_Data/Krigging_Results.csv* - Table of Input Parameters, R2 Score, RMSE, and MAE.

## Other Simple Regression Testing

**File Path(s):**
- ./Single_Regressor/Other_Simple_Regressors.py

**Description:** Tests performance of a variety of simple regression models at predicting VWC with  different topographic input variables.

**Output(s):**
- *./Single_Regressor/Output_Data/Other_Simple_Regressor_Results.csv* - Table of Model, Input Parameters, R2 Score, RMSE, and MAE.

# Hybrid Models

## Classified Gradient Boosting Regression Testing
**File Path(s):**
- ./[Classifier_Directory]/Gradient_Boosting_Regressor.py

**Description:** Breaking up study area into topographically similar regions using a specified clusting model, tests the performance of using Gradient Boosting Regression for each region at predicting VWC with different topographic and cluster quantities.

**Output(s):**
- *./[Classifier_Directory]/Output_Data/Performance/Gradient_Boosting_Results.csv* - Table of Input Parameters, Cluster Quantity, R2 Score, RMSE, and MAE.
- *./[Classifier_Directory]/Output_Data/Predicted_Actual/Gradient_Boosting_Results.csv* - Table of Predicted and Actual VWC predictions for top performing model and parameters.


## Classified MLP Regression Testing
**File Path(s):**
- ./[Classifier_Directory]/MLP_Regressor.py

**Description:** Breaking up study area into topographically similar regions using a specified clusting model, tests the performance of using MLP Regression for each region at predicting VWC with different topographic and cluster quantities.

**Output(s):**
- *./[Classifier_Directory]/Output_Data/Performance/MLP_Regressor_Results.csv* - Table of Input Parameters, Cluster Quantity, R2 Score, RMSE, and MAE.
- *./[Classifier_Directory]/Output_Data/MLP_Regressor_Results.csv* - Table of Predicted and Actual VWC predictions for top performing model and parameters.


## Classified Random Forest Regression Testing
**File Path(s):**
- ./[Classifier_Directory]/Random_Forest_Regressor.py

**Description:** Breaking up study area into topographically similar regions using a specified clusting model, tests the performance of using Random Forest Regression for each region at predicting VWC with different topographic and cluster quantities.

**Output(s):**
- *./[Classifier_Directory]/Output_Data/Performance/Random_Forest_Regressor_Results.csv* - Table of Input Parameters, Cluster Quantity, R2 Score, RMSE, and MAE.
- *./[Classifier_Directory]/Output_Data/Predicted_Actual/Random_Forest_Regressor_Results.csv* - Table of Predicted and Actual VWC predictions for top performing model and parameters.


## Classified Support Vector Machine Regression Testing
**File Path(s):**
- ./[Classifier_Directory]/Support_Vector_Machine_Regressor.py

**Description:** Breaking up study area into topographically similar regions using a specified clusting model, tests the performance of using Support Vector Machine Regression for each region at predicting VWC with different topographic and cluster quantities.

**Output(s):**
- *./[Classifier_Directory]/Output_Data/Performance/Support_Vector_Machine_Results.csv* - Table of Input Parameters, Cluster Quantity, R2 Score, RMSE, and MAE.
- *./[Classifier_Directory]/Output_Data/Predicted_Actual/Support_Vector_Machine_Results.csv* - Table of Predicted and Actual VWC predictions for top performing model and parameters.


## Classified Universal Krigging Regression Testing
**File Path(s):**
- ./[Classifier_Directory]/Universal_Kriging_Regressor.py

**Description:** Breaking up study area into topographically similar regions using a specified clusting model, tests the performance of using Universal Krigging for each region at predicting VWC with different topographic and cluster quantities.

**Output(s):**

- *./[Classifier_Directory]/Output_Data/Performance/Universal_Kriging_Results.csv* - Table of Input Parameters, Cluster Quantity, R2 Score, RMSE, and MAE.
- *./[Classifier_Directory]/Output_Data/Predicted_Actual/Universal_Kriging_Results.csv* - Table of Predicted and Actual VWC predictions for top performing model and parameters.

## Hybrid Model Predicted vs Actual Scatter Plots
**File Path(s):**
- ./[Classifier_Directory]/Scatter_Plot.py

**Description:** Using the best predicted and actual VWC values produced by each hybrid model, generates predicted vs actual scatter plots for a specified clustering technique.

**Output(s):**
- ./[Classifier_Directory]/Output_Data/Predicted_Actual/Scatter_Plots/Support_Vector_Machine.tiff - See Desciption.
- ./[Classifier_Directory]/Output_Data/Predicted_Actual/Scatter_Plots/Universal_Kriging.tiff - See Desciption.
- ./[Classifier_Directory]/Output_Data/Predicted_Actual/Scatter_Plots/Gradient_Boosting.tiff - See Desciption.
- ./[Classifier_Directory]/Output_Data/Predicted_Actual/Scatter_Plots/MLP_Regressor.tiff - See Desciption.
- ./[Classifier_Directory]/Output_Data/Predicted_Actual/Scatter_Plots/Random_Forest_Regressor.tiff - See Desciption.

## Top Hybrid Model Tuning

**File Path(s):**
- ./Top_Models_Tuning/Gaussian_Mixture_Tuning.py
- ./Top_Models_Tuning/KMeans_Tuning.py

**Description:** Selecting only the best performing hybrid models with default hyperparameters and input variable parameters (R2 Score > .5), tests performance of different MLPRegressor shapes.

**Output(s):**
- *./Top_Models_Tuning/Output_Data/KMeans_Shape_Results.csv* - Table of Regressor Shape, Input Parameters, Cluster Quantity, R2 Score, RMSE, and MAE for top models with K-Means classification.
- *./Top_Models_Tuning/Output_Data/Gaussian_Mixture_Shape_Results.csv* - Table of Regressor Shape, Input Parameters, Cluster Quantity, R2 Score, RMSE, and MAE for top models with K-Means classification.

## Interpolation Heatmap Generation

**File Path(s):**
- ./Heatmap/Generate_Heatmap.py

**Description:** Using a hybrid model, K-Means classification with MLP Regression, with optimized hyperparameters, generates heatmap of VWC across the study area.

**Output(s):**
- *./Heatmap/Output_Data/Heatmap_Results.csv* - Table of Latitude, Longitude, and VWC predictions.
- ./Heatmap/Output_Data/Heatmap_Visualization.tiff - Plot of VWC predictions over study area on a color map.
- ./Heatmap/Output_Data/Heatmap_Overlay_Visualization.tiff - Plot of VWC predictions color map, overlayed on satellite image of study area.
- ./Heatmap/Output_Data/Heatmap_Overlay_Colorbar_Visualization.tiff - Plot of VWC predictions color map, overlayed on satellite image of study area with a color bar scale.

