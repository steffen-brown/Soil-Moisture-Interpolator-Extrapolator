# Heatmaps

## Single Day VWC Extrapolation Heatmap Generation

**File Path(s):**
- ./Heatmaps/Single_Day_Heatmap.py

**Description:** Generates VWC extrapolation heatmap using WUSN data and a transferred interpolation model for the same day as the manually collected VWC samples were taken.

**Output(s):**
- *./Heatmaps/Output_Data/Single_Day/Heatmap_Results.csv* - Table of Latitude, Longitude, and VWC predictions.
- *./Heatmaps/Output_Data/Single_Day/Heatmap_Visualization.tiff* - Plot of VWC predictions over study area on a color map.
- *./Heatmaps/Output_Data/Single_Day/Heatmap_Overlay_Visualization.tiff* - Plot of VWC predictions color map, overlayed on satellite image of study area.
- *./Heatmaps/Output_Data/Single_Day/Heatmap_Overlay_Colorbar_Visualization.tiff* - Plot of VWC predictions color map, overlayed on satellite image of study area with a color bar scale.

## Multiple Day VWC Extrapolation

**File Path(s):**
- ./Heatmaps/Mutli_Day_Validation.py

**Description:**  Taking WUSN datapoints from a series of days, chooses training points (inside WUSN range) and testing points (on border of WUSN range) for extrapolation model to evaluate its performance.

**Output(s):**
- *./Heatmaps/Output_Data/Multi_Day/Validation/Preformance.csv* - Table of extrapolation MLPRegressor Size, R2 Score, RMSE, and MAE.
- *./Heatmaps/Output_Data/Multi_Day/Validation/Predicted_Actual.csv* - Table of predicted and actual VWC values for best extrapolation model shape.

## Multi-Day VWC Extrapolation Heatmap Generation

**File Path(s):**
- *./Heatmaps/Multi_Day_Heatmap.py*

**Description:** Generates a series of VWC extrapolation heatmaps using WUSN data and a transferred interpolation model, trained on the manually collected VWC samples.

**Output(s):**
- *./Heatmaps/Output_Data/Multi_Day/Maps/Heatmap_Results_[Date].csv* - Table of Latitude, Longitude, and VWC predictions.
- *./Heatmaps/Output_Data/Multi_Day/Maps/Heatmap_Visualization_[Date].tiff* - Plot of VWC predictions over study area on a color map.
- *./Heatmaps/Output_Data/Multi_Day/Maps/Heatmap_Overlay_Visualization_[Date].tiff* - Plot of VWC predictions color map, overlayed on satellite image of study area.
- *./Heatmaps/Output_Data/Multi_Day/Maps/Heatmap_Overlay_Colorbar_Visualization_[Date].tiff* - Plot of VWC predictions color map, overlayed on satellite image of study area with a color bar scale.

# Extrapolation vs Interpolation Map Comparison

## Extrapolation VWC Heatmap, Interpolation VWC Heatmap Comparitor

**File Path(s):**
- *./Extrapolation_Interpolation_Comparison/Map_Comparer.py*

**Description:** Compares each geo-pixel's VWC between the interpolation heatmap and extrapolation heatmap on the day of manual datapoint collection.

**Output(s):**
- *./Extrapolation_Interpolation_Comparison/Output_Data/Difference_Heatmap_Results.csv* - Table of Latitude, Longitude, and VWC Difference Between Maps.
- *./Extrapolation_Interpolation_Comparison/Output_Data/Extrapolation_vs_Interpolation.tiff* - Scatter plot of extrapolation VWC prediction and interpolation VWC predction per geo-pixel.
- *./Extrapolation_Interpolation_Comparison/Output_Data/Difference_Heatmap_Visual.tiff* - Map visalization of difference in extrapolation VWC prediction and interpolation VWC prediction (Map Value = Extrapolation - Interpolation).