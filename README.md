
## Set Up

**Windows**
1. Download and install Python 3 from https://www.python.org/downloads/.
2. Navigate to *./Set_Up/windows.bat* in File Explorer and double-click it to install necessary Python libraries.
3. Using Command Prompt, navigate to the repository Python file you are interested in executing, and run *python3 [python file name]* to execute it.

**Mac**
1. Install HomeBrew by running */bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"* in Terminal and following the prompts.
2. In Terminal, run *brew install python*.
3. Navigate to *./Set_Up/unix.sh* in Terminal and run *sh unix.sh* to install necessary Pythong libraries.
4. Using Terminal, navigate to the repository Python file you are interested in executing, and run *python3 [python file name]* to execute it.


## File Directory

Files are presented in a roughly sequential order, representitive of research progress.

**File Organization**
- Input_Data Directories - Stores data files to-be inputted into Python scripts.
- Python Files - Execute respective operation.
- Output_Data Directories - Locations for Python scripts to save their outputs.

**Note:** Some scripts have prerequisites (pr) which must ran prior to their own execution as they rely on outputs produced by the prerequisite scipts.\
**Note:** Input_Data directories' contents ommited for clarity
**Note:** Details of Output_Data directories' content elaborated on in internal *README* files.

```
.
├── Set_Up/
│   ├── unix.sh
│   └── windows.bat
├── Interpolation/
│   ├── Single_Regressor/
│   │   ├── Multiple_Linear_Regressor.py
│   │   ├── Other_Simple_Regressors.py
│   │   ├── Universal_Krigging.py
│   │   ├── output_data/
│   │   └── input_data/
│   └── Hybrid_Model/
│       ├── Agglomerative_Clustering/
│       │   ├── 1 Gradient_Boosting_Regressor.py
│       │   ├── 1 MLP_Regressor.py
│       │   ├── 1 Random_Forest_Regressor.py
│       │   ├── 1 Support_Vector_Machine_Regressor.py
│       │   ├── 1 Universal_Kriging_Regressor.py
│       │   ├── Scatter_Plot.py (pr:1)
│       │   └── Output_Data/
│       │       ├── Performance/
│       │       └── Predicted_Actual/
│       │           └── Scatter_Plots/
│       ├── DBSCAN_Clustering/
│       │   ├── 2 Gradient_Boosting_Regressor.py
│       │   ├── 2 MLP_Regressor.py
│       │   ├── 2 Random_Forest_Regressor.py
│       │   ├── 2 Support_Vector_Machine_Regressor.py
│       │   ├── 2 Universal_Kriging_Regressor.py
│       │   ├── Scatter_Plot.py (pr:2)
│       │   └── Output_Data/
│       │       ├── Performance/
│       │       └── Predicted_Actual/
│       │           └── Scatter_Plots/
│       ├── Gussian_Mixture_Clustering/
│       │   ├── 3 Gradient_Boosting_Regressor.py
│       │   ├── 3 MLP_Regressor.py
│       │   ├── 3 Random_Forest_Regressor.py
│       │   ├── 3 Support_Vector_Machine_Regressor.py
│       │   ├── 3 Universal_Kriging_Regressor.py
│       │   ├── Scatter_Plot.py (pr:3)
│       │   └── Output_Data/
│       │       ├── Performance/
│       │       └── Predicted_Actual/
│       │           └── Scatter_Plots/
│       ├── KMeans_Clustering/
│       │   ├── 4 Gradient_Boosting_Regressor.py
│       │   ├── 4 MLP_Regressor.py
│       │   ├── 4 Random_Forest_Regressor.py
│       │   ├── 4 Support_Vector_Machine_Regressor.py
│       │   ├── 4 Universal_Kriging_Regressor.py
│       │   ├── Scatter_Plot.py (pr:4)
│       │   └── Output_Data/
│       │       ├── Performance/
│       │       └── Predicted_Actual/
│       │           └── Scatter_Plots/
│       ├── Mean_Shift_Clustering/
│       │   ├── 5 Gradient_Boosting_Regressor.py
│       │   ├── 5 MLP_Regressor.py
│       │   ├── 5 Random_Forest_Regressor.py
│       │   ├── 5 Support_Vector_Machine_Regressor.py
│       │   ├── 5 Universal_Kriging_Regressor.py
│       │   ├── Scatter_Plot.py (pr:5)
│       │   └── Output_Data/
│       │       ├── Performance/
│       │       └── Predicted_Actual/
│       │           └── Scatter_Plots/
│       ├── Top_Models_Tuning/
│       │   ├── Gaussian_Mixture_Tuning.py
│       │   ├── KMean_Tuning.py
│       │   └── Output_Data/
│       ├── Heatmap/
│       │   ├── 6 Generate_Heatmap.py
│       │   └── Output_Data/
│       └── README.md
├── Extrapolation/
│   ├── Heatmaps/
│   │   ├── 6 Single_Day_Heatmap.py
│   │   ├── Multi_Day_Validation.py
│   │   ├── Multi_Day_Heatmap.py
│   │   └── Output_Data/
│   │       ├── Single_Day/
│   │       └── Multi_Day/
│   │           ├── Validation/
│   │           └── Maps/
│   ├── Extrapolation_Interpolation_Comparison/
│   │   ├── Map_Comparer.py (pr:6)
│   │   └── Output_Data/
│   ├── Input_Data/
│   └── README.md
├── Raster/
│   ├── Terrain_Attributes.zip
│   ├── Texture.zip
│   └── NDVI.zip
└── README.md
```

## Input Data Files
- **in-situ_Data.csv** - Manually samples Volumetric Water Content values across the study area.
- **Map_Data.csv** - Topographic information for each geo-pixel of the study area.
- **WUSN_Locations.csv** - Locations of each of the VWC sensors in the study area.
- **SingleDay_WUSN_Data.csv** - Volumetric Water Content samples collected by sensor network on same day as in-situ samples.
- **MultiDay_WUSN_Data.csv** - Volumetric Water Content samples collected by sensor network across a precipitation event.