import numpy as np
import matplotlib.pyplot as plt
import pandas

extrapolation = pandas.read_csv("../Heatmaps/Output_Data/Single_Day/Heatmap_Results.csv").to_numpy()[:,3]
interpolation = pandas.read_csv("../../Interpolation/Hybrid_Model/Heatmap/Output_Data/Heatmap_Results.csv").to_numpy()[:,3]

plt.scatter(extrapolation, interpolation)
plt.title("Extrapolation vs Interpolation VWC")
plt.text(0.95, 0.05, 'Correlation Coef. : ' + str(np.corrcoef(interpolation, extrapolation)[0,1]), horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
plt.savefig("./Output_Data/Extrapolation_vs_Interpolation.tiff", format="tiff")
plt.close()

map_df = pandas.read_csv("../Input_Data/Map_Data.csv")
cord_in = map_df[["POINT_X", "POINT_Y"]].to_numpy()
ext_in = map_df[["DEM", "PlnCurv", "TWI"]].to_numpy()

scaledX = np.ceil((cord_in[:,0] - min(cord_in[:,0]))/9.4)
scaledY = np.ceil((cord_in[:,1] - min(cord_in[:,1]))/9.4)
scaledCoord = np.array([scaledX, scaledY]).T

predictions = []
heatmap = np.full((int(max(scaledY) + 1), int(max(scaledX) + 1)), np.NaN)
for i in range(0, len(scaledX)):
    heatmap[int(scaledY[i]), int(scaledX[i])] = extrapolation[i] - interpolation[i]
    predictions.append(extrapolation[i] - interpolation[i])

np_heatmap = np.array(heatmap)
cords = map_df[["POINT_X", "POINT_Y"]].to_numpy()
output_dataframe = pandas.DataFrame({
    "Latitude": cords[:,0],
    "Logitude": cords[:,1],
    "VWC": predictions
})

output_dataframe.to_csv("./Output_Data/Difference_Heatmap_Results.csv")

plt.imshow(np.flipud(heatmap), interpolation='none', vmax=-.25, vmin=.25)
plt.colorbar()
plt.title("Difference Extrapolation vs Interpolation (Ext - Int)")
plt.savefig("./Output_Data/Difference_Heatmap_Visual.tiff", format="tiff")
plt.close()