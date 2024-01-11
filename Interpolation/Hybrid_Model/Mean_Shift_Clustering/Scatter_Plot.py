import matplotlib.pyplot as plt
import pandas
import numpy as np

regression = ["Gradient_Boosting", "MLP_Regressor", "Random_Forest_Regressor", "Support_Vector_Machine", "Universal_Kriging"]
regression_label = ["Gradient Boosting", "MLP Regressor", "Random Forest Regressor", "Support Vector Machine", "Universal Kriging"]

for r, l in zip(regression, regression_label):
    data_file = "./Output_Data/Predicted_Actual/" + r + "_Results.csv"
    data_frame = pandas.read_csv(data_file)

    predicted_data = data_frame['Predicted'].to_numpy().astype(np.float64)
    actual_data = data_frame['Actual'].to_numpy().astype(np.float64)

    plt.scatter(predicted_data, actual_data, c="red", label=l)
    slope, intercept = np.polyfit(predicted_data, actual_data, 1)
    plt.plot(np.array(np.arange(.25, .5+.0125, .0125)), slope * np.arange(.25, .5+.0125, .0125) + intercept, linewidth=2)

    classification_label = "Mean Shift Clustering"

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Actual vs. Predicted Volumetric Water Content with\n" + classification_label + " Classification and " + l)
    plt.xlim(.25, .5)
    plt.ylim(.25, .5)
    plt.legend()

    output_file = './Output_Data/Predicted_Actual/Scatter_Plots/' + r
    plt.savefig(output_file+".tiff", format='tiff', dpi=300)

    plt.clf()
