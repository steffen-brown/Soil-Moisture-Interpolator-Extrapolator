import numpy as np
from sklearn.cluster import KMeans
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import gaussian_filter

def KMeansClassifiedNN_predict(input_external, input_coordinates):
    predictor_rows = []
    for r in range(0, len(input_external)):
        predictor_rows.append([input_coordinates[r, 0], input_coordinates[r,1], np.NaN, input_external[r,0], np.NaN, np.NaN, input_external[r,1], np.NaN, input_external[r,2], np.NaN, np.NaN])
    predictor_rows = np.array(predictor_rows)

    df = pandas.read_csv("Final.csv")
    columns = list(df.columns.values)
    scaler = StandardScaler()
    npdf = scaler.fit(np.vstack((df.to_numpy(), predictor_rows))).transform(np.vstack((df.to_numpy(), predictor_rows)))
    df = pandas.DataFrame(npdf, columns=columns)
    print(df)

    np.random.seed(22)
    np.random.set_state(np.random.get_state())

    response_variable = np.squeeze(df[["VWC"]].to_numpy())
    external_variable = df[["DEM_QGIS", "PlnCurv", "TWI"]].to_numpy()
    coordinates = df[["X", "Y"]].to_numpy()

    num_classes = 7
    kmeans = KMeans(n_clusters=num_classes, random_state=23)
    labels = kmeans.fit_predict(np.hstack((external_variable, coordinates)))
    print(labels[12000])

    trainer_labels = labels[:64]
    predictor_labels = labels[64:]

    np.random.seed(22)
    np.random.set_state(np.random.get_state())
    response_variable_train, _, external_variable_train, _, coordinates_train, _, labels_train, _ = train_test_split(response_variable[:64], external_variable[:64], coordinates[:64], trainer_labels, test_size=.01)

    NN_models = []

    for class_label in np.unique(labels_train):
        class_idx = np.where(labels_train == class_label)[0]
        class_response = response_variable_train[class_idx]
        class_external = external_variable_train[class_idx]
        class_corrdinates = coordinates_train[class_idx]

        np.random.seed(22)
        np.random.set_state(np.random.get_state())
        NN = MLPRegressor(hidden_layer_sizes=(5,100))
        NN.fit(np.hstack((class_external, class_corrdinates)), class_response)

        NN_models.append(NN)

    predictions = []

    external_variable = external_variable[64:, :]
    coordinates = coordinates[64:, :]

    for i in range(0,len(predictor_labels)):
        catagory = predictor_labels[i]
        NN = NN_models[catagory]

        prediction = NN.predict(np.array([np.hstack((external_variable[i,:], coordinates[i,:]))]))

        predictions.append(prediction[0])

    mean = scaler.mean_[2]
    std = scaler.scale_[2]

    scaled_predicted = np.array(predictions) * std + mean

    return scaled_predicted, predictor_labels

map_df = pandas.read_csv("mapData.csv")
cord_in = map_df[["POINT_X", "POINT_Y"]].to_numpy()
ext_in = map_df[["DEM", "PlnCurv", "TWI"]].to_numpy()

scaledX = np.ceil((cord_in[:,0] - min(cord_in[:,0]))/9.4)
scaledY = np.ceil((cord_in[:,1] - min(cord_in[:,1]))/9.4)
scaledCoord = np.array([scaledX, scaledY]).T

pixels, classes = KMeansClassifiedNN_predict(ext_in, cord_in)

heatmap = []

for y in reversed(range(int(scaledY.min()), int(scaledY.max()))):
    row = []
    for x in range(int(scaledX.min()), int(scaledX.max())):
        try:
            index = scaledCoord.tolist().index([x,y])
            row.append(pixels[index])
        except:
            row.append(np.NaN)
    heatmap.append(row)

np_heatmap = np.array(heatmap)


xmin = 895169.7597  # Replace with the appropriate value in your coordinate system
ymin = 4644761.831  # Replace with the appropriate value in your coordinate system
pixel_width = 9  # Each pixel represents 9 meters in the x-direction
pixel_height = 9  # Each pixel represents 9 meters in the y-direction

crs = rasterio.crs.CRS.from_epsg(32615)  # UTM Zone 15N

# Define transformation from UTM coordinates to pixel grid
transform = rasterio.transform.from_origin(xmin, ymin, pixel_width, pixel_height)

# Save as GeoTIFF
output_path = "VWC_Heatmap.tif"
with rasterio.open(output_path, 'w', driver='GTiff', height=np_heatmap.shape[0], width=np_heatmap.shape[1], count=1, dtype=np_heatmap.dtype, crs=crs, transform=transform) as dst:
    dst.write(np_heatmap, 1)

# Display the original and smoothed heatmaps
plt.subplot(1, 2, 1)
plt.imshow(np_heatmap, interpolation='none')
plt.title('Original Heatmap')

plt.subplot(1, 2, 2)
plt.imshow(gaussian_filter(np_heatmap, sigma=1), interpolation='none')
plt.title('Smoothed Heatmap')

plt.tight_layout()
plt.colorbar()
plt.title("VWC")
plt.show()

plt.imshow(np_heatmap, cmap='gray', interpolation='none')

# Optionally, remove the axes for a cleaner image
plt.axis('off')

# Save the Heatmap as a PNG file
plt.savefig("heatmap_original_grayscale.png", bbox_inches='tight', pad_inches=0, dpi=300)

# heatmap_array = np.array(heatmap)
# plt.imshow(heatmap_array)  # You can adjust the colormap ('hot' in this case) to your preference
# plt.axis('off')  # Turn off the axis
# plt.title("")   # Set an empty title
# plt.colorbar()
# plt.colorbar().remove()  # Remove the colorbar

# # Save the heatmap as a TIFF image
# output_path = "VWC_Heatmap.tiff"
# plt.savefig(output_path, bbox_inches='tight', pad_inches=0, format='tiff', dpi=300)  # Adjust dpi as needed
# plt.close()

# # Optional: Open the TIFF image and display it
# heatmap_image = Image.open(output_path)
# heatmap_image.show()