import numpy as np
from sklearn.cluster import KMeans
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import utm
import folium
import io
from PIL import Image
import matplotlib as mpl
import matplotlib.colors as mcolors

import warnings
warnings.filterwarnings("ignore")

def KMeansClassifiedNN_predict(input_external, input_coordinates):
    predictor_rows = []
    for r in range(0, len(input_external)):
        predictor_rows.append([input_coordinates[r, 0], input_coordinates[r,1], np.NaN, input_external[r,0], np.NaN, np.NaN, input_external[r,1], np.NaN, input_external[r,2], np.NaN, np.NaN])
    predictor_rows = np.array(predictor_rows)

    df = pandas.read_csv("../Input_Data/in-situ_Data.csv")
    columns = list(df.columns.values)
    scaler = StandardScaler()
    npdf = scaler.fit(np.vstack((df.to_numpy(), predictor_rows))).transform(np.vstack((df.to_numpy(), predictor_rows)))
    df = pandas.DataFrame(npdf, columns=columns)

    np.random.seed(22)
    np.random.set_state(np.random.get_state())

    response_variable = np.squeeze(df[["VWC"]].to_numpy())
    external_variable = df[["DEM_QGIS", "PlnCurv", "TWI"]].to_numpy()
    coordinates = df[["X", "Y"]].to_numpy()

    num_classes = 7
    kmeans = KMeans(n_clusters=num_classes, random_state=23)
    labels = kmeans.fit_predict(np.hstack((external_variable, coordinates)))

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

print("Generating Heatmap", end=" ", flush=True)
map_df = pandas.read_csv("../Input_Data/Map_Data.csv")
cord_in = map_df[["POINT_X", "POINT_Y"]].to_numpy()
ext_in = map_df[["DEM", "PlnCurv", "TWI"]].to_numpy()

scaledX = np.ceil((cord_in[:,0] - min(cord_in[:,0]))/9.4)
scaledY = np.ceil((cord_in[:,1] - min(cord_in[:,1]))/9.4)
scaledCoord = np.array([scaledX, scaledY]).T

pixels, classes = KMeansClassifiedNN_predict(ext_in, cord_in)

heatmap = []

for y in reversed(range(int(scaledY.min()), int(scaledY.max()))):
    print(".", end=" ", flush=True)
    row = []
    for x in range(int(scaledX.min()), int(scaledX.max())):
        try:
            index = scaledCoord.tolist().index([x,y])
            row.append(pixels[index])
        except:
            row.append(np.NaN)
    heatmap.append(row)

np_heatmap = np.array(heatmap)
cords = map_df[["POINT_X", "POINT_Y"]].to_numpy()
output_dataframe = pandas.DataFrame({
    "Latitude": cords[:,0],
    "Logitude": cords[:,1],
    "VWC": pixels
})

output_dataframe.to_csv("./Output_Data/Heatmap_Results.csv")

print("\nGenerating Heatmap Visualization")
# Display the original and smoothed heatmaps
plt.imshow(np_heatmap, interpolation='none')
plt.title('Original Heatmap')

plt.tight_layout()
plt.colorbar()
plt.title("VWC")
# Save the Heatmap as a PNG file
plt.savefig("./Output_Data/Heatmap_Visualization.tiff", format="tiff")
plt.close()

#//////////

print("Generating Heatmap Overlay Visualization")

def normalize_values(values, min_val, max_val):
    return [(v - min_val) / (max_val - min_val) for v in values]

def get_hex_colors(values, cmap_name='viridis'):
    colormap = plt.get_cmap(cmap_name)
    normalized_values = normalize_values(values, .2, .45)
    return [mcolors.to_hex(colormap(v)) for v in normalized_values]

cord_left = map_df[["POINT_X", "POINT_Y"]].to_numpy()

pixel_width = 9.3993

cord_left = pandas.read_csv("../Input_Data/Map_Data.csv")[["POINT_X", "POINT_Y"]].to_numpy()
cord_right = []

for c in cord_left:
    coverted_cord_right = utm.to_latlon(c[0]+pixel_width,c[1]-pixel_width, 15, northern=True)
    cord_right.append([coverted_cord_right[0], coverted_cord_right[1]])

    coverted_cord_left = utm.to_latlon(c[0],c[1], 15, northern=True)
    c[0] = coverted_cord_left[0]
    c[1] = coverted_cord_left[1]

map = folium.Map(location=[41.859811, -88.228508], zoom_start=16)
colors = get_hex_colors(pixels)

for i in range(0, len(cord_left)):
    folium.Rectangle([(cord_left[i][0], cord_left[i][1]), (cord_right[i][0], cord_right[i][1])], weight=0, fill_color=colors[i], fill_opacity=.5).add_to(map)



tile = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
       ).add_to(map)

img_data = map._to_png(5)
img = Image.open(io.BytesIO(img_data))
img.save('./Output_Data/Heatmap_Overlay_Visualization.tiff', format="tiff")


image_path = './Output_Data/Heatmap_Overlay_Visualization.tiff'  # Replace with actual image path
img = Image.open(image_path)

# Convert the PIL image to an array
img_array = np.array(img)

# Create a new figure to append the colorbar beneath the image
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={'height_ratios': [20, 0.5]})

# Display the image in the first subplot
ax1.imshow(img_array)
ax1.axis('off')  # Turn off axis

# Create the colorbar in the second subplot
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=.2, vmax=.45)
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, orientation='horizontal')
cb.set_label('Volumetric Water Content')

# Adjust layout
plt.tight_layout()
plt.savefig("./Output_Data/Heatmap_Overlay_Colorbar_Visualization.tiff", format="tiff")