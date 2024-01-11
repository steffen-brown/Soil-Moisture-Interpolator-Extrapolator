import numpy as np
from sklearn.cluster import KMeans
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import utm
import folium
import io
from PIL import Image
import matplotlib as mpl

import warnings
warnings.filterwarnings("ignore")

NN_models = []

def KMeansClassifiedNN_init_train(input_external, input_coordinates):
    global NN_models, secondary_NN

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

    response_variable_train = response_variable[:63]
    external_variable_train = external_variable[:63]
    coordinates_train = coordinates[:63]
    labels_train = labels[:63]
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

    secondary_NN = np.full(len(NN_models), np.NaN).tolist()

    return scaler, kmeans, labels[64:]

def KMeansClassifiedNN_tranfer_train(sensor_data, scaler, kmeans):

    sensor_locations = pandas.read_csv("../Input_Data/WUSN_Locations.csv")[["ID", "POINT_X", "POINT_Y"]].to_numpy()
    device_id = list(np.unique(sensor_data[["Device ID"]].to_numpy().flatten()))

    device_coordinates = []
    for dev in device_id.copy():
        idx = np.where(sensor_locations[:, 0] == dev)[0]
        if(len(idx) == 0):
            device_id.remove(dev)
        else:
            device_coordinates.append([sensor_locations[idx, 1][0], sensor_locations[idx, 2][0]])

    device_coordinates = np.array(device_coordinates)

    device_vwc = []

    # Coordinate scaling based on trained scaler
    scaled_device_coordinates = device_coordinates.copy()
    scaled_device_coordinates[:,0] = (device_coordinates[:,0] - scaler.mean_[0]) / scaler.scale_[0]
    scaled_device_coordinates[:,1] = (device_coordinates[:,1] - scaler.mean_[1]) / scaler.scale_[1]

    # Loading additional sensor data for model refinement
    np_sensor_data = sensor_data[["Device ID", "Volumetric Water Content"]].to_numpy()

    # Data extraction and processing for each sensor
    for sensor in device_id:
        filtered_rows = np_sensor_data[np_sensor_data[:,0] == sensor]
        sensor_vwc = filtered_rows[:,1].tolist()
        device_vwc.append(sensor_vwc)

    # Computing average volumetric water content
    device_avg_vwc = []
    for vwcs in device_vwc:
        avg = np.mean(np.array(vwcs))
        device_avg_vwc.append(avg)

    # Processing map data for model input
    map_data = pandas.read_csv("../Input_Data/Map_Data.csv").to_numpy()
    int_device_coordinates_x = np.ceil((device_coordinates[:,0] - min(map_data[:,0]))/9.4)
    int_device_coordinates_y = np.ceil((device_coordinates[:,1] - min(map_data[:,1]))/9.4)
    map_data[:,0] = np.ceil((map_data[:,0] - min(map_data[:,0]))/9.4)
    map_data[:,1] = np.ceil((map_data[:,1] - min(map_data[:,1]))/9.4)

    # Extraction and scaling of external variables from sensor data
    sensor_external_variables = []
    for i in range(0, len(int_device_coordinates_x)):
        mask = (map_data[:, 0] == int_device_coordinates_x[i]) & (map_data[:, 1] == int_device_coordinates_y[i])
        sensor_external_variables.append(map_data[mask][:, [2,3,7]][0])

    sensor_external_variables = np.array(sensor_external_variables)
    sensor_external_variables[:,0] = (sensor_external_variables[:,0] - scaler.mean_[3]) / scaler.scale_[3]
    sensor_external_variables[:,1] = (sensor_external_variables[:,1] - scaler.mean_[6]) / scaler.scale_[6]
    sensor_external_variables[:,2] = (sensor_external_variables[:,2] - scaler.mean_[8]) / scaler.scale_[8]

    # Scaling average VWC data for sensor devices
    device_avg_vwc = (device_avg_vwc - scaler.mean_[2]) / scaler.scale_[2]

    # Clustering of sensor data for model refinement
    labels = kmeans.predict(np.hstack((sensor_external_variables, scaled_device_coordinates)))

    np.random.seed(22)
    np.random.set_state(np.random.get_state())
    # sev_train, sev_test, sdc_train, sdc_test, labels_train, labels_test, davwc_train, davwc_test = train_test_split(sensor_external_variables, scaled_device_coordinates, labels, device_avg_vwc, train_size=.9)

    idxs = [0,1,2,3,6,7,8,10,11,13]
    sdc_train = scaled_device_coordinates[idxs]
    sev_train = sensor_external_variables[idxs]
    labels_train = labels[idxs]
    davwc_train = device_avg_vwc[idxs]

    # plt.scatter(sdc_test[:, 0] * scaler.scale_[0] + scaler.mean_[0], sdc_test[:, 1] * scaler.scale_[1] + scaler.mean_[1], color='blue')
    # plt.scatter(sdc_train[:, 0] * scaler.scale_[0] + scaler.mean_[0], sdc_train[:, 1] * scaler.scale_[1] + scaler.mean_[1], color='red')

    # plt.show()

    global first, second
    secondary_NN = MLPRegressor(hidden_layer_sizes=(10,5), max_iter=500)
    train_dataset = []
    day_avg_vwc = 0
    # Training loop
    for i in range(0, len(labels_train)):
        day_avg_vwc = np.mean(davwc_train)
        geo_spat_vwc = NN_models[labels_train[i]].predict([np.hstack((sev_train[i], sdc_train[i]))])[0]
        train_dataset.append(np.hstack((day_avg_vwc, geo_spat_vwc)))

    np.random.seed(22)
    np.random.set_state(np.random.get_state())
    secondary_NN.fit(train_dataset, davwc_train)

    return secondary_NN, day_avg_vwc, device_id

sensor_data = pandas.read_csv("../Input_Data/SingleDay_WUSN_Data.csv")
sensor_data['Datetime Slot'] = pandas.to_datetime(sensor_data['Datetime Slot'])

map_df = pandas.read_csv("../Input_Data/Map_Data.csv")
cord_in = map_df[["POINT_X", "POINT_Y"]].to_numpy()
ext_in = map_df[["DEM", "PlnCurv", "TWI"]].to_numpy()

scaledX = np.ceil((cord_in[:,0] - min(cord_in[:,0]))/9.4)
scaledY = np.ceil((cord_in[:,1] - min(cord_in[:,1]))/9.4)
scaledCoord = np.array([scaledX, scaledY]).T

scaler, kmeans, labels = KMeansClassifiedNN_init_train(ext_in, cord_in)

cord_in[:,0] = (cord_in[:,0] - scaler.mean_[0]) / scaler.scale_[0]
cord_in[:,1] = (cord_in[:,1] - scaler.mean_[1]) / scaler.scale_[1]

ext_in[:,0] = (ext_in[:,0] - scaler.mean_[3]) / scaler.scale_[3]
ext_in[:,1] = (ext_in[:,1] - scaler.mean_[6]) / scaler.scale_[6]
ext_in[:,2] = (ext_in[:,2] - scaler.mean_[8]) / scaler.scale_[8]

predictions = []

df_day = sensor_data
SNN, day_avg_vwc, active_sensors = KMeansClassifiedNN_tranfer_train(df_day, scaler, kmeans)

print("Generating Heatmap", end=" ", flush=True)
heatmap = np.full((int(max(scaledY) + 1), int(max(scaledX) + 1)), np.NaN)
for i in range(0, len(scaledX)):
    print(".", end=" ")
    reg = labels[i]
    NN = NN_models[reg]
    geo_spat_pred = NN.predict([np.hstack((ext_in[i], cord_in[i]))])

    prediction = SNN.predict([np.hstack((day_avg_vwc, geo_spat_pred))])

    prediction = prediction * scaler.scale_[2] + scaler.mean_[2]
    predictions.append(prediction[0])
    heatmap[int(scaledY[i]), int(scaledX[i])] = prediction[0]

cords = map_df[["POINT_X", "POINT_Y"]].to_numpy()
output_dataframe = pandas.DataFrame({
    "Latitude": cords[:,0],
    "Logitude": cords[:,1],
    "VWC": predictions
})

output_dataframe.to_csv("./Output_Data/Single_Day/Heatmap_Results.csv")

print("\nGenerating Heatmap Visualization")
plt.imshow(np.flipud(heatmap), interpolation='none', vmax=.45, vmin=.2)
plt.colorbar()
plt.title("2021-11-18")
plt.savefig("./Output_Data/Single_Day/Heatmap_Visualization.tiff", format="tiff")
plt.close()

# Mapover lay

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
colors = get_hex_colors(predictions)

for i in range(0, len(cord_left)):
    folium.Rectangle([(cord_left[i][0], cord_left[i][1]), (cord_right[i][0], cord_right[i][1])], weight=0, fill_color=colors[i], fill_opacity=.5).add_to(map)

sensor_locations = pandas.read_csv("../Input_Data/WUSN_Locations.csv")[["ID", "POINT_X", "POINT_Y"]].to_numpy()
mask = np.isin(sensor_locations[:, 0], active_sensors)
sensor_locations = sensor_locations[mask]
for i in range(0, len(sensor_locations)):
    converted_cord = utm.to_latlon(sensor_locations[i][1], sensor_locations[i][2], 15, northern=True)
    folium.CircleMarker(location=np.array([converted_cord[0], converted_cord[1]]), radius=2, color='red', fill=True).add_to(map)

tile = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
       ).add_to(map)

img_data = map._to_png(5)
img = Image.open(io.BytesIO(img_data))
img.save('./Output_Data/Single_Day/Heatmap_Overlay_Visualization.tiff', format="tiff")

image_path = './Output_Data/Single_Day/Heatmap_Overlay_Visualization.tiff'  # Replace with actual image path
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
plt.savefig("./Output_Data/Single_Day/Heatmap_Overlay_Colorbar_Visualization.tiff", format="tiff")

