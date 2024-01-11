import numpy as np
from sklearn.cluster import KMeans
import pandas
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

    scaled_device_coordinates = device_coordinates.copy()
    scaled_device_coordinates[:,0] = (device_coordinates[:,0] - scaler.mean_[0]) / scaler.scale_[0]
    scaled_device_coordinates[:,1] = (device_coordinates[:,1] - scaler.mean_[1]) / scaler.scale_[1]

    np_sensor_data = sensor_data[["Device ID", "Volumetric Water Content"]].to_numpy()

    for sensor in device_id:
        filtered_rows = np_sensor_data[np_sensor_data[:,0] == sensor]
        sensor_vwc = filtered_rows[:,1].tolist()
        device_vwc.append(sensor_vwc)

    device_avg_vwc = []
    for vwcs in device_vwc:
        avg = np.mean(np.array(vwcs))
        device_avg_vwc.append(avg)

    map_data = pandas.read_csv("../Input_Data/Map_Data.csv").to_numpy()
    int_device_coordinates_x = np.ceil((device_coordinates[:,0] - min(map_data[:,0]))/9.4)
    int_device_coordinates_y = np.ceil((device_coordinates[:,1] - min(map_data[:,1]))/9.4)
    map_data[:,0] = np.ceil((map_data[:,0] - min(map_data[:,0]))/9.4)
    map_data[:,1] = np.ceil((map_data[:,1] - min(map_data[:,1]))/9.4)

    sensor_external_variables = []
    for i in range(0, len(int_device_coordinates_x)):
        mask = (map_data[:, 0] == int_device_coordinates_x[i]) & (map_data[:, 1] == int_device_coordinates_y[i])
        sensor_external_variables.append(map_data[mask][:, [2,3,7]][0])

    sensor_external_variables = np.array(sensor_external_variables)
    sensor_external_variables[:,0] = (sensor_external_variables[:,0] - scaler.mean_[3]) / scaler.scale_[3]
    sensor_external_variables[:,1] = (sensor_external_variables[:,1] - scaler.mean_[6]) / scaler.scale_[6]
    sensor_external_variables[:,2] = (sensor_external_variables[:,2] - scaler.mean_[8]) / scaler.scale_[8]

    device_avg_vwc = (device_avg_vwc - scaler.mean_[2]) / scaler.scale_[2]

    labels = kmeans.predict(np.hstack((sensor_external_variables, scaled_device_coordinates)))

    np.random.seed(22)
    np.random.set_state(np.random.get_state())

    sdc_train = scaled_device_coordinates
    sev_train = sensor_external_variables
    labels_train = labels
    davwc_train = device_avg_vwc

    global first, second
    secondary_NN = MLPRegressor(hidden_layer_sizes=(10,5), max_iter=500)
    train_dataset = []
    day_avg_vwc = 0
    for i in range(0, len(labels_train)):
        day_avg_vwc = np.mean(davwc_train)
        geo_spat_vwc = NN_models[labels_train[i]].predict([np.hstack((sev_train[i], sdc_train[i]))])[0]
        train_dataset.append(np.hstack((day_avg_vwc, geo_spat_vwc)))

    np.random.seed(22)
    np.random.set_state(np.random.get_state())
    secondary_NN.fit(train_dataset, davwc_train)

    return secondary_NN, day_avg_vwc, device_id

sensor_data = pandas.read_csv("../Input_Data/MultiDay_WUSN_Data.csv")
sensor_data['Datetime Slot'] = pandas.to_datetime(sensor_data['Datetime Slot'])

days = ['2019-12-25', '2019-12-26', '2019-12-27', '2019-12-28', '2019-12-29', '2019-12-30', '2019-12-31','2020-01-01','2020-01-02','2020-01-03','2020-01-04','2020-01-05','2020-01-06','2020-01-07', '2020-01-08','2020-01-09', '2020-01-10',]

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



for d in days:
    print("-- New Day --")
    predictions = []

    df_day = sensor_data[sensor_data['Datetime Slot'].dt.date == pandas.to_datetime(d).date()]
    SNN, day_avg_vwc, active_sensors = KMeansClassifiedNN_tranfer_train(df_day, scaler, kmeans)

    
    print("Generating Heatmap")
    heatmap = np.full((int(max(scaledY) + 1), int(max(scaledX) + 1)), np.NaN)
    for i in range(0, len(scaledX)):
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

    output_dataframe.to_csv("./Output_Data/Multi_Day/Maps/Heatmap_Results_" + d + ".csv")

    print("Generating Heatmap Visual")
    plt.imshow(np.flipud(heatmap), interpolation='none', vmin=.30, vmax=.45)
    plt.colorbar()
    plt.title(d)
    plt.savefig("./Output_Data/Multi_Day/Maps/Heatmap_Visual_" + d + ".tiff", format="tiff")
    plt.close()

    print("Generating Heatmap Overlay Visual")
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
        folium.Rectangle([(cord_left[i][0], cord_left[i][1]), (cord_right[i][0], cord_right[i][1])], weight=0, fill_color=colors[i], fill_opacity=.75).add_to(map)

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
    img.save("./Output_Data/Multi_Day/Maps/Heatmap_Overlay_Visual_" + d + ".tiff", format="tiff")

    img = Image.open("./Output_Data/Multi_Day/Maps/Heatmap_Overlay_Visual_" + d + ".tiff")

    img_array = np.array(img)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={'height_ratios': [20, 0.5]})

    ax1.imshow(img_array)
    ax1.axis('off')

    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=.2, vmax=.45)
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_label('Volumetric Water Content')

    plt.tight_layout()
    plt.savefig("./Output_Data/Multi_Day/Maps/Heatmap_Overlay_Colorbar_Visual_" + d + ".tiff")
    plt.close()
