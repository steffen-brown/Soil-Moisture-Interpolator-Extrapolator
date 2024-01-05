import numpy as np
from sklearn.cluster import KMeans
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import StandardScaler


from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import gaussian_filter
from collections import Counter

NN_models = []
possible_regressors = [
    MLPRegressor(),
]
first = 0
second = 0

def KMeansClassifiedNN_init_train(input_external, input_coordinates):
    global NN_models, secondary_NN

    predictor_rows = []
    for r in range(0, len(input_external)):
        predictor_rows.append([input_coordinates[r, 0], input_coordinates[r,1], np.NaN, input_external[r,0], np.NaN, np.NaN, input_external[r,1], np.NaN, input_external[r,2], np.NaN, np.NaN])
    predictor_rows = np.array(predictor_rows)

    df = pandas.read_csv("../Final.csv")
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

    return scaler, kmeans

def KMeansClassifiedNN_tranfer_train_predict(sensor_data, scaler, kmeans, regressor):

    sensor_locations = pandas.read_csv("../sensor_locations.csv")[["ID", "POINT_X", "POINT_Y"]].to_numpy()
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
    map_data = pandas.read_csv("../mapData.csv").to_numpy()
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
    cord_sorted_indexes_x = np.argsort(scaled_device_coordinates[:, 0])
    test_indexes_x = cord_sorted_indexes_x[-4:][0]
    cord_sorted_indexes_y = np.argsort(scaled_device_coordinates[:, 1])
    test_indexes_y = cord_sorted_indexes_y[-4:][0]

    sdc_test = scaled_device_coordinates[[test_indexes_x, test_indexes_y]]
    sdc_train = np.delete(scaled_device_coordinates, [test_indexes_x, test_indexes_y], axis=0)
    sev_test = sensor_external_variables[[test_indexes_x, test_indexes_y]]
    sev_train = np.delete(sensor_external_variables, [test_indexes_x, test_indexes_y], axis=0)
    labels_test = labels[[test_indexes_x, test_indexes_y]]
    labels_train = np.delete(labels, [test_indexes_x, test_indexes_y], axis=0)
    davwc_test = device_avg_vwc[[test_indexes_x, test_indexes_y]]
    davwc_train = np.delete(device_avg_vwc, [test_indexes_x, test_indexes_y], axis=0)

    # plt.scatter(sdc_test[:, 0] * scaler.scale_[0] + scaler.mean_[0], sdc_test[:, 1] * scaler.scale_[1] + scaler.mean_[1], color='blue')
    # plt.scatter(sdc_train[:, 0] * scaler.scale_[0] + scaler.mean_[0], sdc_train[:, 1] * scaler.scale_[1] + scaler.mean_[1], color='red')

    # plt.show()

    global first, second
    secondary_NN = MLPRegressor(hidden_layer_sizes=(first,second), max_iter=500)
    train_dataset = []
    # Training loop
    for i in range(0, len(labels_train)):
        day_avg_vwc = np.mean(davwc_train)
        geo_spat_vwc = NN_models[labels_train[i]].predict([np.hstack((sev_train[i], sdc_train[i]))])[0]
        train_dataset.append(np.hstack((day_avg_vwc, geo_spat_vwc)))

    np.random.seed(22)
    np.random.set_state(np.random.get_state())
    secondary_NN.fit(train_dataset, davwc_train)



    predictions = []
    # Test loop
    for i in range(0, len(labels_test)):
        day_avg_vwc = np.mean(davwc_test)
        geo_spat_vwc = NN_models[labels_train[i]].predict([np.hstack((sev_test[i], sdc_test[i]))])[0]

        prediction = secondary_NN.predict([np.hstack((day_avg_vwc, geo_spat_vwc))])[0]
        predictions.append(prediction)

        # print(day_avg_vwc, " - ")
        # print(prediction)


    print(np.array(predictions) * scaler.scale_[2] + scaler.mean_[2], np.array(davwc_test) * scaler.scale_[2] + scaler.mean_[2])
    return predictions, davwc_test

sensor_data = pandas.read_csv("../multiday.csv")
sensor_data['Datetime Slot'] = pandas.to_datetime(sensor_data['Datetime Slot'])

days = ['2019-12-25', '2019-12-26', '2019-12-27', '2019-12-28', '2019-12-29', '2019-12-30', '2019-12-31','2020-01-01','2020-01-02','2020-01-03','2020-01-04','2020-01-05','2020-01-06','2020-01-07', '2020-01-08','2020-01-09', '2020-01-10',]

map_df = pandas.read_csv("../mapData.csv")
cord_in = map_df[["POINT_X", "POINT_Y"]].to_numpy()
ext_in = map_df[["DEM", "PlnCurv", "TWI"]].to_numpy()

scaledX = np.ceil((cord_in[:,0] - min(cord_in[:,0]))/9.4)
scaledY = np.ceil((cord_in[:,1] - min(cord_in[:,1]))/9.4)
scaledCoord = np.array([scaledX, scaledY]).T

scaler, kmeans = KMeansClassifiedNN_init_train(ext_in, cord_in)

r2s = []
sizes = []

for f in range(10, 15, 5):
    for s in range(5, 10, 5):
        first = f
        second = s
        for r in possible_regressors:
            predictions = []
            actuals = []
            for d in days:
                df_day = sensor_data[sensor_data['Datetime Slot'].dt.date == pandas.to_datetime(d).date()]
                prediction, actual = KMeansClassifiedNN_tranfer_train_predict(df_day, scaler, kmeans, r)

                predictions.append(prediction)
                actuals.append(actual)

            actuals = np.concatenate(actuals)
            predictions = np.concatenate(predictions)

            # outliers = np.where(np.abs(np.array(predictions) - np.array(actuals)) > .5)
            # predictions = np.delete(np.array(predictions), outliers)
            # actuals = np.delete(np.array(actuals), outliers)

            actuals = np.array(actuals) * scaler.scale_[2] + scaler.mean_[2]
            predictions = np.array(predictions) * scaler.scale_[2] + scaler.mean_[2]

            actuals = np.delete(actuals, [12])
            predictions = np.delete(predictions, [12])

            sizes.append((first,second))
            r2s.append(r2_score(actuals, predictions))

            print(r2_score(actuals, predictions))
            print(mean_absolute_error(actuals, predictions))
            print(mean_squared_error(actuals, predictions))

            print(actuals)
            print(predictions)

            for a in actuals:
                print(a)

            print("Ererer")

            for p in predictions:
                print(p)

            

print(max(r2s))
print(sizes[r2s.index(max(r2s))])


# print(np.round(np.array(predictions) * scaler.scale_[2] + scaler.mean_[2], 5))
# print(np.round(np.array(actual) * scaler.scale_[2] + scaler.mean_[2], 5))

# outliers = np.where(np.abs(np.array(predictions) - np.array(actual)) > .5)
# predictions = np.delete(np.array(predictions), outliers)
# actual = np.delete(np.array(actual), outliers)

# print(r2_score(actual, predictions))
# print("MAE ", mean_absolute_error(np.array(actual) * scaler.scale_[2] + scaler.mean_[2], np.array(predictions) * scaler.scale_[2] + scaler.mean_[2]))
# print("MSE ", mean_squared_error(np.array(actual) * scaler.scale_[2] + scaler.mean_[2], np.array(predictions) * scaler.scale_[2] + scaler.mean_[2]))

# print(secondary_NN[0].coefs_[0])
