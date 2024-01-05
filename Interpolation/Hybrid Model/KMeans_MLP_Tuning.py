import numpy as np
from sklearn.cluster import KMeans
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPRegressor

df = pandas.read_csv("Final.csv")
columns = list(df.columns.values)
scaler = StandardScaler()
npdf = scaler.fit(df.to_numpy()).transform(df.to_numpy())
df = pandas.DataFrame(npdf, columns=columns)

lf = df.drop(columns=["X", "Y", "VWC"])
values = list(lf.columns.values)

cols = ["DEM_QGIS", "PlnCurv", "TWI"]

network_shapes = []

for i in range(5, 100, 5):
    network_shapes.append((i,))

for i in range(5, 105, 5):
    for e in range(5, 105, 5):
        network_shapes.append((i,e))

r2s = []

trail = 1
trial_total = len(network_shapes)

for s in network_shapes:
    np.random.seed(22)
    np.random.set_state(np.random.get_state())

    # Sample data (replace with your data)
    response_variable = np.squeeze(df[["VWC"]].to_numpy())
    external_variable = df[cols].to_numpy()
    coordinates = df[["X", "Y"]].to_numpy()

    # Perform KMeans clustering on the external variable
    np.random.seed(23)
    np.random.set_state(np.random.get_state())
    kmeans = KMeans(n_clusters=7)
    labels = kmeans.fit_predict(np.hstack((external_variable, coordinates)))
    num_classes = len(np.unique(labels))

    np.random.seed(22)
    np.random.set_state(np.random.get_state())
    response_variable_train, response_variable_test, external_variable_train, external_variable_test, coordinates_train, coordinates_test, labels_train, labels_test = train_test_split(response_variable, external_variable, coordinates, labels, test_size=.3)

    NN_models = []
    NN_models_label = []


    for class_label in np.unique(labels_train):
        class_idx = np.where(labels_train == class_label)[0]
        class_response = response_variable_train[class_idx]
        class_external = external_variable_train[class_idx]
        class_corrdinates = coordinates_train[class_idx]

        np.random.seed(22)
        np.random.set_state(np.random.get_state())
        NN = MLPRegressor(hidden_layer_sizes=s)
        NN.fit(np.hstack((class_external, class_corrdinates)), class_response)

        NN_models.append(NN)
        NN_models_label.append(class_label)

    predictions = []

    for i in range(0,len(labels_test)):
        catagory = labels_test[i]
        NN = NN_models[catagory]

        inp = external_variable_test[i,0:len(cols)]

        try:
            prediction = NN.predict(np.hstack((inp, np.array([coordinates_test[i,:]]))))
        except ValueError:
            inp = np.array([inp])
            prediction = NN.predict(np.hstack((inp, np.array([coordinates_test[i,:]]))))

        predictions.append(prediction[0])

    print(str(trail) + "/" + str(trial_total))
    trail += 1
    r2s.append(r2_score(response_variable_test, predictions))

data = {"Shape":network_shapes, "R2":r2s}
out_df = pandas.DataFrame(data)

csv = "shape_comparision_results.csv"

out_df.to_csv(csv, index=False)