import numpy as np
from sklearn.mixture import GaussianMixture
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

import warnings
warnings.filterwarnings("ignore")

df = pandas.read_csv("../Input_Data/in-situ_Data.csv")
columns = list(df.columns.values)
scaler = StandardScaler()
npdf = scaler.fit(df.to_numpy()).transform(df.to_numpy())
df = pandas.DataFrame(npdf, columns=columns)

lf = df.drop(columns=["X", "Y", "VWC", "Slope"])
values = list(lf.columns.values)

cols = [['DEM_QGIS', 'PlnCurv', 'TWI'], ['ProfCurv', 'PlnCurv', 'TWI'], ['DEM_QGIS', 'PlnCurv', 'TWI'], ['PlnCurv', 'TWI'], ['PlnCurv', 'TWI', 'NDVI'], ['PlnCurv', 'TWI'], ['PlnCurv', 'TPI', 'TWI']]
clas = [8, 4, 9, 9, 9, 4, 7]

network_shapes = []

for i in range(5, 100, 5):
    network_shapes.append((i,))

for i in range(5, 105, 5):
    for e in range(5, 105, 5):
        network_shapes.append((i,e))

r2s = []
RMSE = []
MAE = []
parameters = []
classes = []
shape = []

for co, cl in zip(cols, clas):
    print("Testing Shapes", end=" ")
    for s in network_shapes:
        np.random.seed(22)
        np.random.set_state(np.random.get_state())

        # Sample data (replace with your data)
        response_variable = np.squeeze(df[["VWC"]].to_numpy())
        external_variable = df[co].to_numpy()
        coordinates = df[["X", "Y"]].to_numpy()

        # Perform KMeans clustering on the external variable
        np.random.seed(23)
        np.random.set_state(np.random.get_state())
        kmeans = GaussianMixture(n_components=cl, random_state=23)
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

        print(".", end=" ", flush=True)
        MAE.append(mean_absolute_error(response_variable_test * scaler.scale_[2] + scaler.mean_[2], np.array(predictions) * scaler.scale_[2] + scaler.mean_[2]))
        RMSE.append(np.sqrt(mean_squared_error(np.array(response_variable_test) * scaler.scale_[2] + scaler.mean_[2], np.array(predictions) * scaler.scale_[2] + scaler.mean_[2])))
        r2s.append(r2_score(response_variable_test, predictions))
        classes.append(cl)
        parameters.append(co)
        shape.append(s)

data = {"Shape":shape, "Input Parameters": parameters, "Class Quantity": classes, "R2 Score":r2s, "RMSE": RMSE, "MAE": MAE}
out_df = pandas.DataFrame(data)
out_df.sort_values(by='R2 Score', ascending=False, inplace=True)

csv = "./Output_Data/Gaussian_Mixture_Shape_Results.csv"

out_df.to_csv(csv)