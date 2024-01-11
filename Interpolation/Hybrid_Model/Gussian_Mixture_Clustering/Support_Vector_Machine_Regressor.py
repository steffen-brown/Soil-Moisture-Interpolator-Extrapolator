import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.mixture import GaussianMixture

df = pandas.read_csv("../Input_Data/in-situ_Data.csv")
columns = list(df.columns.values)
scaler = StandardScaler()
npdf = scaler.fit(df.to_numpy()).transform(df.to_numpy())
df = pandas.DataFrame(npdf, columns=columns)

lf = df.drop(columns=["X", "Y", "VWC", "Slope"])
values = list(lf.columns.values)

r2s = []
RMSE = []
MEA = []
parameters = []
nums_classes = []
actual = []
predicted = []

print("Testing Models", end=" ")
for r in range(1, 10):
    combinations = itertools.combinations(values, r)
    for combination in combinations:
        cols = list(combination)
        for e in range(1,10):
            # Sample data (replace with your data)
            response_variable = np.squeeze(df[["VWC"]].to_numpy())
            external_variable = df[cols].to_numpy()
            coordinates = df[["X", "Y"]].to_numpy()

            # Perform KMeans clustering on the external variable
            num_classes = e
            kmeans = GaussianMixture(n_components=num_classes, random_state=23)
            labels = kmeans.fit_predict(np.hstack((external_variable, coordinates)))

            np.random.seed(22)
            np.random.set_state(np.random.get_state())
            response_variable_train, response_variable_test, external_variable_train, external_variable_test, coordinates_train, coordinates_test, labels_train, labels_test = train_test_split(response_variable, external_variable, coordinates, labels, test_size=.3)

            KN_models = []

            try:
                for class_label in np.unique(labels_train):
                    class_idx = np.where(labels_train == class_label)[0]
                    class_response = response_variable_train[class_idx]
                    class_external = external_variable_train[class_idx]
                    class_corrdinates = coordinates_train[class_idx]

                    np.random.seed(22)
                    np.random.set_state(np.random.get_state())
                    KN = SVR()
                    KN.fit(np.hstack((class_external, class_corrdinates)), class_response)

                    KN_models.append(KN)

                predictions = []

                for i in range(0,len(labels_test)):
                    catagory = labels_test[i]
                    KN = KN_models[catagory]

                    inp = external_variable_test[i,0:len(cols)]

                    try:
                        prediction = KN.predict(np.hstack((inp, np.array([coordinates_test[i,:]]))))
                    except ValueError:
                        inp = np.array([inp])
                        prediction = KN.predict(np.hstack((inp, np.array([coordinates_test[i,:]]))))

                    predictions.append(prediction[0])

                r2s.append(r2_score(response_variable_test, predictions))
                MEA.append(mean_absolute_error(np.array(response_variable_test) * scaler.scale_[2] + scaler.mean_[2], np.array(predictions) * scaler.scale_[2] + scaler.mean_[2]))
                RMSE.append(np.sqrt(mean_squared_error(np.array(response_variable_test) * scaler.scale_[2] + scaler.mean_[2], np.array(predictions) * scaler.scale_[2] + scaler.mean_[2])))
                parameters.append(cols)
                nums_classes.append(e)
                actual.append(response_variable_test)
                predicted.append(predictions)
                print(".", end=" ", flush=True)
            except:
                print("x", end=" ", flush=True)

idx = r2s.index(max(r2s))
mean = scaler.mean_[2]
std = scaler.scale_[2]

scaled_actual = np.array(actual[idx]) * std + mean
scaled_predicted = np.array(predicted[idx]) * std + mean

rawDF = pandas.DataFrame({
    "Predicted":scaled_predicted,
    "Actual":scaled_actual
})

rawDF.to_csv("./Output_Data/Predicted_Actual/Support_Vector_Machine_Results.csv")

sorted_items = sorted(enumerate(r2s), key=lambda x: x[1], reverse=True)

top_r2s = [item[1] for item in sorted_items]
top_indexes = [item[0] for item in sorted_items]
top_parameters = [parameters[index] for index in top_indexes]
top_classes = [nums_classes[index] for index in top_indexes]
top_RMSE = [RMSE[index] for index in top_indexes]
top_MEA = [MEA[index] for index in top_indexes]

top5 = pandas.DataFrame({
    "R2 (Validation)":top_r2s,
    "RMSE":top_RMSE,
    "MAE":top_MEA,
    "Input Parameters":top_parameters,
    "Class Quantity":top_classes
})

top5.to_csv("./Output_Data/Performance/Support_Vector_Machine_Results.csv")