import numpy as np
from sklearn.cluster import KMeans
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.cluster import DBSCAN

df = pandas.read_csv("Final.csv")
columns = list(df.columns.values)
scaler = StandardScaler()
npdf = scaler.fit(df.to_numpy()).transform(df.to_numpy())
df = pandas.DataFrame(npdf, columns=columns)

lf = df.drop(columns=["X", "Y", "VWC"])
values = list(lf.columns.values)

r2s = []
parameters = []
nums_classes = []
actual = []
predicted = []

for r in range(1, 10):
    combinations = itertools.combinations(values, r)
    for combination in combinations:
        cols = list(combination)
        for e in range(1,10):
            for n in range(1,10):
                # Sample data (replace with your data)
                response_variable = np.squeeze(df[["VWC"]].to_numpy())
                external_variable = df[cols].to_numpy()
                coordinates = df[["X", "Y"]].to_numpy()

                # Perform KMeans clustering on the external variable
                kmeans = DBSCAN(eps=e, min_samples=n)
                labels = kmeans.fit_predict(np.hstack((external_variable, coordinates)))
                num_classes = len(np.unique(labels))

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
                            print(inp.shape)
                            print(np.array([coordinates_test[i,:]]).shape)
                            prediction = KN.predict(np.hstack((inp, np.array([coordinates_test[i,:]]))))

                        predictions.append(prediction[0])

                    r2s.append(r2_score(response_variable_test, predictions))
                    parameters.append(cols)
                    nums_classes.append(num_classes)
                    actual.append(response_variable_test)
                    predicted.append(predictions)
                    print("Add")
                except:
                    print("fail")

print(max(r2s))
idx = r2s.index(max(r2s))
mean = scaler.mean_[2]
std = scaler.scale_[2]

scaled_actual = np.array(actual[idx]) * std + mean
scaled_predicted = np.array(predicted[idx]) * std + mean

print(scaled_actual)
print(scaled_predicted)

rawDF = pandas.DataFrame({
    "Predicted":scaled_predicted,
    "Actual":scaled_actual
})

rawDF.to_csv("RawData/D_SVM.csv", index=False)
# print(parameters[idx])
# print(nums_classes[idx])

# average_r2_by_class = []
# for i in range(min(nums_classes), max(nums_classes)):
#     indices = [index for index, value in enumerate(nums_classes) if value == i]

#     mx = r2s[indices[0]]
#     for id in indices:
#         if r2s[id] > mx:
#             mx = r2s[id] 

#     average_r2_by_class.append(mx)

# top_n = 5
# sorted_items = sorted(enumerate(r2s), key=lambda x: x[1], reverse=True)[:top_n]

# top_r2s = [item[1] for item in sorted_items]
# top_indexes = [item[0] for item in sorted_items]
# top_parameters = [parameters[index] for index in top_indexes]
# top_classes = [nums_classes[index] for index in top_indexes]

# top5 = pandas.DataFrame({
#     "R2 (Validation)":top_r2s,
#     "Input Parameters":top_parameters,
#     "Class Quantity":top_classes
# })

# print(top5.head(5))

# # Create the line graph
# plt.plot(range(min(nums_classes), max(nums_classes)), average_r2_by_class, marker='o', linestyle='-', color='b')

# # Add a title
# plt.title('Number of Classificaitions vs Max R2')

# # Add axis labels
# plt.xlabel('Number of Classifications')
# plt.ylabel('Maximum R2')

# # Show the plot
# plt.show()