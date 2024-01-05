import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pandas
import matplotlib.pyplot as plt

np.random.seed(42)

dataframe = pandas.read_csv("Data1.csv")
dataArray = dataframe.to_numpy()

scaler = StandardScaler().fit(dataArray)
scaledData = scaler.transform(dataArray)

features = np.delete(scaledData, [2,10], axis=1)

moisture = scaledData[:,2]

featuresTrain, featuresTest, moistureTrain, moistureTest = train_test_split(features, moisture, test_size=0.3)

model = LinearRegression()
sfs = SequentialFeatureSelector(model, scoring="r2", k_features=9, forward=True, verbose=2)
sfs.fit(featuresTrain, moistureTrain)

model.fit(featuresTrain[:, sfs.k_feature_idx_], moistureTrain)
predictions = model.predict(featuresTest[:, sfs.k_feature_idx_])

mae = mean_absolute_error(moistureTest, predictions)
print("\nMAE:", mae)
mse = mean_squared_error(moistureTest, predictions)
print("MSE:", mse)
evs = explained_variance_score(moistureTest, predictions)
print("EVS:", evs)
r2 = r2_score(moistureTest, predictions)
print("R2:", evs)

