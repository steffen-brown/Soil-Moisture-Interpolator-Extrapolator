import numpy as np
import itertools
import pandas

from pykrige.uk import UniversalKriging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

np.random.seed(23)

dataframe = pandas.read_csv("./input_data/in-situ_Data.csv")
dataArray = dataframe.to_numpy()
scaler = StandardScaler().fit(dataArray)
scaledData = scaler.transform(dataArray)

x = scaledData[:,0]
y = scaledData[:,1]
dem = scaledData[:,3]
profCurve = scaledData[:,4]
pinCurv = scaledData[:,6]
tpi = scaledData[:,7]
twi = scaledData[:,8]
texture = scaledData[:,9]
ndvi = scaledData[:,10]
moisture = scaledData[:,2]

xTrain, xTest, yTrain, yTest, moistureTrain, moistureTest, demTrain, demTest, profCurvTrain, profCurveTest, pinCurvTrain, pinCurvTest, tpiTrain, tpiTest, twiTrain, twiTest, textureTrain, textureTest, ndviTrain, ndviTest = train_test_split(x, y, moisture, dem, profCurve, pinCurv, tpi, twi, texture, ndvi, test_size=.3)

extVariableTrain = [demTrain, profCurvTrain, pinCurvTrain, tpiTrain, twiTrain, textureTrain, ndviTrain]
extVariableTest = [demTest, profCurveTest, pinCurvTest, tpiTest, twiTest, textureTest, ndviTest]
extVariableLabel = ["DEM_QGIS", "ProfCurv", "PlnCurv", "TPI", "TWI", "Texture", "NDVI"]

R2Params = []
R2Models = []
R2test = []
RSME = []
MAE = []

print("Testing model .", end=" ")

UK = UniversalKriging(
    xTrain,
    yTrain,
    moistureTrain,
)

predictionsWOD, sigmasq = UK.execute('points', xTest, yTest)

R2Params.append("N/A")
R2Models.append("Universal Krigging without External Drift")
R2test.append(r2_score(moistureTest, predictionsWOD))
RSME.append(mean_squared_error(moistureTest * scaler.scale_[2] + scaler.mean_[2], predictionsWOD * scaler.scale_[2] + scaler.mean_[2]))
MAE.append(r2_score(moistureTest * scaler.scale_[2] + scaler.mean_[2], predictionsWOD * scaler.scale_[2] + scaler.mean_[2]))

print(".", end="\n")

for r in range(1, 10):
    train_comb = itertools.combinations(extVariableTrain, r)
    test_comb = itertools.combinations(extVariableTest, r)
    label_comb = itertools.combinations(extVariableLabel, r)
    for train, test, label in zip(train_comb, test_comb, label_comb):
        UKwED = UniversalKriging(
            xTrain,
            yTrain,
            moistureTrain,
            drift_terms=["specified"],
            specified_drift=list(train)
        )

        prediction, sigmasq = UKwED.execute('points', xTest, yTest, specified_drift_arrays=list(test))

        R2Params.append(list(label))
        R2Models.append("Universal Krigging with External Drift")
        R2test.append(r2_score(moistureTest, prediction))
        RSME.append(np.sqrt(mean_squared_error(moistureTest * scaler.scale_[2] + scaler.mean_[2], prediction * scaler.scale_[2] + scaler.mean_[2])))
        MAE.append(mean_absolute_error(moistureTest * scaler.scale_[2] + scaler.mean_[2], prediction * scaler.scale_[2] + scaler.mean_[2]))

R2Results = pandas.DataFrame({
    "Model":R2Models,
    "Input Parameters":R2Params,
    "R2 Score":R2test,
    "RMSE": RSME,
    "MAE": MAE
})

top5Results = R2Results.groupby('Model').apply(lambda x: x.nlargest(20, 'R2 Score')).reset_index(drop=True)
top5Results.to_csv("./Output_Data/Krigging_Results.csv")