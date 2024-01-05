import numpy as np
from pykrige.uk import UniversalKriging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pandas
import matplotlib.pyplot as plt

np.random.seed(39)

dataframe = pandas.read_csv("Data1.csv")
dataArray = dataframe.to_numpy()
scaler = StandardScaler().fit(dataArray)
scaledData = scaler.transform(dataArray)

x = scaledData[:,0]
y = scaledData[:,1]
dem = scaledData[:,3]
profCurve = scaledData[:,4]
slope = scaledData[:,5]
pinCurv = scaledData[:,6]
tpi = scaledData[:,7]
twi = scaledData[:,8]
texture = scaledData[:,9]

moisture = scaledData[:,2]

xTrain, xTest, yTrain, yTest, moistureTrain, moistureTest, demTrain, demTest, profCurvTrain, profCurveTest, slopeTrain, slopeTest, pinCurvTrain, pinCurvTest, tpiTrain, tpiTest, twiTrain, twiTest, textureTrain, textureTest = train_test_split(x, y, moisture, dem, profCurve, slope, pinCurv, tpi, twi, texture, test_size=.3)

UKwED = UniversalKriging(
    xTrain,
    yTrain,
    moistureTrain,
    drift_terms=["specified"],
    variogram_model='linear',
    specified_drift=[slopeTrain, demTrain, textureTrain, tpiTrain]
)

predictions, sigmasq = UKwED.execute('points', xTest, yTest, specified_drift_arrays=[slopeTest, demTest, textureTest, tpiTest])

print("UK With External Drift")
mae = mean_absolute_error(moistureTest, predictions)
print("MAE:", mae)
mse = mean_squared_error(moistureTest, predictions)
print("MSE:", mse)
evs = explained_variance_score(moistureTest, predictions)
print("EVS:", evs)
r2 = r2_score(moistureTest, predictions)
print("R2:", r2)

UK = UniversalKriging(
    xTrain,
    yTrain,
    moistureTrain,
    variogram_model='linear'
)

predictionsWOD, sigmasq = UK.execute('points', xTest, yTest)

print("UK Without External Drift")
mae = mean_absolute_error(moistureTest, predictionsWOD)
print("MAE:", mae)
mse = mean_squared_error(moistureTest, predictionsWOD)
print("MSE:", mse)
evs = explained_variance_score(moistureTest, predictionsWOD)
print("EVS:", evs)
r2 = r2_score(moistureTest, predictionsWOD)
print("R2:", r2)

fig, ax = plt.subplots(1,1)
ax.scatter(xTrain,yTrain, color="red")
ax.scatter(xTest,yTest, color="blue")

for (xi, yi, mi) in zip(xTrain, yTrain, moistureTrain):
    ax.text(xi, yi, round(mi, 3), va="bottom", ha="center")
for (xi, yi, mi, mti) in zip(xTest, yTest, predictions, moistureTest):
    ax.text(xi, yi, "p:" + str(round(mi, 3)), va="bottom", ha="center")
    ax.text(xi, yi, "a:" + str(round(mti, 3)), va="top", ha="center")

fig,ax = plt.subplots(1,1)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.scatter(moistureTest, predictions)

fig,ax = plt.subplots(2,1) 

nums = np.arange(0, len(moistureTest))

ax[0].scatter(nums, moistureTest)
ax[0].scatter(nums, predictions)
ax[0].legend(["Validation Values", "Prediction Values"])

for num, res in zip(nums, moistureTest - predictions):
    if(abs(res) > .5):
        ax[1].scatter(num, res, color="red")
    else:
        ax[1].scatter(num, res, color="blue")

ax[1].axhline(y=0)
ax[1].legend(["Residual"])

resid = np.zeros(len(moisture))

for i in range(0, len(moisture)):
    for j in range(0, len(moistureTest)):
        if(moisture[i] == moistureTest[j]):
            resid[i] = moistureTest[j] - predictions[j]

resid = np.array([resid]).T
appendedData = np.concatenate((scaledData, resid), axis=1)

df = pandas.DataFrame(appendedData, columns=["X","Y","VWC","DEM","ProfCurv","Slope","PlnCurv","TPI","TWI","Texture","VWC","Residual"])
df.to_csv("processedData.csv", index=False)

plt.show()
