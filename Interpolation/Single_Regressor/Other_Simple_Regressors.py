import itertools
import pandas
import numpy as np

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

rs = np.random.RandomState(22)

df = pandas.read_csv("./input_data/in-situ_Data.csv")
columns = list(df.columns.values)
scaler = StandardScaler()
npdf = scaler.fit(df.to_numpy()).transform(df.to_numpy())
df = pandas.DataFrame(npdf, columns=columns)

models = [
    KNeighborsRegressor(),
    AdaBoostRegressor(random_state=rs),
    SVR(), 
    DecisionTreeRegressor(random_state=rs), 
    ExtraTreesRegressor(random_state=rs),
    RandomForestRegressor(random_state=rs), 
    MLPRegressor(random_state=rs),
    GradientBoostingRegressor(random_state=rs),
    GaussianProcessRegressor(random_state=rs)
]

VWC = np.squeeze(df[["VWC"]].to_numpy())
imputDataframe = df.drop(columns=["X", "Y", "VWC", "Slope"])
values = list(imputDataframe.columns.values)

R2Params = []
R2Models = []
R2test = []
RSME = []
MAE = []

print("Testing models", end=" ", flush=True)

for m in models:
    print(".", end=" ")
    np.random.seed(22)
    np.random.set_state(np.random.get_state())
    for r in range(1, 10):
        combinations = itertools.combinations(values, r)
        for combination in combinations:
            inputColumns = list(combination)
            inputs = df[inputColumns].to_numpy()

            inputsTrain, inputsTest, VWCTrain, VWCTest = train_test_split(inputs, VWC, test_size=.3)
            
            np.random.seed(22)
            np.random.set_state(np.random.get_state())

            m.fit(inputsTrain, VWCTrain)

            np.random.seed(22)
            np.random.set_state(np.random.get_state())

            R2Params.append(inputColumns)
            R2Models.append(type(m).__name__)

            predictions = m.predict(inputsTest)

            R2test.append(r2_score(VWCTest, predictions))
            RSME.append(np.sqrt(mean_squared_error(VWCTest * scaler.scale_[2] + scaler.mean_[2], predictions * scaler.scale_[2] + scaler.mean_[2])))
            MAE.append(mean_absolute_error(VWCTest * scaler.scale_[2] + scaler.mean_[2], predictions * scaler.scale_[2] + scaler.mean_[2]))

print("", end="\n")

R2Results = pandas.DataFrame({
    "Model":R2Models,
    "Input Parameters":R2Params,
    "R2 Score":R2test,
    "RMSE": RSME,
    "MAE": MAE
})

top5Results = R2Results.groupby('Model').apply(lambda x: x.nlargest(5, 'R2 Score')).reset_index(drop=True)
top5Results.to_csv("./output_data/Other_Simple_Regressor_Results.csv")
            