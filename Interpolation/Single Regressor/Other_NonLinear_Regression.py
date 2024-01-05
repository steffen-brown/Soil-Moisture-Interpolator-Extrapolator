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

rs = np.random.RandomState(22)


df = pandas.read_csv("Final.csv")
columns = list(df.columns.values)
npdf = StandardScaler().fit(df.to_numpy()).transform(df.to_numpy())
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
imputDataframe = df.drop(columns=["X", "Y", "VWC"])

values = list(imputDataframe.columns.values)

R2Params = []
R2Models = []
R2test = []
RSMe = []
MAe = []
R2train = []

for m in models:
    np.random.seed(22)
    np.random.set_state(np.random.get_state())
    for r in range(2, 20):
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
            score = m.score(inputsTest, VWCTest)
            R2test.append(score)
            score = r2_score(VWCTrain, m.predict(inputsTrain))
            R2train.append(score)

            score = mean_squared_error(VWCTrain, m.predict(inputsTrain))
            RSMe.append(np.sqrt(score))
            score = mean_absolute_error(VWCTrain, m.predict(inputsTrain))
            MAe.append(score)

            print("Testing " + str(type(m).__name__) + " with parameters " + str(inputColumns) + " with score " + str(score))

R2Results = pandas.DataFrame({
    "Model":R2Models,
    "Input Parameters":R2Params,
    "R2 (Test Dataset)":R2test,
    "RMSE": RSMe,
    "MAE": MAe
})

top5Results = R2Results.groupby('Model').apply(lambda x: x.nlargest(5, 'R2 (Test Dataset)')).reset_index(drop=True)
top1Results = R2Results.groupby('Model').apply(lambda x: x.nlargest(1, 'R2 (Test Dataset)')).reset_index(drop=True)

top5Results.to_csv("Top_5.csv")
top1Results.to_csv("Top_1.csv")
            