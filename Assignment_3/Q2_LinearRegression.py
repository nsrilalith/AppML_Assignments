from sklearn import linear_model as linmod
from sklearn import metrics
from sklearn import preprocessing as preproc
import pandas as pd
import numpy as np

def showStats(W, X, Yact, Ypred):
    print("R2 = %f, MSE = %f" % (mlr.score(X, Yact), metrics.mean_squared_error(Yact, Ypred)))
    print("W: ", W)


fileName = "BattingSalariesData.xlsx" # read from Excel file
targetName = "salary"
IDName = ["playerID", "yearPlsyerID"]
doScale = True
dataFrame = pd.read_excel(fileName, sheet_name='extended')
trainX = dataFrame.drop([IDName, targetName], axis=1).to_numpy()

if (doScale):
    scalerX = preproc.MinMaxScaler(feature_range=(-1, 1))
    scalerX.fit(trainX)
    trainX = scalerX.transform(trainX)

trainY = dataFrame[targetName].to_numpy()

mlr = linmod.LinearRegression() # creates the regressor object
mlr.fit(trainX, trainY)
showStats(np.append(np.array(mlr.intercept_), mlr.coef_), trainX, trainY, mlr.predict(trainX))

poly = preproc.PolynomialFeatures(2) # object to generate polynomial basis functions
bigTrainX = poly.fit_transform(trainX)

mlr = linmod.LinearRegression() # creates the regressor object
mlr.fit(bigTrainX, trainY)
showStats(np.append(np.array(mlr.intercept_), mlr.coef_), bigTrainX, trainY, mlr.predict(bigTrainX))