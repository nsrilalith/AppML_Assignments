from sklearn import linear_model as linmod
from sklearn import metrics, model_selection
from sklearn import preprocessing as preproc
from sklearn import impute
import pandas as pd
import numpy as np


def showStats(W, X, Yact, Ypred, mlr):
    """Shows stats about the linear regression(Used from lecture notes)"""
    print("R2 = %f, MSE = %f" % (mlr.score(X, Yact), metrics.mean_squared_error(Yact, Ypred)))
    print("W: ", W)


def GetDF(path):
    """Just reads the excel file and returns it as a Dataframe"""
    DF = pd.read_excel(path)
    return DF


def prerocessDF(DF):
    """"PREPROCESS STEPS"""

    """Dropping ID Columns"""
    targetName = "Salary"
    IDName = ["playerID", "yearPlayer"]
    DF.drop(columns=IDName, inplace=True)

    """Splitting Target and Features right here to make imputation easier"""
    X = DF.drop(columns=targetName, axis=1)
    Y = DF[targetName]

    """"Handling Categorical Columns"""

    """I decided to use OneHotEncoding, since the cardinality of categorical columns is not too high (35, 2)"""
    object_cols = X[['lgID', 'teamID']]  # Selecting the columns to be encoded
    encoder = preproc.OneHotEncoder()
    encoded_objects = encoder.fit_transform(object_cols)
    encoded_df = pd.DataFrame(encoded_objects.toarray(), columns=encoder.get_feature_names_out(['lgID', 'teamID']))
    X.reset_index(drop=True, inplace=True)
    X = pd.concat([X.drop(['lgID', 'teamID'], axis=1), encoded_df], axis=1)

    """Handle Missing Values and Outliers"""

    # I ran the Data_statistics.py report gen from HW1 on BattingSalariesData.xlsx, so I know in which column
    # the oultiers are, and that they are 99999 and 9999,
    """So I decided to use the Simple Imputator for outlier with a mean strategy"""
    X.replace([99999, 9999], np.nan, inplace=True)
    outlier_impute = impute.SimpleImputer(missing_values=np.nan, strategy="mean")
    X = outlier_impute.fit_transform(X)

    # I also know that the target Salary has about a 3rd of its values missing(from the report)
    """So for missing values in target column, I decided to use KNNimputer"""
    Y_reshaped = Y.values.reshape(-1, 1)
    # Initialize the KNNImputer
    KNNimputer = impute.KNNImputer(n_neighbors=3)
    # Use the imputer to fill in missing values in Y
    Y_imputed = KNNimputer.fit_transform(Y_reshaped)
    # If you need the result as a Series again
    Y_imputed_series = pd.Series(Y_imputed.flatten())

    return X, Y_imputed_series


def linregTotal(X, Y):
    doScale = False

    trainX, testX, trainY, testY = model_selection.train_test_split(X, Y, test_size=0.3, random_state=22222)

    if doScale:
        scalerX = preproc.MinMaxScaler(feature_range=(-1, 1))
        scalerX.fit(trainX)
        trainX = scalerX.transform(trainX)

    mlr = linmod.LinearRegression()  # creates the regressor object
    mlr.fit(trainX, trainY)
    showStats(np.append(np.array(mlr.intercept_), mlr.coef_), trainX, trainY, mlr.predict(trainX), mlr)

    poly = preproc.PolynomialFeatures(2)  # object to generate polynomial basis functions
    bigTrainX = poly.fit_transform(trainX)

    mlr = linmod.LinearRegression()  # creates the regressor object
    mlr.fit(bigTrainX, trainY)
    showStats(np.append(np.array(mlr.intercept_), mlr.coef_), bigTrainX, trainY, mlr.predict(bigTrainX), mlr)


if __name__ == '__main__':
    path = "/Users/srilalithnampally/Classes/AppML_Assignments/Assignment_3/BattingSalariesData.xlsx"

    print("Creating DF . . .\n")
    df = GetDF(path)

    print("Performing pre-processing . . .\n")
    feature, target = prerocessDF(df)

    print("Linear Regression . . .\n")
    linreg(feature, target)
