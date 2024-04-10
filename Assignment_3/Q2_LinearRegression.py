from sklearn import linear_model as linmod
from sklearn import metrics, model_selection, feature_selection
from sklearn import preprocessing as preproc
from sklearn import impute
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def showStats(W, X, Yact, Ypred, mlr):
    """Shows stats about the linear regression(Used from lecture notes)"""
    print("R2 = %f, MSE = %f" % (mlr.score(X, Yact), metrics.mean_squared_error(Yact, Ypred)))
    print("W: ", W)


def GetDF(path):
    """Just reads the excel file and returns it as a Dataframe"""
    DF = pd.read_excel(path)
    return DF


def prerocessDF(df):
    """"PREPROCESS STEPS"""

    """Dropping ID Columns"""
    IDName = ["playerID", "yearPlayer"]
    df.drop(columns=IDName, inplace=True)

    """Splitting Target and Features right here to make imputation easier"""
    #X = df.drop(columns=targetName, axis=1)
    #Y = df[targetName]

    """"Handling Categorical Columns"""

    """I decided to use OneHotEncoding, since the cardinality of categorical columns is not too high (35, 2)"""
    object_cols = df[['lgID', 'teamID']]  # Selecting the columns to be encoded
    encoder = preproc.OneHotEncoder()
    encoded_objects = encoder.fit_transform(object_cols)
    encoded_df = pd.DataFrame(encoded_objects.toarray(), columns=encoder.get_feature_names_out(['lgID', 'teamID']))
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df.drop(['lgID', 'teamID'], axis=1), encoded_df], axis=1)

    """Handle Missing Values and Outliers"""

    # I ran the Data_statistics.py report generator from HW1 on BattingSalariesData.xlsx, so I know in which column
    # the oultiers are, and that they are 99999 and 9999,
    # I also know that the target Salary has about a 3rd of its values missing(from the report)
    """So I decided to use the KNN imputator for outliers and missing values"""
    df.replace([99999, 9999], np.nan, inplace=True)

    # Initialize the KNNImputer
    KNNimputer = impute.KNNImputer(missing_values=np.nan, n_neighbors=5)
    transform_df = KNNimputer.fit_transform(df)
    df_imputed = pd.DataFrame(transform_df, columns=df.columns, index=df.index) #converting the np array back into a dataframe
    # Y_imputed_series = pd.Series(Y_imputed.flatten())
    
    return df_imputed


def tryVariableSelection(xtrain, xtest, ytrain, ytest, sel, dir, labels, model):
    
    if sel == 'sequential':
        selector = feature_selection.SequentialFeatureSelector(model, direction=dir, n_features_to_select='auto', scoring='r2')
    elif sel == 'RFE':
        selector = feature_selection.RFE(model, step=1)
    elif sel == 'RFECV':                                      # This works Best
        selector = feature_selection.RFECV(model, step=1, cv=5)
    
    selector.fit(xtrain, ytrain)
    newxtrain = selector.transform(xtrain)
    newxtest = selector.transform(xtest)
    model.fit(newxtrain, ytrain)
    print("\nUsing: {0}".format(labels[selector.get_support() == True]))
    print("Method {0}: Training set R-sq={1:8.5f}, test set MSE={2:e}".format(dir, model.score(newxtrain,
    ytrain), metrics.mean_squared_error(ytest, model.predict(newxtest))))


def linregTotal(X, Y, labels):
    doScale = True  

    trainX, testX, trainY, testY = model_selection.train_test_split(X, Y, test_size=0.2, random_state=22222)

    if doScale:
        scalerX = preproc.MinMaxScaler(feature_range=(-1, 1))
        scalerX.fit(trainX)
        trainX = scalerX.transform(trainX)
        scalerX.fit(testX)
        testX = scalerX.transform(testX)

    mlr = linmod.LinearRegression()  # creates the regressor object
    
    #Uncomment for Vanilla Linear Regression
    '''mlr.fit(trainX, trainY)
    print("On Train Set\n")
    print("R2 is {}; RMSE is {}".format(mlr.score(trainX, trainY), metrics.mean_squared_error(trainY, mlr.predict(trainX))))
    print("W = ", mlr.intercept_, mlr.coef_)
    print("\nOn Test Set\n")
    print("R2 is {}; RMSE is {}".format(mlr.score(testX, testY), metrics.mean_squared_error(testY, mlr.predict(testX))))
    print("W = ", mlr.intercept_, mlr.coef_)'''

    #tryVariableSelection(trainX, testX, trainY, testY, 'sequential', 'forward', labels, model=mlr)
    #tryVariableSelection(trainX, testX, trainY, testY, 'sequential', 'backward', labels, model=mlr)
    #tryVariableSelection(trainX, testX, trainY, testY, 'RFE', 'RFE', labels, model=mlr)

    tryVariableSelection(trainX, testX, trainY, testY, 'RFECV', 'RFECV', labels, model=mlr) #This was giving me the best result so I Ended up using only RFECV

    #Uncomment if you want to try ridge regression
    '''print("\n\nUsing Ridge Regression")
    ridge_reg = linmod.Ridge(alpha=10, solver='sag', random_state=22222)
    ridge_reg.fit(trainX, trainY)
    print("Training complete; {} epochs".format(ridge_reg.n_iter_))
    print("On Train Set\n")
    print("R2 is {}; RMSE is {}".format(ridge_reg.score(trainX, trainY), metrics.mean_squared_error(trainY, ridge_reg.predict(trainX))))
    print("W = ", ridge_reg.intercept_, ridge_reg.coef_)
    print("\nOn Test Set\n")
    print("R2 is {}; RMSE is {}".format(ridge_reg.score(testX, testY), metrics.mean_squared_error(testY, ridge_reg.predict(testX))))
    print("W = ", ridge_reg.intercept_, ridge_reg.coef_)'''

    #Uncomment If you want to try polynomial feauture enhanced regression
    '''poly = preproc.PolynomialFeatures(2)  # object to generate polynomial basis functions
    bigTrainX = poly.fit_transform(trainX)
    mlr = linmod.LinearRegression()  # creates the regressor object
    mlr.fit(bigTrainX, trainY)
    showStats(np.append(np.array(mlr.intercept_), mlr.coef_), bigTrainX, trainY, mlr.predict(bigTrainX), mlr)'''



def LinRegByYear(df):
    # Initialize lists to store results
    years = []
    dataset_sizes = []
    mse_values = []
    r2_values = []

    # Print header for the outputs
    print(f"{'Year':<6} {'Size':<6} {'MSE':<20} {'R2':<6}")

    # Loop through each year
    for year in df['yearID'].unique():
        # Filter the data for the given year
        yearly_data = df[df['yearID'] == year]

        # Define the predictors and target variable
        X = yearly_data.drop(['Salary', 'yearID'], axis=1)
        y = yearly_data['Salary']

        # Split the data
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=22222)

        #Scaler Normalization USed from Class notes
        scalerX = preproc.MinMaxScaler(feature_range=(-1, 1))
        scalerX.fit(X_train)
        X_train = scalerX.transform(X_train)
        scalerX.fit(X_test)
        X_test = scalerX.transform(X_test)

        # Initialize Linear Regression
        model = linmod.LinearRegression()

        #Using RFECV variable selection (gave me best result in previous reg)
        selector = feature_selection.RFECV(model, step=1, cv=5)
        selector.fit(X_train, y_train)
        newxtrain = selector.transform(X_train)
        newxtest = selector.transform(X_test)

        #Fitting Model
        model.fit(newxtrain, y_train)
        predictions = model.predict(newxtest)
        
        # Evaluation
        mse = metrics.mean_squared_error(y_test, predictions)
        r2 = metrics.r2_score(y_test, predictions)

        # Store the results
        years.append(year)
        dataset_sizes.append(len(yearly_data))
        mse_values.append(mse)
        r2_values.append(r2)
        
        # Print results for each year
        print(f"{year:<6} {len(yearly_data):<6} {mse:<20} {r2:<6}")
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(years, mse_values, marker='x')
    plt.title('MSE vs Year for Salary Prediction')
    plt.xlabel('Year')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    path = "BattingSalariesData.xlsx"

    print("Creating DF . . .")
    df = GetDF(path)
    print("Got Dataframe ", type(df), "Size:", df.shape)
    print("\nPerforming pre-processing . . .")
    ndf = prerocessDF(df)
    print("Pre-Processed Dataframe ", type(df), "Size:", df.shape)
    X = ndf.drop(columns='Salary', axis=1)
    Y = ndf['Salary']
    labels = X.columns
    print("\nLinear Regression On Total Dataset. . .\n")
    linregTotal(X, Y, labels)
    
    print("\nLinear Regression By Year . . .\n")
    LinRegByYear(ndf)
    
