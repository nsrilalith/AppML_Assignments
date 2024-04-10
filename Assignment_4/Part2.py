import pandas as pd
import numpy as np
from sklearn import preprocessing as preproc
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import model_selection as modelsel
from sklearn import neural_network as ann
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import time


def getData(path):
    data_frame = pd.read_excel(path)

    return data_frame


def doPreprocessing(data_frame):
    # 1. Separate features and target
    features = data_frame.drop(columns=['AGI'])  # Assuming 'AGI' is the target
    target = data_frame['AGI']

    # 2. Remove ID columns
    features = features.drop(columns=['HSUP_WGT', 'MARSUPWT', 'FSUP_WGT'])
    
    remaining_features = features.columns.tolist()

    binary_features = ['A_SEX', 'HAS_DIV'] # impute missing values by knn
    ordinal_features = ['PEINUSYR'] # impute missing values by knn
    categorical_features = ['PAW_YN', 'A_MARITL', 'PENATVTY'] # one hot encoding and impute missing values by knn

    numeric_features = set(remaining_features) - set(binary_features) - set(ordinal_features) - set(categorical_features)

    numeric_features = list(numeric_features)
    # print(numeric_features)

    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    # Preprocessing for ordinal and binary features: KNN imputation
    knn_transformer = Pipeline(steps=[
        ('imputer', KNNImputer())
    ])
    
    # Preprocessing for categorical features: One-hot encoding followed by KNN imputation
    categorical_transformer = Pipeline(steps=[
        ('onehot', preproc.OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('imputer', KNNImputer())
    ])
    
    # Bundle preprocessing for numeric and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('knn_b', knn_transformer, binary_features),
            ('knn_o', knn_transformer, ordinal_features),
            ('cat', categorical_transformer, ['PAW_YN', 'A_MARITL', 'PENATVTY'])
        ])
    
    # Create a preprocessing pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    # Fit and transform the features
    features_processed = pipeline.fit_transform(features)
    
    # Convert back into pandas DF
    features_processed_df = pd.DataFrame(features_processed)

    return features_processed_df, target
    


def doNormalize(X):
    scalerX = preproc.MinMaxScaler(feature_range=(-1,1))
    scalerX.fit(X)
    X_scaled = scalerX.transform(X)
    
    return X_scaled



def getMetrics(hl, clf, trainX, testX, trainY, testY):       
        # i. Architecture
        print(f"\nArchitecture (hidden layer sizes): {hl}")

        # ii. Number of epochs
        print(f"Number of epochs: {clf.n_iter_}")

        # iii. Training set metrics
        train_score = clf.score(trainX, trainY)
        train_mse = mean_squared_error(trainY, clf.predict(trainX))
        train_mae = mean_absolute_error(trainY, clf.predict(trainX))
        print(f"Training Set - Coefficient of determination (R^2): {train_score:.4f}, MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")

        # iv. Test set metrics
        test_score = clf.score(testX, testY)
        test_mse = mean_squared_error(testY, clf.predict(testX))
        test_mae = mean_absolute_error(testY, clf.predict(testX))
        print(f"Test Set - Coefficient of determination (R^2): {test_score:.4f}, MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")

        # v. Generalization gap (using R^2 for illustration)
        generalization_gap = train_score - test_score
        print(f"Generalization gap (R^2): {generalization_gap:.4f}\n") 



def getPlot(hl, clf):
    trainingLoss = np.asarray(clf.loss_curve_)
    validation_loss = np.sqrt(1 - np.asarray(clf.validation_scores_))
    factor = trainingLoss[1] / validation_loss[1]
    validation_loss = validation_loss*factor

    # Plot setup
    xlabel = "epochs (hl=" + str(hl) + ")"
    fig, ax = plt.subplots()

    # Plot training loss on the primary y-axis
    ax.plot(trainingLoss, color="blue", label='Training Loss')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Training Loss", color="blue", fontsize=10)

    # Create a secondary y-axis for validation loss
    ax2 = ax.twinx()
    ax2.plot(validation_loss, color="red", label='Validation Loss')
    ax2.set_ylabel("Validation Loss", color="red", fontsize=10)

    # Set y-axis scale
    ax.set_yscale('log')
    ax2.set_yscale('log')

    # Show plot with proper layout
    fig.tight_layout()
    plt.show()


def doANNRegression(feature, target):
    trainX, testX, trainY, testY = modelsel.train_test_split(feature, target, test_size=0.3, random_state=241)

    hidden_layers =[(4,4), (10,6), (32,16), (8,3,5), (12,9,10)]

    for hl in hidden_layers:
        clf = ann.MLPRegressor(hidden_layer_sizes=hl, activation='relu', early_stopping=True, tol=0.0005, alpha=0.0001, max_iter=1000)
        clf.fit(trainX, trainY)
        getMetrics(hl, clf, trainX, testX, trainY, testY)
        getPlot(hl, clf)


if __name__ == '__main__':
    start = time.time()
    df = getData('Census_Supplement_Data.xlsx')
    end = time.time()
    print('Time taken to read excel file: {:.4f} seconds'.format(end - start))
    # print(f'\nData:\n{df.head(10)}')

    # Preprocessing
    X, Y = doPreprocessing(df)
    # print(type(X), type(Y))

    # Normalization
    X_norm = doNormalize(X)
    # print(type(X_norm))

    # ANN
    doANNRegression(X_norm, Y)



