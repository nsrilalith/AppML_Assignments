import numpy as np
from sklearn import model_selection, neighbors
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


def getDF(path):
    """Returns the csv as a pandas dataframe"""
    print("Reading . . .")
    df = pd.read_csv(path, low_memory=False)
    print("Done")
    return df


def pp(pdframe):
    """Preprocessing Steps
    Input - Dataframe
    Output - Feature_DF, Target_DF"""
    pdframe.replace('?', 0, inplace=True)                           #Replacing '?' values with 0s, since its numeric column
    X = pdframe.drop(pdframe.columns[-1], axis=1)                   #Dropping the target column
    Y = pdframe.iloc[:, -1]                                         #Making the Target into a seperate DF
    X = X.to_numpy(np.float64)
    Y.replace(('inactive', 'active'), range(2), inplace=True)       #Converting target values to numbers
    Y = Y.to_numpy(np.float64)
    return X, Y


def KNN_unbalanced(feature, target, score_df):
    '''KNN for Unbalanced Dataset type'''
    trainX, testX, trainY, testY = model_selection.train_test_split(feature, target, test_size=0.3, random_state=92313)

    #iterate through neighbors and weighting
    for nnbrs in [1, 3, 5, 9, 11, 15, 21]:  # [1, 3, 5, 9, 11, 15, 21]
        for weighting in ['uniform', 'distance']:
            clf = neighbors.KNeighborsClassifier(n_neighbors=nnbrs, weights=weighting)
            clf = clf.fit(trainX, trainY.ravel())
            accuracy_test = clf.score(testX, testY)
            accuracy_train = clf.score(trainX, trainY)
            print(
                f"For {nnbrs}-NN Classifier on (Unbalanced) data with weighting type ({weighting}); \n accuracy for test_set = {accuracy_test:.4f} \n accuracy for train_set = {accuracy_train:.4f} \n difference in test_set & train_set performace = {abs(accuracy_train - accuracy_test):.4f}\n")
            row = {'Dataset': 'Unbalanced', 'Neighbors': nnbrs, 'Weighting': weighting, 'Accuracy_test': accuracy_test,
                   'Accuracy_train': accuracy_train, 'Diff in Train&Test Perf': abs(accuracy_train - accuracy_test)}
            score_df.loc[len(score_df.index)] = row


def KNN_blanaced(feature, target, score_df):
    '''KNN for Balanced Dataset'''
    rus = RandomUnderSampler(random_state=0)                        #Used randomundersampler to create the 50-50
    X_resampled, y_resampled = rus.fit_resample(feature, target)

    trainX, testX, trainY, testY = model_selection.train_test_split(X_resampled, y_resampled, test_size=0.3,
                                                                    random_state=92313)
    #iterate through neighbors and weighting
    for nnbrs in [1, 3, 5, 9, 11, 15, 21]:  # [1, 3, 5, 9, 11, 15, 21]
        for weighting in ['uniform', 'distance']:
            clf = neighbors.KNeighborsClassifier(n_neighbors=nnbrs, weights=weighting)
            clf = clf.fit(trainX, trainY.ravel())
            accuracy_test = clf.score(testX, testY)
            accuracy_train = clf.score(trainX, trainY)
            print(f"For {nnbrs}-NN Classifier on (Balanced) data with weighting type ({weighting}); \n accuracy for test_set = {accuracy_test:.4f} \n accuracy for train_set = {accuracy_train:.4f} \n difference in test_set & train_set performace = {abs(accuracy_train - accuracy_test):.4f}\n")
            row = {'Dataset': 'Balanced', 'Neighbors': nnbrs, 'Weighting': weighting, 'Accuracy_test': accuracy_test,
                   'Accuracy_train': accuracy_train, 'Diff in Train&Test Perf': abs(accuracy_train - accuracy_test)}
            score_df.loc[len(score_df.index)] = row


if __name__ == "__main__":
    filename = "K8.csv"
    print("Making a Dataframe . . .")
    df = getDF(filename)
    feature, target = pp(df)
    score_df = pd.DataFrame(
        columns=['Dataset', 'Neighbors', 'Weighting', 'Accuracy_test', 'Accuracy_train', 'Diff in Train&Test Perf'])
    print("Accuracy Results: \n")
    KNN_unbalanced(feature, target, score_df)
    KNN_blanaced(feature, target, score_df)
    print('Performance Table: \n')
    print(score_df)
    score_df.to_csv('K8_KNN_results.csv', index=False)
