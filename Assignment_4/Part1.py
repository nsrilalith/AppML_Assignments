import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing as preproc
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import time

from sklearn.preprocessing import PolynomialFeatures


def getDF(path):
    df = pd.read_excel(path)
    return df


def getNormalized_and_train_test(df):
    # Separate features and target
    X = df.iloc[:, :-1]  # All rows, exclude the last column
    y = df.iloc[:, -1]  # All rows, just the last column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 2: Fit the scaler on the training data
    scaler = preproc.MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train)  # Compute the min and max values to be used for scaling

    # Step 3: Transform both the training and test data with the fitted scaler
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def getPolyTransform(X_train, X_test):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    return X_train_poly, X_test_poly


def getLogTransform(X_train, X_test):
    X_train_log = np.log(X_train + 1 - X_train.min())
    X_test_log = np.log(X_test + 1 - X_test.min())

    return X_train_log, X_test_log


def getCombinationTransform(X_train_poly, X_test_poly, X_train_log, X_test_log):
    X_train_combo = np.hstack([X_train_poly, X_train_log])
    X_test_combo = np.hstack([X_test_poly, X_test_log])

    return X_train_combo, X_test_combo


def logisticRegression(X_train, Y_train):
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, Y_train)

    return logistic_model


def getMetrics(model, X_test, Y_test):
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

    # Evaluation Metrics
    n_iterations = model.n_iter_[0]
    accuracy = accuracy_score(Y_test, y_pred)
    roc_auc = roc_auc_score(Y_test, y_pred_proba)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    conf_matrix = confusion_matrix(Y_test, y_pred)

    # Confusion matrix components
    tn, fp, fn, tp = conf_matrix.ravel()
    tpr = tp / (tp + fn)  # True Positive Rate
    fnr = fn / (fn + tp)  # False Negative Rate
    fpr = fp / (fp + tn)  # False Positive Rate
    tnr = tn / (tn + fp)  # True Negative Rate

    # Print metrics
    print(f'Classification test: [{n_iterations}] iterations', end=', ')
    print(f'accuracy: {accuracy:.4f}', end=', ')
    print(f'AUC: {roc_auc:.4f}')
    print(f'Precision: {precision:.6f}', end=', ')
    print(f'Recall: {recall:.6f}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'TPR: {tpr:.4f}, FNR: {fnr:.4f}, FPR: {fpr:.4f}, TNR: {tnr:.4f}')


if __name__ == '__main__':
    start = time.time()
    df1 = getDF('VWXYZ.xlsx')
    end = time.time()
    print('Time it took to read the excel file: ', end - start)
    # print(df1)

    X_train_scaled, X_test_scaled, y_train, y_test = getNormalized_and_train_test(df1)
    originalModel = logisticRegression(X_train_scaled, y_train)
    print('\n\nMetrics for Original Dataset\n')
    getMetrics(originalModel, X_test_scaled, y_test)

    X_poly_train, X_poly_test = getPolyTransform(X_train_scaled, X_test_scaled)
    polyModel = logisticRegression(X_poly_train, y_train)
    print('\n\nMetrics for Polynomial Deg 2 Transformed Dataset\n')
    getMetrics(polyModel, X_poly_test, y_test)

    X_log_train, X_log_test = getLogTransform(X_train_scaled, X_test_scaled)
    logModel = logisticRegression(X_log_train, y_train)
    print('\n\nMetrics for Log Transformation Dataset\n')
    getMetrics(logModel, X_log_test, y_test)

    X_combi_train, X_combi_test = getCombinationTransform(X_poly_train, X_poly_test, X_log_train, X_log_test)
    comboModel = logisticRegression(X_combi_train, y_train)
    print('\n\nMetrics for Combination Transformation\n')
    getMetrics(comboModel, X_combi_test, y_test)

