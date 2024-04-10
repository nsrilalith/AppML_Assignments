import pandas as pd
import numpy as np
from sklearn import preprocessing as preproc
import time


def getData(path):
    data_frame = pd.read_excel(path)

    return data_frame


def doPreprocessing(data_frame):
    # Separate features and target
    X = df.iloc[:, :-1]  # All rows, exclude the last column
    y = df.iloc[:, -1]  # All rows, just the last column


if __name__ == '__main__':
    start = time.time()
    df = getData('Census_Supplement_Data.xlsx')
    end = time.time()
    print('Time taken to read excel file: {:.4f} seconds'.format(end - start))
    # print(f'\nData:\n{df.head(10)}')
