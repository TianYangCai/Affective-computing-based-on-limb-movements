"""
Processing the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

def process_data(train, test):
    files_train = os.listdir(train)
    X_train, y_train = [], []
    for file in files_train:  # 遍历文件夹
        arousal = int(file.split('.')[0].split('_')[1])
        valence = int(file.split('.')[0].split('_')[2])
        y_train.append([arousal,valence])
        position = train + '/' + file
        df1 = pd.read_csv(position, encoding='utf-8')
        df1 = np.array(df1)
        X_train.append(df1)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    files_test = os.listdir(test)
    X_test, y_test = [], []
    for file in files_test:  # 遍历文件夹
        arousal = int(file.split('.')[0].split('_')[1])
        valence = int(file.split('.')[0].split('_')[2])
        y_test.append([arousal,valence])
        position = test + '/' + file
        df1 = pd.read_csv(position, encoding='utf-8')
        df1 = np.array(df1)
        X_test.append(df1)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test
