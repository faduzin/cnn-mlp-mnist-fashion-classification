import pandas as pd
import numpy as np

def cnn_reshape(X):
    try:
        X = X.reshape(-1, 28, 28, 1)
        print('Data reshaped successfully.')
    except Exception as e:
        print('Error reshaping data:', e)
        return None

    return X


def data_normalize(X):
    try:
        X = X / 255.0
        print('Data normalized successfully.')
    except Exception as e:
        print('Error normalizing data:', e)
        return None

    return X


def data_preprocess(data, cnn=True):
    try:
        if cnn:
            X = data.iloc[:, 1:].values
            X = cnn_reshape(X)
            y = data.iloc[:, 0].values
            X = data_normalize(X)
        else:
            X = data.iloc[:, 1:].values
            y = data.iloc[:, 0].values
            X = data_normalize(X)
        print('Data preprocessed successfully.')
    except Exception as e:
        print('Error preprocessing data:', e)
        return None

    return X, y