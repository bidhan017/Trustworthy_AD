import os
import pickle

import numpy as np
import pandas as pd
from sklearn import preprocessing

from .dataset import Dataset


class RealDataset(Dataset):
    def __init__(self, raw_path, **kwargs):
        super().__init__(**kwargs)
        self.raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/raw/", raw_path)


class RealPickledDataset(Dataset):
    """Class for pickled datasets from https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection"""

    def __init__(self, name, training_path):
        self.name = name
        self.training_path = training_path
        self.test_path = self.training_path.replace("train", "test")
        self._data = None

    def data(self):
        if self._data is None:
            with open(self.training_path, 'rb') as f:
                X_train = pd.DataFrame(pickle.load(f))
                X_train = X_train.iloc[:, :-1]

            #MinMax Normalization
            #scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            #names = X_train.columns
            #d = scaler.fit_transform(X_train)
            #X_train = pd.DataFrame(d, columns=names)


            #Z-score normalization
            mean, std = X_train.mean(), X_train.std()
            X_train = (X_train - mean) / std


            with open(self.test_path, 'rb') as f:
                X_test = pd.DataFrame(pickle.load(f))
            y_test = X_test.iloc[:, -1]
            X_test = X_test.iloc[:, :-1]
            #X_test = (X_test - mean) / std

            #Min-Max norm
            #scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            #names1 = X_test.columns
            #d1 = scaler1.fit_transform(X_test)
            #X_test = pd.DataFrame(d1, columns=names1)
            #print(np.zeros(len(X_train)))
            #print(y_test)

            self._data = X_train, np.zeros(len(X_train)), X_test, y_test
        return self._data
