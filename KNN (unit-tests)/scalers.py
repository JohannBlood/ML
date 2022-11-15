import numpy as np


class MinMaxScaler:
    mini, ro = np.asarray([]), np.asarray([])

    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.mini = np.asarray([min(i) for i in data.T])
        self.ro = np.asarray([max(i) - min(i) for i in data.T])

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        return (data - self.mini) / self.ro


class StandardScaler:
    e, otkl = np.asarray([]), np.asarray([])

    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.e = np.asarray([sum(i) / len(i) for i in data.T])
        self.otkl = np.asarray([sum([(j - sum(i) / len(i)) ** 2 for j in i]) / len(i) for i in data.T])

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        return (data - self.e) / self.otkl ** 0.5
