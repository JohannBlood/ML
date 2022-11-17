import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        # your code here
        res = list()
        x = X.to_numpy()
        h, w = x.shape
        for j in range(w):
            a = []
            for i in range(h):
                if x[i][j] not in a:
                    a.append(x[i][j])
            res.append(sorted(a))
        self.x_un = res

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        # your code here
        fl = False
        x = X.to_numpy()
        h, w = x.shape
        for j in range(w):
            a = np.zeros((h, len(self.x_un[j])))
            for i in range(h):
                for index, elem in enumerate(self.x_un[j]):
                    if elem == x[i][j]:
                        a[i][index] = 1
                        break
            if fl:
                res = np.concatenate((res, a), axis=1)
            else:
                fl = True
                res = a
        return res

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        # your code here
        x = X.to_numpy()
        y = Y.to_numpy()
        res = []
        h, w = x.shape
        for j in range(w):
            a = {}
            for i in range(h):
                if x[i][j] not in a:
                    c, s = 0, 0
                    for k in range(h):
                        if x[k][j] == x[i][j]:
                            c += 1
                            s += y[k]
                    s = s / c
                    c = c / h
                    a[x[i][j]] = np.array([s, c, 0])
            res.append(a)
        self.dic = res

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        # your code here
        fl = False
        x = X.to_numpy()
        h, w = x.shape
        for j in range(w):
            buf = np.zeros((h, 3))
            for i in range(h):
                buf[i] = self.dic[j][x[i][j]]
                buf[i][2] = (buf[i][0] + a) / (buf[i][1] + b)
            if fl:
                res = np.concatenate((res, buf), axis=1)
            else:
                res = buf
                fl = True
        return res

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        # your code here
        self.gr = group_k_fold(X.shape[0], self.n_folds, seed)
        res = []
        for i, j in self.gr:
            counter = SimpleCounterEncoder()
            counter.fit(X.iloc[j], Y.iloc[j])
            res.append((i, counter))
        self.res = res

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        # your code here
        fl = False
        for i, j in self.res:
            con = np.concatenate((j.transform(X.iloc[i], a, b), np.reshape(np.array(i), (len(i), 1))), axis=1)
            if fl:
                buf = np.concatenate((buf, con), axis=0)
            else:
                fl = True
                buf = con
        buf = buf[buf[:, -1].argsort()]
        return np.delete(buf, -1, 1)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    # your code here
    mn = set(x)
    res = np.array([0.0]*len(mn))
    for i, j in enumerate(mn):
        res[i] = sum(y[x == j]) / list(x).count(j)
    return res
