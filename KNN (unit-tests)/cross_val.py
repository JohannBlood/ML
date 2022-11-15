import numpy as np
from collections import defaultdict


def kfold_split(num_objects, num_folds):
    res = []
    f_len = num_objects // num_folds
    for i in range(num_folds-1):
        res.append((np.asarray([j for j in range(num_objects) if j not in range(f_len*i, f_len*(i+1))]), np.asarray([k for k in range(f_len*i, f_len*(i+1))])))
    res.append((np.asarray([j for j in range(f_len*(i+1))]), np.asarray([j for j in range(f_len*(i+1), num_objects)])))
    return res


def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    res = dict()
    for l in parameters['normalizers']:
        for i in parameters['n_neighbors']:
            for j in parameters['metrics']:
                for k in parameters['weights']:
                    model = knn_class(n_neighbors=i, weights=k, metric=j)
                    mas = []
                    for train_idx, test_idx in folds:
                        if l[0] is not None:
                            l[0].fit(X[train_idx])
                            X_train, X_test = l[0].transform(X[train_idx]), l[0].transform(X[test_idx])
                        else:
                            X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        model.fit(X_train, y_train)
                        Z = model.predict(X_test)
                        mas.append(score_function(y_test, Z))
                    res[(l[1], i, j, k)] = np.mean(mas)
    return res
