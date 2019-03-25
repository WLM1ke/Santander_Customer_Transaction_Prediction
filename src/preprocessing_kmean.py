"""Процессинг исходных данных."""
import pandas as pd
import numpy as np
from sklearn import cluster, preprocessing

from src import conf

DTYPES = dict(target="int8", **{f"var_{i}": "float32" for i in range(200)})


def read_train():
    """Загружает обучающие данные."""
    df = pd.read_csv(conf.DATA_TRAIN, index_col=0, dtype=DTYPES)
    return df.iloc[:, 1:], df.iloc[:, 0]


def read_test():
    """Загружает тестовые данные."""
    df = pd.read_csv(conf.DATA_TEST, index_col=0, dtype=DTYPES)
    return df


def make_features():
    """Добавляет признаки к модели."""
    x_train, y_train = read_train()
    x_test = read_test()
    x_all = pd.concat([x_train, x_test], axis=0)
    x_all = pd.DataFrame(preprocessing.StandardScaler().fit_transform(x_all), columns=x_all.columns)

    n_clusters = 10  # #########
    clf = cluster.KMeans(n_clusters=n_clusters, random_state=conf.SEED, n_jobs=-1)
    cluster_id = clf.fit_predict(x_all).reshape((-1, 1))
    cluster_dist = clf.transform(x_all)
    cluster_feat = np.hstack((cluster_id, cluster_dist, x_all.values))
    x_all = pd.DataFrame(cluster_feat, index=x_all.index)

    return x_all.iloc[: len(y_train)], y_train, x_all.iloc[len(y_train):]


def augment(x, y, t=2):
    """Аугаментация данных."""
    xs, xn = [], []
    for i in range(t):
        mask = y > 0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:, c] = x1[ids][:, c]
        xs.append(x1)

    for i in range(t // 2):
        mask = y == 0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:, c] = x1[ids][:, c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x, xs, xn])
    y = np.concatenate([y, ys, yn])
    return x, y
