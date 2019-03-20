"""Обучение моделей c помощью catboost."""
import logging
import time

import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn import metrics
from sklearn import model_selection

from src import conf
from src import processing

SEED = 284_702
N_SPLITS = 5
ITERATIONS = 30000
FOLDS = model_selection.StratifiedKFold(
    n_splits=N_SPLITS, shuffle=True, random_state=SEED
)

PARAMS = {
    "bagging_freq": 5,
    "bagging_fraction": 0.335,
    "boost_from_average": "false",
    "boost": "gbdt",
    "feature_fraction": 0.041,
    "learning_rate": 0.0083,
    "max_depth": -1,
    "metric": "auc",
    "min_data_in_leaf": 80,
    "min_sum_hessian_in_leaf": 10.0,
    "num_leaves": 13,
    "num_threads": 4,
    "tree_learner": "serial",
    "objective": "binary",
    "verbosity": -1,
}


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


def train_lightgbm():
    """Тренировка с помощью LightGBM"""
    x, y, test_x = processing.make_features()

    oof_y = pd.Series(0, index=y.index, name="oof_y")
    trees = []

    for trn_idx, val_idx in FOLDS.split(x, y):
        x_train, y_train = x.iloc[trn_idx], y.iloc[trn_idx]
        x_valid, y_valid = x.iloc[val_idx], y.iloc[val_idx]
        x_train, y_train = augment(x_train.values, y_train.values)

        trn_data = lgb.Dataset(x_train, label=y_train)
        val_data = lgb.Dataset(x_valid, label=y_valid)

        clf = lgb.train(
            PARAMS,
            trn_data,
            ITERATIONS,
            valid_sets=[trn_data, val_data],
            verbose_eval=ITERATIONS // 100,
            early_stopping_rounds=ITERATIONS // 10,
        )
        oof_y.iloc[val_idx] = clf.predict(x_valid, num_iteration=clf.best_iteration)
        trees.append(clf.best_iteration)
        print("\n")

    logging.info(f"Количество деревьев: {trees}")
    logging.info(
        f"Среднее количество деревьев: {np.mean(trees):.0f} +/- {np.std(trees):.0f}"
    )

    oof_auc = metrics.roc_auc_score(y, oof_y, "micro")
    logging.info(f"OOF AUC: {oof_auc:0.4f}")

    x, y = augment(x.values, y.values)
    trn_data = lgb.Dataset(x, label=y)
    num_iteration = sorted(trees)[N_SPLITS // 2 + 1]
    clf = lgb.train(PARAMS, trn_data, num_iteration, verbose_eval=ITERATIONS // 100)

    sub = pd.DataFrame(
        clf.predict(test_x, num_iteration), index=test_x.index, columns=["target"]
    )
    sub.to_csv(
        conf.DATA_PROCESSED
        + f"{time.strftime('%Y-%m-%d_%H-%M')}_AUC-{oof_auc:0.4f}_gbm.csv"
    )


if __name__ == "__main__":
    train_lightgbm()
