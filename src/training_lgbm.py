"""Обучение моделей c помощью LightGBM."""
import logging
import time

import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn import metrics

from src import conf
from src import processing
from src.conf import FOLDS

ITERATIONS = 40000
CLF_PARAMS = {
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


def train_lightgbm():
    """Тренировка с помощью LightGBM"""
    x, y, test_x = processing.make_features()

    trees = []
    scores = []
    oof_y = pd.Series(0, index=y.index, name="oof_y")
    sub = pd.DataFrame(
        0, index=test_x.index, columns=["target"]
    )

    for train_index, valid_index in FOLDS.split(x, y):
        x_train, y_train = x.iloc[train_index], y.iloc[train_index]
        x_valid, y_valid = x.iloc[valid_index], y.iloc[valid_index]

        x_train, y_train = processing.augment(x_train.values, y_train.values)

        data_train = lgb.Dataset(x_train, label=y_train)
        data_test = lgb.Dataset(x_valid, label=y_valid)

        clf = lgb.train(
            CLF_PARAMS,
            data_train,
            ITERATIONS,
            valid_sets=[data_train, data_test],
            verbose_eval=ITERATIONS // 100,
            early_stopping_rounds=ITERATIONS // 10,
        )

        trees.append(clf.best_iteration)
        scores.append(clf.best_score["valid_1"]["auc"])

        oof_y.iloc[valid_index] = clf.predict(x_valid, num_iteration=clf.best_iteration)
        sub["target"] = sub["target"] + clf.predict(test_x, num_iteration=clf.best_iteration) / FOLDS.get_n_splits()
        print("\n")

    logging.info(f"Количество деревьев: {trees}")
    logging.info(f"Среднее количество деревьев: {np.mean(trees):.0f} +/- {np.std(trees):.0f}")
    logging.info(f"AUC на кроссвалидации: {str(np.round(scores, 5))}")
    logging.info(f"AUC среднее: {np.mean(scores):0.4f} +/- {np.std(scores):0.4f}")

    oof_auc = metrics.roc_auc_score(y, oof_y, "micro")
    logging.info(f"OOF AUC: {oof_auc:0.4f}")

    sub.to_csv(
        conf.DATA_PROCESSED
        + f"{time.strftime('%Y-%m-%d_%H-%M')}_AUC-{oof_auc:0.4f}_gbm{FOLDS.get_n_splits()}.csv"
    )


if __name__ == "__main__":
    train_lightgbm()
