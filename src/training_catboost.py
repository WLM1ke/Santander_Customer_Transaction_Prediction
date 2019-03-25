"""Обучение моделей c помощью catboost."""
import logging
import time

from catboost import EFstrType
from sklearn import metrics
import catboost
import pandas as pd
import numpy as np
import boruta
import lightgbm

from src import conf
from src import processing
from src.conf import FOLDS
from src.conf import SEED

ITERATIONS = 50000
CLF_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    random_state=SEED,
    depth=3,
    od_type="Iter",
    od_wait=ITERATIONS // 10,
    verbose=ITERATIONS // 100,
    learning_rate=0.02,
    iterations=ITERATIONS,
    allow_writing_files=False,

)


def train_catboost():
    """Обучение catboost."""
    x, y, test_x = processing.make_features()

    trees = []
    scores = []
    oof_y = pd.Series(0, index=y.index, name="oof_y")
    sub = pd.DataFrame(
        0, index=test_x.index, columns=["target"]
    )

    for index_train, index_valid in FOLDS.split(x, y):
        x_train, y_train = x.iloc[index_train], y.iloc[index_train]
        x_valid, y_valid = x.iloc[index_valid], y.iloc[index_valid]

        x_train, y_train = processing.augment(x_train.values, y_train.values)

        pool_train = catboost.Pool(
            data=x_train,
            label=y_train,
            cat_features=None,
            weight=None

        )
        pool_valid = catboost.Pool(
            data=x_valid,
            label=y_valid,
            cat_features=None,
            weight=None
        )

        clf = catboost.CatBoostClassifier(**CLF_PARAMS)
        fit_params = dict(
            X=pool_train,
            eval_set=[pool_valid],
        )
        clf.fit(**fit_params)

        trees.append(clf.tree_count_)
        scores.append(clf.best_score_['validation_0']['AUC'])

        oof_y.iloc[index_valid] = clf.predict_proba(pool_valid)[:, 1]
        sub["target"] = sub["target"] + clf.predict_proba(test_x)[:, 1] / FOLDS.get_n_splits()

    logging.info(f"Количество деревьев: {trees}")
    logging.info(f"Среднее количество деревьев: {np.mean(trees):.0f} +/- {np.std(trees):.0f}")
    logging.info(f"AUC на кроссвалидации: {str(np.round(scores, 5))}")
    logging.info(f"AUC среднее: {np.mean(scores):0.4f} +/- {np.std(scores):0.4f}")

    oof_auc = metrics.roc_auc_score(y, oof_y, "micro")
    logging.info(f"OOF AUC: {oof_auc:0.4f}")

    sub.to_csv(
        conf.DATA_PROCESSED
        + f"{time.strftime('%Y-%m-%d_%H-%M')}_AUC-{oof_auc:0.4f}_cat{FOLDS.get_n_splits()}.csv")

    """
    logging.info("Важность признаков:")
    for i, v in clf.get_feature_importance(pool_full, prettified=True):
        logging.info(i.ljust(20) + str(v))

    pd.DataFrame(clf.get_feature_importance(pool_full, prettified=True)).set_index(0).to_pickle(
        conf.DATA_PROCESSED + "importance.pickle"
    )

    logging.info("Попарная важность признаков:")
    for i, j, value in clf.get_feature_importance(pool_full, type=EFstrType.Interaction, prettified=True)[:20]:
        logging.info(x.columns[i].ljust(20) + x.columns[j].ljust(20) + str(value))
    """


def feat_sel():
    """Выбор признаков."""
    x, y = processing.read_train()
    clf = lightgbm.LGBMClassifier(boosting_type="rf",
                                  bagging_freq=1,
                                  bagging_fraction=0.632,
                                  feature_fraction=0.632)
    feat_selector = boruta.BorutaPy(clf, n_estimators=ITERATIONS, verbose=2)
    feat_selector.fit(x.values, y.values)
    print(x.columns[feat_selector.support_weak_])
    print(x.columns[feat_selector.support_])
    print(pd.Series(feat_selector.ranking_, index=x.columns).sort_values())


if __name__ == '__main__':
    train_catboost()
    # feat_sel()
