"""Обучение моделей."""
import logging
import time

from catboost import EFstrType
from sklearn import metrics
from sklearn import model_selection
import catboost
import pandas as pd
import numpy as np
import boruta
import lightgbm

from src import conf
from src import processing

SPEED = 10

# Настройки валидации
SEED = 284702
N_SPLITS = 5
DEPTH = 6
ITERATIONS = 30000 // SPEED
LEARNING_RATE = 0.01 * SPEED
FOLDS = model_selection.StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
DROP = [
]

CLF_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    random_state=SEED,
    depth=6,
    od_type="Iter",
    od_wait=ITERATIONS // 10,
    verbose=ITERATIONS // 100,
    learning_rate=LEARNING_RATE,
    iterations=ITERATIONS,
    allow_writing_files=False
)


def train_catboost():
    """Обучение catboost."""
    x, y, test_x = processing.make_features()
    x = x.drop(DROP, axis=1)

    oof_y = pd.Series(0, index=y.index, name="oof_y")
    trees = []
    scores = []

    for train_index, valid_index in FOLDS.split(x, y):

        pool_train = catboost.Pool(
            data=x.iloc[train_index],
            label=y.iloc[train_index],
            cat_features=None,
            weight=None

        )

        pool_valid = catboost.Pool(
            data=x.iloc[valid_index],
            label=y.iloc[valid_index],
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
        oof_y.iloc[valid_index] = clf.predict_proba(pool_valid)[:, 1]

    logging.info(f"Количество деревьев: {trees}")
    logging.info(f"Среднее количество деревьев: {np.mean(trees):.0f} +/- {np.std(trees):.0f}")
    logging.info(f"AUC на кроссвалидации: " + str(np.round(scores, 5)))
    logging.info(f"AUC среднее: {np.mean(scores):0.4f} +/- {np.std(scores):0.4f}")

    oof_auc = metrics.roc_auc_score(y, oof_y, "micro")
    logging.info(f"OOF AUC: {oof_auc:0.4f}")

    pd.concat([y, oof_y], axis=1).to_pickle(
        conf.DATA_PROCESSED + "oof.pickle"
    )

    CLF_PARAMS["iterations"] = sorted(trees)[N_SPLITS // 2 + 1]
    clf = catboost.CatBoostClassifier(**CLF_PARAMS)
    pool_full = catboost.Pool(
        data=x,
        label=y,
        cat_features=None,
        weight=None

    )
    fit_params = dict(
        X=pool_full,
        cat_features=None
    )
    clf.fit(**fit_params)

    sub = pd.DataFrame(clf.predict_proba(test_x)[:, 1], index=test_x.index, columns=["target"])
    sub.to_csv(conf.DATA_PROCESSED + f"{time.strftime('%Y-%m-%d_%H-%M')}_AUC-{oof_auc:0.4f}.csv")

    logging.info("Важность признаков:")
    for i, v in clf.get_feature_importance(pool_full, prettified=True):
        logging.info(i.ljust(20) + str(v))

    pd.DataFrame(clf.get_feature_importance(pool_full, prettified=True)).set_index(0).to_pickle(
        conf.DATA_PROCESSED + "importance.pickle"
    )

    logging.info("Попарная важность признаков:")
    for i, j, value in clf.get_feature_importance(pool_full, type=EFstrType.Interaction, prettified=True)[:20]:
        logging.info(x.columns[i].ljust(20) + x.columns[j].ljust(20) + str(value))


def feat_sel():
    """Выбор признаков."""
    x, y = processing.read_train()
    x = x.drop(DROP, axis=1)
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
