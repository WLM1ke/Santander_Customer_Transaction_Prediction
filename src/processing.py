"""Процессинг исходных данных."""
import pandas as pd
from sklearn import preprocessing

from src import conf

DTYPES = dict(
    target="int8",
    **{f"var_{i}": "float32" for i in range(200)}
)


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
    base_col = x_all.columns

    x_all.iloc[:, :] = preprocessing.normalize(x_all, axis=0)

    x_all["mean"] = x_all[base_col].mean(axis=1)
    x_all["std"] = x_all[base_col].std(axis=1)
    x_all["dist"] = (x_all[base_col] ** 2).mean(axis=1)
    x_all["median"] = x_all[base_col].median(axis=1)
    x_all["q10"] = x_all[base_col].quantile(0.1, axis=1)
    x_all["q20"] = x_all[base_col].quantile(0.2, axis=1)
    x_all["q30"] = x_all[base_col].quantile(0.3, axis=1)
    x_all["q40"] = x_all[base_col].quantile(0.4, axis=1)
    x_all["q60"] = x_all[base_col].quantile(0.6, axis=1)
    x_all["q70"] = x_all[base_col].quantile(0.7, axis=1)
    x_all["q80"] = x_all[base_col].quantile(0.8, axis=1)
    x_all["q90"] = x_all[base_col].quantile(0.9, axis=1)

    return x_all.iloc[:len(y_train)], y_train, x_all.iloc[len(y_train):]


if __name__ == '__main__':
    x = read_test()
    print(x)
    print(x.describe())
