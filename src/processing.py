"""Процессинг исходных данных."""
import pandas as pd
from src import conf


def read_train():
    """Загружает обучающие данные."""
    dtypes = dict(
        target="int8",
        **{f"var_{i}": "float32" for i in range(200)}
    )
    df = pd.read_csv(conf.DATA_TRAIN, index_col=0, dtype=dtypes)
    return df.iloc[:, 1:], df.iloc[:, 0]


def read_test():
    """Загружает тестовые данные."""
    dtypes = dict(
        **{f"var_{i}": "float32" for i in range(200)}
    )
    df = pd.read_csv(conf.DATA_TEST, index_col=0, dtype=dtypes)
    return df


if __name__ == '__main__':
    x, y = read_train()
    print(x)
    print(y.describe())
