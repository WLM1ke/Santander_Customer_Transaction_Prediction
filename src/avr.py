"""Простое усреднение."""
import time

import pandas as pd


from src import conf


def avr_sub(file_names: list):
    """Усредняет несколько сабмишенов."""
    df = None
    for file in file_names:
        if df is None:
            df = pd.read_csv(conf.DATA_PROCESSED + file, index_col=0)
            continue
        df += pd.read_csv(conf.DATA_PROCESSED + file, index_col=0)

    return df / len(file_names)


if __name__ == '__main__':
    rez = avr_sub(
        [
            "2019-03-20_18-53_AUC-0.9013_gbm11_best.csv",
        ]
    )
    rez.to_csv(
        conf.DATA_PROCESSED
        + f"{time.strftime('%Y-%m-%d_%H-%M')}_avr.csv")
