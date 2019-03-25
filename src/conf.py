"""Основные параметры."""
import logging

import numpy as np
from sklearn import model_selection

# Конфигурация логгера
logging.basicConfig(level=logging.INFO)

# Пути к данным
DATA_TRAIN = "../raw/train.csv"
DATA_TEST = "../raw/test.csv"
DATA_SUB = "../raw/sample_submission.csv"
DATA_PROCESSED = "../processed/"

# Общие настройки
SEED = 284_702
np.random.seed(SEED)
N_SPLITS = 11
FOLDS = model_selection.StratifiedKFold(
    n_splits=N_SPLITS, shuffle=True, random_state=SEED
)
