import datetime

import pandas as pd

ONE_SECOND_DELTA = datetime.timedelta(seconds=1)
TWELVE_HOURS_DELTA = datetime.timedelta(hours=12)
MIN_DATETIME = pd.Timestamp.min.to_pydatetime()

TEMPERATURES_HISTORY = datetime.timedelta(days=4)

NN_NORMALIZING_EXPECTATION_EVALUATION = 500.
NN_NORMALIZING_STD_EVALUATION = 100.
NN_PERIOD = datetime.timedelta(days=2)
NN_INPUT_TIME_INTERVALS_NUMBER = 20
NN_OUTPUT_FEATURES_NUMBER = 5

INPUT_DATETIME_COLUMN = 'Timestamp'
MODEL_DATETIME_COLUMN = 'Timestamp'
OUTPUT_DATETIME_COLUMN = 'Дата'
