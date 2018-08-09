import pandas as pd

from io.source import SQLSource


class InputDataHandler:
    TABLE_NAMES = {
        'catalyst_analysis': 'cat',
        'out_gas_analysis': 'out',
        'smoke_gas_analysis': 'smoke',
        'temperatures': 'temps'
    }

    DATETIME_COL = 'Дата'

    def __init__(self, settings):
        source_params = settings.get_input()
        self._source = SQLSource(source_params, InputDataHandler.DATETIME_COL)
        self._temperatures_data = None
        self._last_temperatures_datetime = None
        self._analysis_data = None
        self._last_analysis_datetime = None

    def get_temperatures(self, since_datetime=None):
        return self._source.get_data_since(InputDataHandler.TABLE_NAMES['temperatures'], since_datetime)

    def get_analysis(self, since_datetime=None):
        analysis_data_list = []
        for table_type, table_name in InputDataHandler.TABLE_NAMES.items():
            if table_type == 'temperatures':
                continue
            analysis_data_list.append(self._source.get_data_since(InputDataHandler.TABLE_NAMES[table_type],
                                                                  since_datetime, False))
        return pd.concat(analysis_data_list, axis=1, sort=True, join='outer')


class OutputDataHandler:
    TABLE_NAMES = {
        'predictions': 'preds',
        'statistics': 'stats'
    }

    DATETIME_COL = 'Дата'

    def __init__(self, settings):
        source_params = settings.get_output()
        self._source = SQLSource(source_params, OutputDataHandler.DATETIME_COL)
        self._last_prediction_datetime = None

    def find_last_datetime(self):
        if self._last_prediction_datetime is None:
            self._last_prediction_datetime = self._source.find_last_datetime(
                OutputDataHandler.TABLE_NAMES['predictions'])
        return self._last_prediction_datetime

    def update_predictions(self, predictions):
        if self._last_prediction_datetime is None:
            self.find_last_datetime()
        filtered_predictions = predictions.loc[predictions.index > self._last_prediction_datetime]
        self._source.write_new_data(OutputDataHandler.TABLE_NAMES['predictions'], filtered_predictions)
        return
