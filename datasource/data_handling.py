from collections import defaultdict

import pandas as pd

import constants
from datasource.source import SQLSource


class InputDataHandler:
    TABLE_NAMES = {
        'catalyst_analysis': 'cat',
        'out_gas_analysis': 'out_gas',
        'smoke_gas_analysis': 'smoke',
        'temperatures': 'temps'
    }

    def __init__(self, settings):
        source_params = settings.get_input()
        self._source = SQLSource(source_params, constants.INPUT_DATETIME_COLUMN)
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
        'predictions': 'predictions',
        'temperatures': 'temperatures',
        'temperatures_diff': 'temps_diff',
        'temperatures_std': 'temps_std'
    }

    def __init__(self, settings):
        source_params = settings.get_output()
        self._source = SQLSource(source_params, constants.OUTPUT_DATETIME_COLUMN)
        self._last_prediction_datetime = self.find_last_datetime()

    def find_last_datetime(self):
        return self._source.find_last_datetime(OutputDataHandler.TABLE_NAMES['predictions'])

    @staticmethod
    def _format_predictions(predictions):
        rows_number = predictions.shape[0]
        formatted_predictions = []
        for col in predictions.columns:
            plate_num, sensor_num, horizon = col.split(':')
            formatted_predictions.append(pd.DataFrame({'Горизонт прогнозирования': [horizon] * rows_number,
                                                       'Решетка': [int(plate_num)] * rows_number,
                                                       'Датчик': [int(sensor_num)] * rows_number,
                                                       'Вероятность коксования': predictions[col]},
                                                      index=predictions.index))
        return pd.concat(formatted_predictions, sort=False)

    @staticmethod
    def _format_temperatures(temperatures):
        rows_number = temperatures.shape[0]
        formatted_temperatures = []
        for col in temperatures.columns:
            plate_num, sensor_num = col.split(':')
            formatted_temperatures.append(pd.DataFrame({'Температура': temperatures[col],
                                                        'Решетка': [int(plate_num)] * rows_number,
                                                        'Датчик': [int(sensor_num)] * rows_number},
                                                       index=temperatures.index))
        return pd.concat(formatted_temperatures, sort=False)

    @staticmethod
    def _build_temperatures_diff(raw_temperatures):
        plates_columns = defaultdict(list)
        for col in raw_temperatures.columns:
            plates_columns[int(col.split(':')[0])].append(col)
        plates = sorted(plates_columns.keys())
        if len(plates) < 2:
            return pd.DataFrame()
        rows_number = raw_temperatures.shape[0]
        diffs = []
        for i in range(len(plates) - 1):
            plate_below, plate_above = plates[i], plates[i + 1]
            plate_below_mean = raw_temperatures[plates_columns[plate_below]].mean(axis=1)
            plate_above_mean = raw_temperatures[plates_columns[plate_above]].mean(axis=1)
            diffs.append(pd.DataFrame({'Решетки': ['{} - {}'.format(plate_above, plate_below)] * rows_number,
                                       'Разность температур': plate_above_mean - plate_below_mean},
                                      index=raw_temperatures.index))
        return pd.concat(diffs, sort=False)

    def update_predictions_and_statistics(self, predictions, temperatures):
        self._last_prediction_datetime = self.find_last_datetime()

        filtered_predictions = predictions.loc[predictions.index > self._last_prediction_datetime]
        self._source.write_new_data(OutputDataHandler.TABLE_NAMES['predictions'],
                                    OutputDataHandler._format_predictions(filtered_predictions))

        last_new_prediction_datetime = filtered_predictions.index.max()
        filtered_temperatures = temperatures.loc[(temperatures.index > self._last_prediction_datetime)
                                                 & (temperatures.index <= last_new_prediction_datetime)]
        formatted_temperatures = OutputDataHandler._format_temperatures(filtered_temperatures)
        self._source.write_new_data(OutputDataHandler.TABLE_NAMES['temperatures'], formatted_temperatures)

        self._source.write_new_data(OutputDataHandler.TABLE_NAMES['temperatures_diff'],
                                    OutputDataHandler._build_temperatures_diff(filtered_temperatures))

        return
