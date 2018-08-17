from collections import defaultdict

import pandas as pd

import constants
from datasource.source import SQLSource


class InputDataHandler:
    def __init__(self, settings):
        source_params = settings.get_input()
        self._source = SQLSource(source_params, constants.INPUT_DATETIME_COLUMN)
        self._table_names = settings.get_input_tables()
        self._temperatures_data = None
        self._last_temperatures_datetime = None
        self._analysis_data = None
        self._last_analysis_datetime = None

    def get_temperatures(self, since_datetime=None):
        return self._source.get_data_since(self._table_names['temperatures'], since_datetime)

    def get_analysis(self, since_datetime=None):
        analysis_data_list = []
        for table_type, table_name in self._table_names.items():
            if table_type == 'temperatures':
                continue
            analysis_data_list.append(self._source.get_data_since(self._table_names[table_type], since_datetime, False))
        return pd.concat(analysis_data_list, axis=1, sort=True, join='outer')


class OutputDataHandler:
    def __init__(self, settings):
        source_params = settings.get_output()
        self._table_names = settings.get_output_tables()
        self._source = SQLSource(source_params, constants.OUTPUT_DATETIME_COLUMN)

    def find_last_prediction_datetime(self):
        return self._source.find_last_datetime(self._table_names['predictions'])

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
        plates_numbers = sorted(plates_columns.keys())
        if len(plates_numbers) < 2:
            return pd.DataFrame()
        rows_number = raw_temperatures.shape[0]
        diffs = []
        for i in range(len(plates_numbers) - 1):
            plate_below_num, plate_above_num = plates_numbers[i], plates_numbers[i + 1]
            plate_below_mean = raw_temperatures[plates_columns[plate_below_num]].mean(axis=1)
            plate_above_mean = raw_temperatures[plates_columns[plate_above_num]].mean(axis=1)
            diffs.append(pd.DataFrame({'Решетки': ['{} - {}'.format(plate_above_num, plate_below_num)] * rows_number,
                                       'Разность температур': plate_above_mean - plate_below_mean},
                                      index=raw_temperatures.index))
        return pd.concat(diffs, sort=False)

    @staticmethod
    def _build_temperatures_std(raw_temperatures):
        plates_columns = defaultdict(list)
        for col in raw_temperatures.columns:
            plates_columns[int(col.split(':')[0])].append(col)
        rows_number = raw_temperatures.shape[0]
        stds = []
        for plate_num, plate_columns in plates_columns.items():
            plate_std = raw_temperatures[plate_columns].std(axis=1)
            stds.append(pd.DataFrame({'Решетка': [int(plate_num)] * rows_number,
                                      'Стандартное отклонение': plate_std},
                                     index=raw_temperatures.index))
        return pd.concat(stds, sort=False)

    def update_predictions_and_statistics(self, predictions, temperatures):
        last_prediction_datetime = self.find_last_prediction_datetime()

        filtered_predictions = predictions.loc[predictions.index > last_prediction_datetime]
        self._source.write_new_data(self._table_names['predictions'],
                                    OutputDataHandler._format_predictions(filtered_predictions))

        last_new_prediction_datetime = filtered_predictions.index.max()
        filtered_temperatures = temperatures.loc[(temperatures.index > last_prediction_datetime)
                                                 & (temperatures.index <= last_new_prediction_datetime)]
        formatted_temperatures = OutputDataHandler._format_temperatures(filtered_temperatures)
        self._source.write_new_data(self._table_names['temperatures'], formatted_temperatures)

        self._source.write_new_data(self._table_names['temperatures_diff'],
                                    OutputDataHandler._build_temperatures_diff(filtered_temperatures))

        self._source.write_new_data(self._table_names['temperatures_std'],
                                    OutputDataHandler._build_temperatures_std(filtered_temperatures))
        return
