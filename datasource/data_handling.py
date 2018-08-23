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
            analysis_data_list.append(self._source.get_data_since(self._table_names[table_type], since_datetime))
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
    def _smooth_statistics(data):
        return data.rolling(constants.STATISTICS_SMOOTHING_PERIOD).mean()

    @staticmethod
    def _filter_statistics(data):
        return data[data.index.map(lambda dt: dt.minute % constants.STATISTICS_INDEX_FILTERING_MINUTES == 0)]

    @staticmethod
    def _format_temperatures(temperatures):
        smoothed_filtered = OutputDataHandler._filter_statistics(OutputDataHandler._smooth_statistics(temperatures))
        rows_number = smoothed_filtered.shape[0]
        formatted_temperatures = []
        for col in smoothed_filtered.columns:
            plate_num, sensor_num = col.split(':')
            formatted_temperatures.append(pd.DataFrame({'Температура': smoothed_filtered[col],
                                                        'Решетка': [int(plate_num)] * rows_number,
                                                        'Датчик': [int(sensor_num)] * rows_number},
                                                       index=smoothed_filtered.index))
        return pd.concat(formatted_temperatures, sort=False)

    @staticmethod
    def _build_temperatures_diff(raw_temperatures):
        plates_columns = defaultdict(list)
        for col in raw_temperatures.columns:
            plates_columns[int(col.split(':')[0])].append(col)
        plates_numbers = sorted(plates_columns.keys())
        if len(plates_numbers) < 2:
            return pd.DataFrame()
        diffs = []
        for i in range(len(plates_numbers) - 1):
            plate_below_num, plate_above_num = plates_numbers[i], plates_numbers[i + 1]
            plate_below_mean = raw_temperatures[plates_columns[plate_below_num]].mean(axis=1)
            plate_above_mean = raw_temperatures[plates_columns[plate_above_num]].mean(axis=1)
            diff = plate_above_mean - plate_below_mean
            smoothed_filtered_diff = OutputDataHandler._filter_statistics(OutputDataHandler._smooth_statistics(diff))
            rows_number = smoothed_filtered_diff.shape[0]
            diffs.append(pd.DataFrame({'Решетки': ['{} - {}'.format(plate_above_num, plate_below_num)] * rows_number,
                                       'Разность температур': smoothed_filtered_diff},
                                      index=smoothed_filtered_diff.index))
        return pd.concat(diffs, sort=False)

    @staticmethod
    def _build_temperatures_std(raw_temperatures):
        raw_stds = raw_temperatures.rolling(constants.TEMPERATURES_STD_PERIOD).std()
        filtered_stds = OutputDataHandler._filter_statistics(raw_stds)
        rows_number = filtered_stds.shape[0]
        stds = []
        for col in filtered_stds.columns:
            plate_num, sensor_num = col.split(':')
            stds.append(pd.DataFrame({'Стандартное отклонение': filtered_stds[col],
                                      'Решетка': [int(plate_num)] * rows_number,
                                      'Датчик': [int(sensor_num)] * rows_number},
                                     index=filtered_stds.index))
        return pd.concat(stds, sort=False)

    @staticmethod
    def _build_temperatures_plates_std(raw_temperatures):
        plates_columns = defaultdict(list)
        for col in raw_temperatures.columns:
            plates_columns[int(col.split(':')[0])].append(col)
        stds = []
        for plate_num, plate_columns in plates_columns.items():
            plate_std = raw_temperatures[plate_columns].std(axis=1)
            smoothed_std = OutputDataHandler._smooth_statistics(plate_std)
            smoothed_filtered_std = OutputDataHandler._filter_statistics(smoothed_std)
            rows_number = smoothed_filtered_std.shape[0]
            stds.append(pd.DataFrame({'Решетка': [int(plate_num)] * rows_number,
                                      'Стандартное отклонение': smoothed_filtered_std},
                                     index=smoothed_filtered_std.index))
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

        self._source.write_new_data(self._table_names['plates_temperatures_std'],
                                    OutputDataHandler._build_temperatures_plates_std(filtered_temperatures))

        min_temperatures_datetime_for_std = last_prediction_datetime
        if last_prediction_datetime != constants.MIN_DATETIME:
            min_temperatures_datetime_for_std = last_prediction_datetime - constants.TEMPERATURES_STD_PERIOD
        temperatures_filtered_for_std = temperatures.loc[(temperatures.index > min_temperatures_datetime_for_std)
                                                         & (temperatures.index <= last_new_prediction_datetime)]
        temperatures_std = OutputDataHandler._build_temperatures_std(temperatures_filtered_for_std)
        filtered_temperatures_std = temperatures_std.loc[temperatures_std.index > last_prediction_datetime].dropna()
        self._source.write_new_data(self._table_names['temperatures_std'], filtered_temperatures_std)
        return
