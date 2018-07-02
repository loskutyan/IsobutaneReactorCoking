import datetime

import pandas as pd
import numpy as np

import constants
from exceptions import NotReadyModelError


class AnalysisLinearTrendsExtractor:
    TAGS_TO_PROCESS = ["Водород, %", "Азот N2, %", "Окись углерода, %", "Метан, %", "Сумма этан+этилен, %",
                       "Двуокись углерода, %", "Сумма углеводородов С3, %", "Изобутан, %", "н-Бутан, %",
                       "Бутен1+изобутилен, %", "Сумма бутиленов, %", "Бутадиен-1,3, %",
                       "Массовая доля суммы углеводородов С5 и выше, %"]

    def __init__(self, period):
        self._period = period

    @staticmethod
    def calculate_linear_trend(values_series):
        x = values_series.index - values_series.index[0].total_seconds()
        A = np.vstack([x, np.ones(len(x))]).T
        y = values_series.values
        coef, intercept = np.linalg.lstsq(A, y, None)[0]
        return coef, intercept

    def extract_data_with_linear_trends(self, chemical_analysis_data):
        data_to_process = chemical_analysis_data[AnalysisLinearTrendsExtractor.TAGS_TO_PROCESS].dropna(how='all')
        result = pd.DataFrame(data=None, index=data_to_process.index)
        for tag in data_to_process.columns:
            values_series = data_to_process[tag]
            coefs_and_intercepts = []
            index = []
            for i in sorted(values_series.index):
                if values_series.loc[i] is not None and str(values_series.loc[i]) != 'nan':
                    period_values_series = values_series.loc[i - self._period: i]
                    coefs_and_intercepts.append(self.calculate_linear_trend(period_values_series))
                    index.append(i)
            coefs, intercepts = zip(*coefs_and_intercepts)
            tag_result = pd.DataFrame({
                tag: values_series.loc[index].values,
                '{}_coef'.format(tag): coefs,
                '{}_intercept'.format(tag): intercepts
            }, index=index).sort_index(axis=1)
            result = result.merge(tag_result, left_index=True, right_index=True, how='outer')
        return result


class NNTemperaturesFeaturesExtractor:
    FEATURE_NAME_PREFIX = 'Температура_'

    def __init__(self, period, input_time_intervals_number, output_features_number, nn_model=None):
        self._period = period
        self._input_time_intervals_number = input_time_intervals_number
        self._output_features_number = output_features_number
        self._interval = period / input_time_intervals_number
        if self._interval <= constants.ONE_SECOND_DELTA:
            raise ValueError('temperatures interval must be longer than one second')
        self._nn_model = nn_model
        if nn_model is not None:
            self._validate_model_parameters()

    def _validate_model_parameters(self):
        if self._input_time_intervals_number != self._nn_model.input_shape[1]:
            raise ValueError('number of intervals is {} but must be equal\
             to neural network input shape ({})'.format(self._input_time_intervals_number,
                                                        self._nn_model.input_shape[1]))
        if self._output_features_number != self._nn_model.get_layer(index=0).output_shape[1]:
            raise ValueError('number of intervals is {} but must be equal\
             to neural network input shape ({})'.format(self._output_features_number,
                                                        self._nn_model.get_layer(index=0).output_shape[1]))
        return True

    def _calculate_features_row_for_datetime(self, temperature_sensor_data, dt):
        period_sensor_data = temperature_sensor_data.loc[dt - self._period: dt]
        return [period_sensor_data.loc[dt + constants.ONE_SECOND_DELTA - self._interval * (i + 1):
                                       dt - self._interval * i].mean()
                for i in range(self._input_time_intervals_number)]

    def extract_features(self, temperature_sensor_data, timestamps):
        if self._nn_model is None:
            raise NotReadyModelError('temperatures NN-encoded features model wasn\'t loaded or trained')
        nn_input = pd.DataFrame(
            timestamps.map(lambda dt: self._calculate_features_row_for_datetime(temperature_sensor_data,
                                                                                  dt)).tolist(),
            index=timestamps
        )
        nn_input_normalized = ((nn_input - constants.NN_NORMALIZING_EXPECTATION_EVALUATION) \
                    / constants.NN_NORMALIZING_STD_EVALUATION)\
            .fillna(0.0)\
            .values
        nn_output = pd.DataFrame(self._nn_model.predict(nn_input_normalized),
                                 index=timestamps,
                                 columns=[NNTemperaturesFeaturesExtractor.FEATURE_NAME_PREFIX + str(i)
                                          for i in range(self._output_features_number)])
        return nn_output


class FeaturesExtractor:
    ABOVE_TEMP_DELTA_FEATURE_NAME = 'delta_top'
    BELOW_TEMP_DELTA_FEATURE_NAME = 'delta_bot'
    ANGLES_FEATURES_NAME = ['angle{}'.format(i + 1) for i in range(4)]

    def __init__(self, sensor_id, reactor, interval=datetime.timedelta(hours=12)):
        self._sensor_id = sensor_id
        self._plate_name = reactor.find_plate_name(sensor_id)
        if self._plate_name is None:
            raise ValueError('No plate with sensor {} in reactor found'.format(str(sensor_id)))
        self._plate = reactor.get_plate(self._plate_name)
        self._reactor = reactor
        self._interval = interval

    def _collect_interval_mean_temperatures(self, temperature_sensors_data, chemical_analysis_data):
        result_index = chemical_analysis_data.index
        return pd.DataFrame(
            result_index.map(lambda dt: temperature_sensors_data.loc[dt - self._interval: dt].mean().tolist()).tolist(),
            index=result_index
        )

    def _extract_above_temperature_delta_features(self, interval_mean_temperatures):
        above_plate_name = self._reactor.get_plate_name_above(self._plate_name)
        if above_plate_name == self._plate_name:
            return pd.Series(np.zeros(interval_mean_temperatures.shape[0]),
                             index=interval_mean_temperatures.index,
                             name=FeaturesExtractor.ABOVE_TEMP_DELTA_FEATURE_NAME)
        above_plate_sensors = self._reactor.get_plate(above_plate_name).get_sensors_list()
        above_plate_mean_temperatures = interval_mean_temperatures[above_plate_sensors].mean(axis=1)
        return (above_plate_mean_temperatures - interval_mean_temperatures[self._sensor_id]) \
            .rename(FeaturesExtractor.ABOVE_TEMP_DELTA_FEATURE_NAME)

    def _extract_below_temperature_delta_features(self, interval_mean_temperatures):
        below_plate_name = self._reactor.get_plate_name_below(self._plate_name)
        if below_plate_name == self._plate_name:
            return pd.Series(np.zeros(interval_mean_temperatures.shape[0]),
                             index=interval_mean_temperatures.index,
                             name=FeaturesExtractor.BELOW_TEMP_DELTA_FEATURE_NAME)
        below_plate_sensors = self._reactor.get_plate(below_plate_name).get_sensors_list()
        below_plate_mean_temperatures = interval_mean_temperatures[below_plate_sensors].mean(axis=1)
        return (below_plate_mean_temperatures - interval_mean_temperatures[self._sensor_id]) \
            .rename(FeaturesExtractor.BELOW_TEMP_DELTA_FEATURE_NAME)

    def _extract_angle_features(self, index):
        return pd.DataFrame([self._plate.get_angle_array(self._sensor_id)] * len(index),
                            index=index, columns=FeaturesExtractor.ANGLES_FEATURES_NAME)

    def extract_features(self, temperature_sensors_data, chemical_analysis_data):
        interval_mean_temperatures = self._collect_interval_mean_temperatures(temperature_sensors_data,
                                                                              chemical_analysis_data)
        return pd.DataFrame()
