import warnings

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential

import constants
import exceptions

ABOVE_PLATE_TEMPERATURE_DELTA_NAME = 'above_plate_temperature_delta'
BELOW_PLATE_TEMPERATURE_DELTA_NAME = 'below_plate_temperature_delta'


def calculate_duration(timestamps):
    return pd.DataFrame({'duration': (timestamps - timestamps[0]).total_seconds().values / 3600.},
                        index=timestamps)


def collect_interval_mean_temperatures(temperature_sensors_data, interval, timestamps):
    return pd.DataFrame(
        timestamps.map(lambda dt: temperature_sensors_data.loc[dt - interval: dt].mean().tolist()).tolist(),
        index=timestamps,
        columns=temperature_sensors_data.columns
    )


def calculate_above_plate_temperature_delta(interval_mean_temperatures, sensor_id, sensor_plate_number, reactor):
    above_plate_number = reactor.get_plate_number_above(sensor_plate_number)
    if above_plate_number is None:
        return pd.Series(np.zeros(interval_mean_temperatures.shape[0]),
                         index=interval_mean_temperatures.index,
                         name=ABOVE_PLATE_TEMPERATURE_DELTA_NAME)
    above_plate_sensors = reactor.get_plate(above_plate_number).get_sensor_list()
    above_plate_mean_temperatures = interval_mean_temperatures[above_plate_sensors].mean(axis=1)
    return (above_plate_mean_temperatures - interval_mean_temperatures[sensor_id]) \
        .rename(ABOVE_PLATE_TEMPERATURE_DELTA_NAME)


def calculate_below_plate_temperature_delta(interval_mean_temperatures, sensor_id, sensor_plate_number, reactor):
    below_plate_number = reactor.get_plate_number_below(sensor_plate_number)
    if below_plate_number is None:
        return pd.Series(np.zeros(interval_mean_temperatures.shape[0]),
                         index=interval_mean_temperatures.index,
                         name=BELOW_PLATE_TEMPERATURE_DELTA_NAME)
    below_plate_sensors = reactor.get_plate(below_plate_number).get_sensor_list()
    below_plate_mean_temperatures = interval_mean_temperatures[below_plate_sensors].mean(axis=1)
    return (below_plate_mean_temperatures - interval_mean_temperatures[sensor_id]) \
        .rename(BELOW_PLATE_TEMPERATURE_DELTA_NAME)


def extract_sensor_position_features(timestamps, sensor_id, plate):
    return pd.DataFrame([plate.get_angle_array(sensor_id)] * len(timestamps), index=timestamps,
                        columns=['angle_{}'.format(str(x)) for x in range(plate.get_positions_number())])


def calculate_linear_trend(values_series):
    x = (values_series.index - values_series.index[0]).total_seconds() / 3600
    x_matrix = np.vstack([x, np.ones(len(x))]).T
    y = values_series.values
    coef, intercept = np.linalg.lstsq(x_matrix, y, None)[0]
    return coef, intercept


class AnalysisLinearTrendsExtractor:
    def __init__(self, period, tags_to_process):
        self._period = period
        self._tags_to_process = tags_to_process

    def extract(self, chemical_analysis_data):
        missing_tags = [tag for tag in self._tags_to_process if tag not in chemical_analysis_data.columns]
        if len(missing_tags) > 0:
            raise exceptions.MissingTags('tags: {} are missing in chemical analysis')

        data_to_process = chemical_analysis_data[self._tags_to_process].dropna(how='all')
        result = pd.DataFrame(data=None, index=data_to_process.index)
        for tag in data_to_process.columns:
            values_series = data_to_process[tag]
            coefs_and_intercepts = []
            index = []
            for i in sorted(values_series.index):
                if values_series.loc[i] is not None and str(values_series.loc[i]) != 'nan':
                    period_values_series = values_series.loc[i - self._period: i]
                    coefs_and_intercepts.append(calculate_linear_trend(period_values_series))
                    index.append(i)
            coefs, intercepts = zip(*coefs_and_intercepts)
            tag_result = pd.DataFrame({
                '{}_coef'.format(tag): coefs,
                '{}_intercept'.format(tag): intercepts
            }, index=index).sort_index(axis=1)
            result = result.merge(tag_result, left_index=True, right_index=True, how='outer')
        return result


class NNTemperaturesFeaturesExtractor:
    def __init__(self, period, input_time_intervals_number, output_features_number, nn_weights):
        self._period = period
        self._input_time_intervals_number = input_time_intervals_number
        self._output_features_number = output_features_number
        self._interval = period / input_time_intervals_number
        if self._interval <= constants.ONE_SECOND_DELTA:
            raise ValueError('temperatures interval must be longer than one second')
        self._nn_weights = nn_weights
        self._validate_weights()
        self._nn_model = None

    def _validate_weights(self):
        if self._input_time_intervals_number != self._nn_weights.input_shape[1]:
            raise ValueError('number of intervals is {} but must be equal\
             to neural network input shape ({})'.format(self._input_time_intervals_number,
                                                        self._nn_weights.input_shape[1]))
        if self._output_features_number != self._nn_weights.get_layer(index=0).output_shape[1]:
            raise ValueError('number of intervals is {} but must be equal\
             to neural network input shape ({})'.format(self._output_features_number,
                                                        self._nn_weights.get_layer(index=0).output_shape[1]))
        return True

    def _compile_nn_model(self):
        nn_model = Sequential()
        nn_model.add(Dense(self._output_features_number, input_shape=(self._input_time_intervals_number,),
                           activation='linear', use_bias=True))
        nn_model.layers[0].set_weights(self._nn_weights.layers[0].get_weights())
        nn_model.compile(loss='mse', optimizer='sgd')
        self._nn_model = nn_model

    def _calculate_features_row_for_datetime(self, temperature_sensor_data, dt):
        period_sensor_data = temperature_sensor_data.loc[dt - self._period: dt]
        return [period_sensor_data.loc[dt + constants.ONE_SECOND_DELTA - self._interval * (i + 1):
                                       dt - self._interval * i].mean()
                for i in range(self._input_time_intervals_number)]

    def extract(self, temperature_sensor_data, timestamps):
        if self._nn_model is None:
            self._compile_nn_model()
        nn_input = pd.DataFrame(
            timestamps.map(lambda dt: self._calculate_features_row_for_datetime(temperature_sensor_data,
                                                                                dt)).tolist(),
            index=timestamps
        )
        nn_input_normalized = ((nn_input - constants.NN_NORMALIZING_EXPECTATION_EVALUATION)
                               / constants.NN_NORMALIZING_STD_EVALUATION) \
            .fillna(0.0) \
            .values
        nn_output = pd.DataFrame(self._nn_model.predict(nn_input_normalized),
                                 index=timestamps,
                                 columns=['nn_temperature_{}'.format(str(i))
                                          for i in range(self._output_features_number)])
        return nn_output


class FeaturesExtractor:
    def __init__(self, temperatures_features_extractor=None, analysis_features_extractor=None):
        self._temperatures_features_extractor = temperatures_features_extractor
        self._analysis_features_extractor = analysis_features_extractor

    def extract(self, temperature_sensors_data, chemical_analysis_data, sensor_id, reactor,
                mean_temperatures_interval=constants.TWELVE_HOURS_DELTA):
        plate_number = reactor.find_plate_number(sensor_id)
        plate = reactor.get_plate(plate_number)

        timestamps = chemical_analysis_data.index
        interval_mean_temperatures = collect_interval_mean_temperatures(temperature_sensors_data,
                                                                        mean_temperatures_interval, timestamps)
        above_temperature_delta = calculate_above_plate_temperature_delta(interval_mean_temperatures,
                                                                          sensor_id, plate_number, reactor)
        below_temperature_delta = calculate_below_plate_temperature_delta(interval_mean_temperatures,
                                                                          sensor_id, plate_number, reactor)
        position = extract_sensor_position_features(timestamps, sensor_id, plate)
        duration = calculate_duration(timestamps)

        if self._temperatures_features_extractor is None:
            temperatures_features = pd.DataFrame(None, index=timestamps)
            warnings.warn('no custom temperatures features extraction realized', exceptions.MissingComponentsWarning)
        else:
            temperatures_features = self._temperatures_features_extractor.extract(
                temperature_sensors_data[sensor_id],
                timestamps
            )
        if self._analysis_features_extractor is None:
            analysis_features = pd.DataFrame(None, index=timestamps)
            warnings.warn('no custom chemical analysis features extraction realized',
                          exceptions.MissingComponentsWarning)
        else:
            analysis_features = self._analysis_features_extractor.extract(chemical_analysis_data.loc[timestamps])
        return pd.concat([
            chemical_analysis_data.loc[timestamps],
            analysis_features,
            temperatures_features,
            duration,
            above_temperature_delta,
            below_temperature_delta,
            position
        ], axis=1)
