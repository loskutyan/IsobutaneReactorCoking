import os
import pickle

from keras.models import load_model

import constants
import exceptions
from features.features_extraction import NNTemperaturesFeaturesExtractor


class ModelLoader:
    def __init__(self, settings):
        self._keras_weights_dir = settings.get_sensor_keras_model()['dir']
        self._features_models_dir = settings.get_features_models()['dir']
        self._prediction_models_dir = settings.get_prediction_models()['dir']

    class _SpecificModelLoader:
        def __init__(self, path, is_keras=False):
            reactor_names = os.listdir(path)
            self._models = {}
            for reactor_name in reactor_names:
                reactor_path = os.path.join(path, reactor_name)
                saved_models_names = os.listdir(reactor_path)
                self._models[reactor_name] = {
                    name.replace('.pkl', ''): ModelLoader._SpecificModelLoader._load_saved_model(
                        os.path.join(reactor_path, name),
                        is_keras
                    )
                    for name in saved_models_names
                }

        @staticmethod
        def _load_saved_model(path, is_keras):
            if not is_keras:
                return pickle.load(open(path, 'rb'))
            return NNTemperaturesFeaturesExtractor(constants.NN_PERIOD, constants.NN_INPUT_TIME_INTERVALS_NUMBER,
                                                   constants.NN_OUTPUT_FEATURES_NUMBER, load_model(path))

        def find(self, reactor_name, model_name):
            if reactor_name in self._models:
                return self._models[reactor_name].get(model_name)
            return None

    def get_keras_models_loader(self):
        return ModelLoader._SpecificModelLoader(self._keras_weights_dir, True)

    def get_features_models_loader(self):
        return ModelLoader._SpecificModelLoader(self._features_models_dir)

    def get_prediction_models_loader(self):
        return ModelLoader._SpecificModelLoader(self._prediction_models_dir)


class ModelRepository:
    def __init__(self, reactors, settings):
        self._keras_models = {}
        self._features_models = {}
        self._prediction_models = {}
        self._sensors_index = {}
        models_loader = ModelLoader(settings)
        for reactor in reactors:
            reactor_name = reactor.get_name()
            self._sensors_index[reactor_name] = self._build_sensors_index(reactor)
            self._keras_models[reactor_name] = self._build_models_dict(models_loader.get_keras_weights_loader(),
                                                                       reactor)
            self._features_models[reactor_name] = self._build_models_dict(models_loader.get_features_models_loader(),
                                                                          reactor)
            self._prediction_models[reactor_name] = self._build_models_dict(
                models_loader.get_prediction_models_loader(),
                reactor
            )

    def _get_sensor_model(self, reactor_name, sensor, model_type):
        if model_type == 'features':
            models = self._features_models
        elif model_type == 'prediction':
            models = self._prediction_models
        elif model_type == 'keras':
            models = self._keras_models
        else:
            raise ValueError('model type must be \"features\" or \"prediction\"\
             or \"keras\"'.format(str(reactor_name)))
        if reactor_name not in models:
            raise ValueError('no reactor with name {}'.format(str(reactor_name)))
        reactor_model, plates_models = models[reactor_name]
        plate_name = self._sensors_index[reactor_name][sensor]
        plate_model, sensors_models = plates_models[plate_name]
        sensor_model = sensors_models[sensor]
        if sensor_model is not None:
            return sensor_model
        if plate_model is not None:
            return plate_model
        if reactor_model is not None:
            return reactor_model
        raise exceptions.MissingModel('no {} model for sensor {} in reactor {} found'.format(model_type, sensor,
                                                                                             reactor_name))

    def get_sensor_keras_model(self, reactor_name, sensor):
        return self._get_sensor_model(reactor_name, sensor, 'keras')

    def get_sensor_features_model(self, reactor_name, sensor):
        return self._get_sensor_model(reactor_name, sensor, 'features')

    def get_sensor_prediction_model(self, reactor_name, sensor):
        return self._get_sensor_model(reactor_name, sensor, 'prediction')

    @staticmethod
    def _build_models_dict(models_loader, reactor):
        reactor_name = reactor.get_name()
        reactor_model = models_loader.find(reactor_name, reactor_name)
        plates_models = {}
        for plate in reactor.get_all_plates():
            plate_name = plate.get_name()
            plate_model = models_loader.find(reactor_name, plate_name)
            sensors = plate.get_sensor_list()
            sensors_models = {sensor: models_loader.find(reactor_name, sensor) for sensor in sensors}
            plates_models[plate_name] = (plate_model, sensors_models)
        return reactor_model, plates_models

    @staticmethod
    def _build_sensors_index(reactor):
        sensors_index = {}
        for plate in reactor.get_all_plates():
            for sensor in plate.get_sensor_list():
                sensors_index[sensor] = plate.get_name()
        return sensors_index
