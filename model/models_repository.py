import os
import pickle

from keras.models import load_model

import constants
import exceptions
from features.features_extraction import NNTemperaturesFeaturesExtractor


class ModelLoader:
    KERAS_MODELS_ENDING = '.nn'
    PICKLE_ENDING = '.pkl'

    def __init__(self, settings, reactor_names):
        self._reactor_names = reactor_names
        self._keras_weights_dir = settings.get_keras_weights()['dir']
        self._features_models_dir = settings.get_features_models()['dir']
        self._prediction_models_dir = settings.get_prediction_models()['dir']

    @staticmethod
    def _filter_files_by_ending(filenames):
        return [name for name in filenames
                if name.endswith(ModelLoader.KERAS_MODELS_ENDING) or name.endswith(ModelLoader.PICKLE_ENDING)]

    @staticmethod
    def _remove_ending(filename):
        return filename.replace(ModelLoader.KERAS_MODELS_ENDING, '').replace(ModelLoader.PICKLE_ENDING, '')

    class _SpecificModelLoader:
        def __init__(self, path, reactor_names, is_keras=False):
            found_reactor_names = set(os.listdir(path)).intersection(reactor_names)
            self._models = {}
            for reactor_name in found_reactor_names:
                reactor_path = os.path.join(path, reactor_name)
                saved_models_names = ModelLoader._filter_files_by_ending(os.listdir(reactor_path))
                self._models[reactor_name] = {
                    ModelLoader._remove_ending(name): ModelLoader._SpecificModelLoader._load_saved_model(
                        os.path.join(reactor_path, name),
                        is_keras
                    )
                    for name in saved_models_names
                }

        @staticmethod
        def _load_saved_model(path, is_keras):
            if not is_keras:
                return pickle.load(file=open(path, 'rb'))
            return NNTemperaturesFeaturesExtractor(constants.NN_PERIOD, constants.NN_INPUT_TIME_INTERVALS_NUMBER,
                                                   constants.NN_OUTPUT_FEATURES_NUMBER, load_model(path))

        def find(self, reactor_name, model_name):
            if reactor_name in self._models:
                return self._models[reactor_name].get(model_name)
            return None

    def get_keras_models_loader(self):
        return ModelLoader._SpecificModelLoader(self._keras_weights_dir, self._reactor_names, True)

    def get_features_models_loader(self):
        return ModelLoader._SpecificModelLoader(self._features_models_dir, self._reactor_names)

    def get_prediction_models_loader(self):
        return ModelLoader._SpecificModelLoader(self._prediction_models_dir, self._reactor_names)


class ModelRepository:
    def __init__(self, reactors, settings):
        self._keras_models = {}
        self._features_models = {}
        self._prediction_models = {}
        self._sensors_index = {}
        models_loader = ModelLoader(settings, {reactor.get_name() for reactor in reactors})
        for reactor in reactors:
            reactor_name = reactor.get_name()
            self._sensors_index[reactor_name] = self._build_sensors_index(reactor)
            self._keras_models[reactor_name] = self._build_models_dict(models_loader.get_keras_models_loader(),
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
