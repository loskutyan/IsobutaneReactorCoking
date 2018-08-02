import json
import os
import pickle

from domain.reactor_schema import IsobutaneReactor, ReactorPlate


class ReactorsDao:
    def __init__(self):
        path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(path, "../resources/dict/reactors.json")
        self._reactors_dict = {reactor_name: IsobutaneReactor(reactor_name,
                                                              [ReactorPlate(plate_name, sensors_config)
                                                               for plate_name, sensors_config in plates_config])
                               for reactor_name, plates_config in json.load(path).items()}

    def find_reactor_name(self, sensor_id):
        for reactor_name, reactor in self._reactors_dict.items():
            if reactor.find_plate_name(sensor_id) is not None:
                return reactor_name
        raise ValueError('no sensor {} in reactors found'.format(str(sensor_id)))

    def find(self, reactor_name):
        reactor = self._reactors_dict.get(reactor_name)
        if reactor is None:
            raise ValueError('no reactor with name {}'.format(str(reactor_name)))
        return reactor

    def find_all(self):
        return self._reactors_dict.values()


class ModelsDao:
    FEATURES_MODELS_PATH = "../resources/pickled_features_models"
    PREDICTION_MODELS_PATH = "../resources/pickled_prediction_models"

    class _SpecificModelsDao:
        def __init__(self, relative_path):
            path = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(path, relative_path)
            reactor_names = os.listdir(path)
            self._models = {}
            for reactor_name in reactor_names:
                reactor_path = os.path.join(path, reactor_name)
                pickled_models_names = os.listdir(reactor_path)
                self._models[reactor_name] = {name: pickle.load(open(os.path.join(reactor_path, name), 'rb'))
                                              for name in pickled_models_names}

        def find(self, reactor_name, model_name):
            if reactor_name in self._models:
                return self._models[reactor_name].get(model_name)
            return None

    @staticmethod
    def get_features_models_dao():
        return ModelsDao._SpecificModelsDao(ModelsDao.FEATURES_MODELS_PATH)

    @staticmethod
    def get_prediction_models_dao():
        return ModelsDao._SpecificModelsDao(ModelsDao.PREDICTION_MODELS_PATH)


class Dao:
    def __init__(self):
        self._features_models_dao = ModelsDao.get_features_models_dao()
        self._prediction_models_dao = ModelsDao.get_prediction_models_dao()
        self._reactors_dao = ReactorsDao()

    def get_features_models_dao(self):
        return self._features_models_dao

    def get_prediction_models_dao(self):
        return self._prediction_models_dao

    def get_reactors_dao(self):
        return self._reactors_dao
