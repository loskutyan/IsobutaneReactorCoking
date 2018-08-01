import os
import pickle


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

        def get_model_by_name(self, reactor_name, model_name):
            if reactor_name in self._models:
                return self._models[reactor_name].get(model_name)
            return None

    @staticmethod
    def get_features_models_dao():
        return ModelsDao._SpecificModelsDao(ModelsDao.FEATURES_MODELS_PATH)

    @staticmethod
    def get_prediction_models_dao():
        return ModelsDao._SpecificModelsDao(ModelsDao.PREDICTION_MODELS_PATH)
