import configparser


class Settings:
    def __init__(self, path):
        config = configparser.ConfigParser()
        config.read(path)
        self._input = config.items('INPUT')
        self._output = config.items('OUTPUT')
        self._features_models = config.items('FEATURES MODELS')
        self._prediction_models = config.items('PREDICTION MODELS')

    def get_input(self):
        return self._input

    def get_output(self):
        return self._output

    def get_features_models(self):
        return self._features_models

    def get_prediction_models(self):
        return self._prediction_models
