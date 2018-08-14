import configparser


class Settings:
    def __init__(self, path):
        config = configparser.ConfigParser()
        config.read(path)
        self._input = dict(config.items('INPUT'))
        self._output = dict(config.items('OUTPUT'))
        self._keras_weights = dict(config.items('KERAS WEIGHTS'))
        self._features_models = dict(config.items('FEATURES MODELS'))
        self._prediction_models = dict(config.items('PREDICTION MODELS'))

    def get_input(self):
        return dict(self._input)

    def get_output(self):
        return dict(self._output)

    def get_keras_weights(self):
        return dict(self._keras_weights)

    def get_features_models(self):
        return dict(self._features_models)

    def get_prediction_models(self):
        return dict(self._prediction_models)
