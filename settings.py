import configparser


class Settings:
    def __init__(self, path):
        config = configparser.ConfigParser()
        config.read(path)
        reactor_params = dict(config.items('REACTOR'))
        excluded_sensors = reactor_params['exclude_sensors']
        self._reactor_name = reactor_params['name']
        self._excluded_sensors = excluded_sensors.split(',') if excluded_sensors else []
        self._input = config.items('INPUT')
        self._input_tables = config.items('INPUT TABLES')
        self._output = config.items('OUTPUT')
        self._output_tables = config.items('OUTPUT TABLES')
        self._keras_weights = config.items('KERAS WEIGHTS')
        self._features_models = config.items('FEATURES MODELS')
        self._prediction_models = config.items('PREDICTION MODELS')

    def get_reactor_name(self):
        return self._reactor_name

    def get_excluded_sensors(self):
        return self._excluded_sensors

    def get_input(self):
        return dict(self._input)

    def get_input_tables(self):
        return dict(self._input_tables)

    def get_output(self):
        return dict(self._output)

    def get_output_tables(self):
        return dict(self._output_tables)

    def get_keras_weights(self):
        return dict(self._keras_weights)

    def get_features_models(self):
        return dict(self._features_models)

    def get_prediction_models(self):
        return dict(self._prediction_models)
