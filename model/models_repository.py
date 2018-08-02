import exceptions


class ModelRepository:
    def __init__(self, dao):
        self._features_models = {}
        self._prediction_models = {}
        self._sensors_index = {}
        for reactor in dao.get_reactors_dao().find_all():
            reactor_name = reactor.get_name()
            self._sensors_index[reactor_name] = self._build_sensors_index(reactor)
            self._features_models[reactor_name] = self._build_models_dict(dao.get_features_models_dao(), reactor)
            self._prediction_models[reactor_name] = self._build_models_dict(dao.get_prediction_models_dao(), reactor)

    def _get_sensor_model(self, reactor_name, sensor, model_type):
        if model_type == 'features':
            models = self._features_models
        elif model_type == 'prediction':
            models = self._prediction_models
        else:
            raise ValueError('model type must be \"features\" or \"prediction\"'.format(str(reactor_name)))
        if reactor_name not in models:
            raise ValueError('no reactor with name {}'.format(str(reactor_name)))
        reactor_model, plates_models = models[reactor_name]
        found_model = None
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

    def get_sensor_features_model(self, reactor_name, sensor):
        return self._get_sensor_model(reactor_name, sensor, 'features')

    def get_sensor_prediction_model(self, reactor_name, sensor):
        return self._get_sensor_model(reactor_name, sensor, 'prediction')

    @staticmethod
    def _build_models_dict(models_dao, reactor):
        reactor_name = reactor.get_name()
        reactor_model = models_dao.find(reactor_name, reactor_name)
        plates_models = {}
        for plate in reactor.get_all_plates():
            plate_name = plate.get_name()
            plate_model = models_dao.find(reactor_name, plate_name)
            sensors = plate.get_sensor_list()
            sensors_models = {sensor: models_dao.find(reactor_name, sensor) for sensor in sensors}
            plates_models[plate_name] = (plate_model, sensors_models)
        return reactor_model, plates_models

    @staticmethod
    def _build_sensors_index(reactor):
        sensors_index = {}
        for plate in reactor.get_all_plates():
            for sensor in plate.get_sensor_list():
                sensors_index[sensor] = plate.get_name()
        return sensors_index
