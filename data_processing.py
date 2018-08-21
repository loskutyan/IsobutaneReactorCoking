import constants


class DataPreprocessor:
    def __init__(self, dao):
        self._analysis_tags = dao.get_chemical_analysis_tags_dao().findall()
        self._temperatures_tags = dao.get_temperatures_tags_dao().findall()

    @staticmethod
    def _collect_tags_data(reactor_name, tags, data, tags_type):
        if reactor_name not in tags:
            raise ValueError('no {} tags for reactor {}'.format(tags_type, reactor_name))
        reactor_tags = tags[reactor_name]
        result = data[list(reactor_tags.keys())].rename(columns=reactor_tags).dropna(how='all')
        return result.reindex(result.index.rename(constants.MODEL_DATETIME_COLUMN))

    def process_analysis(self, reactor_name, data):
        return DataPreprocessor._collect_tags_data(reactor_name, self._analysis_tags, data, 'analysis').interpolate()

    def process_temperatures(self, reactor_name, data):
        return DataPreprocessor._collect_tags_data(reactor_name, self._temperatures_tags, data, 'temperatures')


class DataPostprocessor:
    def __init__(self, reactor):
        self._reactor = reactor

    def _convert_sensor_id(self, sensor_id):
        plate_num = self._reactor.find_plate_number(sensor_id)
        sensor_num = self._reactor.get_plate(plate_num).find_sensor_number(sensor_id)
        return '{}:{}'.format(str(plate_num), str(sensor_num))

    def process_predictions(self, data):
        new_columns = {}
        for col in data.columns:
            sensor_id, horizon = col.split(':')
            new_columns[col] = '{}:{}'.format(self._convert_sensor_id(sensor_id), horizon)
        return data.rename(columns=new_columns)\
            .reindex(data.index.rename(constants.OUTPUT_DATETIME_COLUMN))\
            .rolling(constants.PREDICTION_SMOOTHING_PERIOD).mean()

    def process_temperatures(self, data):
        return data.rename(columns=self._convert_sensor_id).reindex(data.index.rename(constants.OUTPUT_DATETIME_COLUMN))
