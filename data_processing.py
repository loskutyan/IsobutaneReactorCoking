class DataPreprocessor:
    def __init__(self, dao):
        self._analysis_tags = dao.get_chemical_analysis_tags_dao().findall()
        self._temperatures_tags = dao.get_temperatures_tags_dao().findall()

    @staticmethod
    def _collect_tags_data(reactor_name, tags, data, tags_type):
        if reactor_name not in tags:
            raise ValueError('no {} tags for reactor {}'.format(tags_type, reactor_name))
        reactor_tags = tags[reactor_name]
        return data[list(reactor_tags.keys())].rename(reactor_tags).dropna(how='all')

    def process_analysis(self, reactor_name, data):
        return DataPreprocessor._collect_tags_data(reactor_name, self._analysis_tags, data, 'analysis').interpolate()

    def process_temperatures(self, reactor_name, data):
        return DataPreprocessor._collect_tags_data(reactor_name, self._temperatures_tags, data, 'temperatures')


class DataPostprocessor:
    def __init__(self, reactor):
        self._reactor = reactor

    def process_predictions(self, data):
        new_columns = {}
        for col in data.columns:
            sensor_id, horizon = col.split(':')
            plate_num = self._reactor.find_plate_number(sensor_id)
            sensor_num = self._reactor.get_plate(plate_num).find_sensor_number(sensor_id)
            new_columns[col] = '{}:{}:{}'.format(str(plate_num), str(sensor_num), horizon)
        return data.rename(columns=new_columns)
