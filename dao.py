import json
import os

from domain.reactor_schema import IsobutaneReactor, ReactorPlate


class ReactorsDao:
    def __init__(self):
        path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(path, "./resources/dict/reactors.json")
        with open(path, 'r') as f:
            self._reactors_dict = {reactor_name: IsobutaneReactor(reactor_name,
                                                                  {plate_name: ReactorPlate(plate_name, sensors_config)
                                                                   for plate_name, sensors_config in
                                                                   plates_configs.items()},
                                                                  plates_numbers)
                                   for reactor_name, [plates_numbers, plates_configs] in json.load(f).items()}

    def find_reactor_name(self, sensor_id):
        for reactor_name, reactor in self._reactors_dict.items():
            if reactor.find_plate_number(sensor_id) is not None:
                return reactor_name
        raise ValueError('no sensor {} in reactors found'.format(str(sensor_id)))

    def find(self, reactor_name):
        reactor = self._reactors_dict.get(reactor_name)
        if reactor is None:
            raise ValueError('no reactor with name {}'.format(str(reactor_name)))
        return reactor

    def find_all(self):
        return self._reactors_dict.values()


class ChemicalAnalysisTagsDao:
    def __init__(self):
        path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(path, "./resources/dict/chemical_analysis_tags.json")
        with open(path, 'r') as f:
            self._tags_dict = json.load(f)

    def findall(self):
        return dict(self._tags_dict)


class TemperaturesTagsDao:
    def __init__(self):
        path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(path, "./resources/dict/temperature_sensors_tags.json")
        with open(path, 'r') as f:
            self._tags_dict = json.load(f)

    def findall(self):
        return dict(self._tags_dict)


class Dao:
    def __init__(self):
        self._reactors_dao = ReactorsDao()
        self._chemical_analysis_tags_dao = ChemicalAnalysisTagsDao()
        self._temperatures_tags_dao = TemperaturesTagsDao()

    def get_reactors_dao(self):
        return self._reactors_dao

    def get_chemical_analysis_tags_dao(self):
        return self._chemical_analysis_tags_dao

    def get_temperatures_tags_dao(self):
        return self._temperatures_tags_dao
