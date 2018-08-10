import json
import os

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


class ChemicalAnalysisTagsDao:
    def __init__(self):
        path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(path, "../resources/dict/chemical_analysis_tags.json")
        self._tags_dict = json.load(path)

    def find_all(self):
        return dict(self._tags_dict)


class TemperaturesTagsDao:
    def __init__(self):
        path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(path, "../resources/dict/temperatures_tags.json")
        self._tags_dict = json.load(path)

    def find_all(self):
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
