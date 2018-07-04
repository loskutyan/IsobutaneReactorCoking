import json

from model.reactor_schema import IsobutaneReactor


class ReactorsDao:
    def __init__(self, path):
        self._reactors = {reactor_name: IsobutaneReactor(plates_config)
                          for reactor_name, plates_config in json.load(path).items()}

    def find_reactor_name(self, sensor_id):
        for reactor_name, reactor in self._reactors.items():
            if reactor.find_plate_name(sensor_id) is not None:
                return reactor_name
        return None

    def get_reactor(self, reactor_name):
        reactor = self._reactors.get(reactor_name)
        if reactor is None:
            raise ValueError('no reactor with name {}'.format(str(reactor_name)))
        return reactor
