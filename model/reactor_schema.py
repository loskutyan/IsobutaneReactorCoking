from functools import reduce

import numpy as np


class ReactorPlate:
    ANGLES_ORDER = [0, 90, 180, 270]

    def __init__(self, name, sensors_config):
        self._name = name
        self._sensors_config = sensors_config

    def get_angle_array(self, sensor_id):
        if sensor_id is None:
            raise ValueError('sensor id must be a string')
        if sensor_id not in self._sensors_config:
            raise ValueError('no sensor with name {} on plate {}'.format(str(sensor_id), self._name))
        result = np.zeros(len(ReactorPlate.ANGLES_ORDER), dtype='int8')
        result[self._sensors_config.index(sensor_id)] = 1
        return result

    def get_sensor_list(self):
        return [x for x in self._sensors_config if x is not None]

    def get_sensors_number(self):
        return sum([x is not None for x in self._sensors_config])

    def get_name(self):
        return self._name


class IsobutaneReactor:
    def __init__(self, plate_list):
        self._plates = {plate.get_name(): plate for plate in plate_list}
        self._plates_order = [plate.get_name() for plate in plate_list]
        self._sensors_number = sum([plate.get_sensor_number() for plate in self._plates])

    def find_plate_name(self, sensor_id):
        for plate_name, plate in self._plates.items():
            if sensor_id in plate.get_sensor_list():
                return plate_name
        raise ValueError('No plate with sensor {} in reactor found'.format(str(sensor_id)))

    def get_plate(self, plate_name):
        plate = self._plates.get(plate_name)
        if plate is None:
            raise ValueError('no plate with name {}'.format(str(plate_name)))
        return plate

    def get_plate_name_above(self, plate_name):
        if plate_name not in self._plates_order:
            raise ValueError('no plate with name {}'.format(str(plate_name)))
        plate_idx = self._plates_order.index(plate_name)
        return self._plates_order[plate_idx - 1] if plate_idx > 0 else None

    def get_plate_name_below(self, plate_name):
        if plate_name not in self._plates_order:
            raise ValueError('no plate with name {}'.format(str(plate_name)))
        plate_idx = self._plates_order.index(plate_name)
        return self._plates_order[plate_idx + 1] if plate_idx + 1 < len(self._plates_order) else None

    def get_sensor_list(self):
        return reduce(lambda x, y: x + y, [plate.get_sensor_list() for plate in self._plates])

    def get_sensors_number(self):
        return self._sensors_number

    def exclude_sensors(self, sensor_list):
        return self
