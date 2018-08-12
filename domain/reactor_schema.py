from functools import reduce

import numpy as np


class ReactorPlate:
    def __init__(self, name, sensors_config):
        self._name = name
        self._sensors_config = sensors_config

    def get_angle_array(self, sensor_id):
        if sensor_id is None:
            raise ValueError('sensor id must be a string')
        if sensor_id not in self._sensors_config:
            raise ValueError('no sensor with name {} on plate {}'.format(str(sensor_id), self._name))
        result = np.zeros(len(self._sensors_config), dtype='int8')
        result[len(self._sensors_config) - self._sensors_config.index(sensor_id) - 1] = 1
        return result

    def get_sensor_list(self):
        return [x for x in self._sensors_config if x is not None]

    def get_sensors_number(self):
        return sum([x is not None for x in self._sensors_config])

    def get_positions_number(self):
        return len(self._sensors_config)

    def get_name(self):
        return self._name


class IsobutaneReactor:
    def __init__(self, name, plates, plates_order_numbers):
        self._name = name
        self._plates = {int(plate_num): plates[plate_name] for plate_num, plate_name in plates_order_numbers.items()}
        self._plates_order = sorted(self._plates.keys(), reverse=True)

    def find_plate_number(self, sensor_id):
        for plate_number, plate in self._plates.items():
            if sensor_id in plate.get_sensor_list():
                return plate_number
        raise ValueError('No plate with sensor {} in reactor found'.format(str(sensor_id)))

    def get_plate(self, plate_number):
        plate = self._plates.get(plate_number)
        if plate is None:
            raise ValueError('plate number {} was not set up'.format(str(plate_number)))
        return plate

    def get_plate_number_above(self, plate_number):
        if plate_number not in self._plates:
            raise ValueError('plate number {} was not set up'.format(str(plate_number)))
        plate_idx = self._plates_order.index(plate_number)
        return self._plates_order[plate_idx - 1] if plate_idx > 0 else None

    def get_plate_number_below(self, plate_number):
        if plate_number not in self._plates:
            raise ValueError('plate number {} was not set up'.format(str(plate_number)))
        plate_idx = self._plates_order.index(plate_number)
        return self._plates_order[plate_idx + 1] if plate_idx + 1 < len(self._plates_order) else None

    def get_name(self):
        return self._name

    def get_sensor_list(self):
        return reduce(lambda x, y: x + y, [plate.get_sensor_list() for plate in self._plates])

    def get_all_plates(self):
        return self._plates.values()

    def exclude_sensors(self, sensor_list):
        return self
