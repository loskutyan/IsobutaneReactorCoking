import sys

import pandas as pd

import constants
from dao import Dao
from data_processing import DataPreprocessor, DataPostprocessor
from datasource.data_handling import InputDataHandler, OutputDataHandler
from features.features_extraction import FeaturesExtractor
from model.models_repository import ModelRepository
from settings import Settings


def main(argv):
    reactor_name_to_predict = argv[0]
    settings_path = argv[1]
    bad_sensors = argv[2]
    dao = Dao()
    reactors = dao.get_reactors_dao().findall()
    settings = Settings(settings_path)
    models_repo = ModelRepository(reactors, settings)
    preprocessor = DataPreprocessor(dao)
    reactor = dao.get_reactors_dao().find(reactor_name_to_predict).exclude_sensors(bad_sensors)
    sensor_list = reactor.get_sensor_list()

    input_data_handler = InputDataHandler(settings)
    output_data_handler = OutputDataHandler(settings)

    last_output_datetime = output_data_handler.find_last_datetime()
    since_temperatures_datetime = None
    if last_output_datetime != constants.MIN_DATETIME:
        since_temperatures_datetime = last_output_datetime - constants.TEMPERATURES_HISTORY

    raw_temps = input_data_handler.get_temperatures(since_datetime=since_temperatures_datetime)
    raw_chemical = input_data_handler.get_analysis(since_datetime=last_output_datetime)
    temps = preprocessor.process_temperatures(reactor_name_to_predict, raw_temps)
    chemical = preprocessor.process_analysis(reactor_name_to_predict, raw_chemical)
    predictions = pd.DataFrame()

    for sensor_id in sensor_list:
        nn_extractor = models_repo.get_sensor_keras_model(reactor_name_to_predict, sensor_id)
        trends_extractor, = models_repo.get_sensor_features_model(reactor_name_to_predict, sensor_id)
        features_extractor = FeaturesExtractor(nn_extractor, trends_extractor)
        predictions_dict = {}
        models = models_repo.get_sensor_prediction_model(reactor_name_to_predict, sensor_id)
        features = features_extractor.extract(temps, chemical, sensor_id, reactor)
        # maybe some features postprocessing
        for horizon, model in models:
            predictions_dict['{}:{}'.format(sensor_id, horizon)] = model.predict(features)

        predictions = predictions.merge(pd.DataFrame(predictions_dict, index=features.index),
                                        left_index=True, right_index=True, how='outer')

    postprocessor = DataPostprocessor(reactor)
    predictions_renamed = postprocessor.process_predictions(predictions)
    temps_renamed = postprocessor.process_temperatures(temps)
    output_data_handler.update_predictions_and_statistics(predictions_renamed, temps_renamed)


if __name__ == '__main__':
    main(sys.argv[1:])
