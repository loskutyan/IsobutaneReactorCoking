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

    # move to features postprocessor
    excluded_features = ['Бутадиен-1,3, %', 'Массовая доля суммы углеводородов С5 и выше, %']
    features_order = ['Массовая доля CrO3, %', 'Массовая доля кокса, %', 'Объёмная доля кислорода, %',
                      'Объёмная доля СО2, %', 'Водород, %', 'Водород, %_coef', 'Водород, %_intercept', 'Азот N2, %',
                      'Азот N2, %_coef', 'Азот N2, %_intercept', 'Окись углерода, %', 'Окись углерода, %_coef',
                      'Окись углерода, %_intercept', 'Метан, %', 'Метан, %_coef', 'Метан, %_intercept',
                      'Сумма этан+этилен, %', 'Сумма этан+этилен, %_coef', 'Сумма этан+этилен, %_intercept',
                      'Двуокись углерода, %', 'Двуокись углерода, %_coef', 'Двуокись углерода, %_intercept',
                      'Сумма углеводородов С3, %', 'Сумма углеводородов С3, %_coef',
                      'Сумма углеводородов С3, %_intercept', 'Изобутан, %', 'Изобутан, %_coef', 'Изобутан, %_intercept',
                      'н-Бутан, %', 'н-Бутан, %_coef', 'н-Бутан, %_intercept', 'Бутен1+изобутилен, %',
                      'Бутен1+изобутилен, %_coef', 'Бутен1+изобутилен, %_intercept', 'Сумма бутиленов, %',
                      'Сумма бутиленов, %_coef', 'Сумма бутиленов, %_intercept', 'Бутадиен-1,3, %_coef',
                      'Бутадиен-1,3, %_intercept', 'Массовая доля суммы углеводородов С5 и выше, %_coef',
                      'Массовая доля суммы углеводородов С5 и выше, %_intercept', 'Температура_0', 'Температура_1',
                      'Температура_2', 'Температура_3', 'Температура_4', 'duration', 'delta_top', 'delta_bot', 'angle1',
                      'angle2', 'angle3', 'angle4']

    for sensor_id in sensor_list:
        nn_extractor = models_repo.get_sensor_keras_model(reactor_name_to_predict, sensor_id)
        trends_extractor, = models_repo.get_sensor_features_model(reactor_name_to_predict, sensor_id)
        features_extractor = FeaturesExtractor(nn_extractor, trends_extractor, excluded_features)
        predictions_dict = {}
        models = models_repo.get_sensor_prediction_model(reactor_name_to_predict, sensor_id)
        features = features_extractor.extract(temps, chemical, sensor_id, reactor)[features_order]
        # maybe some features postprocessing
        for horizon, model in models:
            predictions_dict['{}:{}'.format(sensor_id, horizon)] = model.predict_proba(features)[:, 1]

        predictions = predictions.merge(pd.DataFrame(predictions_dict, index=features.index),
                                        left_index=True, right_index=True, how='outer')

    postprocessor = DataPostprocessor(reactor)
    predictions_renamed = postprocessor.process_predictions(predictions)
    temps_renamed = postprocessor.process_temperatures(temps)
    output_data_handler.update_predictions_and_statistics(predictions_renamed, temps_renamed)


if __name__ == '__main__':
    main(sys.argv[1:])
