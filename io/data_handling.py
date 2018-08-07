import pandas as pd

from io.source import SQLSource


class InputDataHandler:
    TABLE_NAMES = {
        'catalyst_analysis': 'cat',
        'out_gas_analysis': 'out',
        'smoke_gas_analysis': 'smoke',
        'temperatures': 'temps'
    }

    def __init__(self, settings):
        source_params = settings.get_input()
        self._source = SQLSource(source_params)
        self._temperatures_data = None
        self._analysis_data = None

    def get_temperatures(self, since_date=None):
        if self._temperatures_data is None:
            self._temperatures_data = self._source.get_data_since(InputDataHandler.TABLE_NAMES['temperatures'],
                                                                  since_date)
        return self._temperatures_data

    def get_analysis(self, since_date=None):
        analysis_data_list = []
        if self._analysis_data is None:
            for table_type, table_name in InputDataHandler.TABLE_NAMES.items():
                if table_type == 'temperatures':
                    continue
                analysis_data_list.append(self._source.get_data_since(InputDataHandler.TABLE_NAMES[table_type],
                                                                      since_date))
            self._analysis_data = pd.concat(analysis_data_list, axis=1, sort=True, join='outer')
        return self._analysis_data
