import pandas as pd
import pymysql
import sqlalchemy
from sqlalchemy.pool import NullPool

import constants

pymysql.install_as_MySQLdb()


class SQLSource:
    DBAPI_DICT = {
        'mysql': 'mysqldb',
        'mssql': 'pymssql'
    }

    DATETIME_CONVERTERS = {
        'mysql': lambda str_dt: 'CONVERT(\'{}\', datetime)'.format(str_dt),
        'mssql': lambda str_dt: 'CONVERT(datetime, \'{}\', 120)'.format(str_dt)
    }

    def __init__(self, params, datetime_col):
        self._db_type = params['db_type']
        if self._db_type not in SQLSource.DBAPI_DICT:
            raise ValueError('database type {} is not provided\nUse one of the following types: {}'.format(
                self._db_type, str(list(SQLSource.DBAPI_DICT.keys()))))

        self._db_name = params['database']
        self._datetime_col = datetime_col

        engine_config = SQLSource._build_engine_config(params['db_type'], params['username'], params['password'],
                                                       params['hostname'], params['port'], params['database'])
        self._engine = sqlalchemy.create_engine(engine_config, poolclass=NullPool)

    @staticmethod
    def _build_engine_config(db_type, username, password, hostname, port, db_name):
        dbapi = SQLSource.DBAPI_DICT[db_type]
        prefix = '+'.join([db_type, dbapi]) if dbapi else db_type
        auth = ':'.join([username, password]) if username else None
        address = ':'.join([hostname, port]) if port else hostname
        return '{}://{}/{}'.format(prefix, '@'.join([auth, address]) if auth else address, db_name)

    def get_data_since(self, table, datetime=None, allow_equality=True):
        query = 'SELECT * FROM {}'.format(table)
        if datetime is not None:
            inequality = '>=' if allow_equality else '>'
            query += ' WHERE {} {} {}'.format(self._datetime_col, inequality,
                                              SQLSource.DATETIME_CONVERTERS[self._db_type](str(datetime)))
            query += ';'
        connection = self._engine.connect()
        result = pd.read_sql(query, connection, index_col=self._datetime_col)
        connection.close()
        return result

    def find_last_datetime(self, table):
        query = 'SELECT MAX({}) from {};'.format(self._datetime_col, table)
        result = self._engine.execute(query).fetchone()[0]
        if result is None:
            return constants.MIN_DATETIME
        return result

    def write_new_data(self, table, data):
        connection = self._engine.connect()
        if data.shape[0] > 0:
            data.to_sql(table, connection, if_exists='append')
        connection.close()
        return
