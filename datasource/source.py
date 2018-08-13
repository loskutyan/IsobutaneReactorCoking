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
        self._db_name = params['database']
        self._datetime_col = datetime_col
        if self._db_type not in SQLSource.DBAPI_DICT:
            raise ValueError('database type {} is not provided\nUse one of the following types: {}'.format(
                self._db_type, str(list(SQLSource.DBAPI_DICT.keys()))))
        dbapi = SQLSource.DBAPI_DICT[self._db_type]
        prefix = '+'.join([self._db_type, dbapi]) if len(dbapi) > 0 else self._db_type
        connection_config = '{}://{}:{}@{}/{}'.format(prefix, params['username'], params['password'],
                                                      params['hostname'], params['database'])
        self._engine = sqlalchemy.create_engine(connection_config, poolclass=NullPool)
        self._table_sizes = None
        self._update_table_sizes()

    def _update_table_sizes(self):
        connection = self._engine.connect()
        query = 'SELECT table_name, table_rows FROM INFORMATION_SCHEMA.TABLES ' \
                'WHERE TABLE_SCHEMA = \'{}\';'.format(self._db_name)
        self._table_sizes = pd.read_sql(query, connection, index_col='TABLE_NAME')['TABLE_ROWS'].to_dict()
        connection.close()

    def is_empty(self, table):
        if table in self._table_sizes:
            return self._table_sizes[table] == 0
        raise ValueError('no table {} in database {}'.format(table, self._db_name))

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
        if self.is_empty(table):
            return constants.MIN_DATETIME
        query = 'SELECT MAX({}) from {};'.format(self._datetime_col, table)
        return self._engine.execute(query).fetchone()[0]

    def write_new_data(self, table, data):
        connection = self._engine.connect()
        if data.shape[0] > 0:
            data.to_sql(table, connection, if_exists='append')
        connection.close()
        return
